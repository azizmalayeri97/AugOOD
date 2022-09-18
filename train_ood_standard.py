import torch
import timm
import time
import os
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torch.nn.functional as F

from timm.data.mixup import Mixup

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg, _create_vision_transformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.optim.optim_factory import create_optimizer
from timm.scheduler import create_scheduler

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from types import SimpleNamespace
from timm.models.helpers import load_pretrained, load_checkpoint

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-epoch-freq', default=100, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument("--known-classes", default='0 1 2 4 5 9', type=str)
    parser.add_argument("--pretraining", default='imagenet', type=str)
    parser.add_argument("--model-type", default='vitb32', type=str)
    
    return parser.parse_args()


args = get_args()

def list_to_str(inputs):
    out = ''
    for element in inputs:
      out += str(element)
    return out

resume = None
seed = args.seed
batch_size = args.batch_size
num_workers = args.num_workers
known_classes = [int(x) for x in args.known_classes.split(' ')]
num_classes = len(known_classes)
save_epoch_freq = args.save_epoch_freq
data = args.data
model_name = data+list_to_str(known_classes)
epochs = args.epochs
pretraining = args.pretraining
model_type = args.model_type

print('Known classes:', known_classes)
print('Model name:', model_name)
print('Num classes:', num_classes)


mixup_args = {

    'mixup_alpha': 0.1,
    'cutmix_alpha': 0.0,
    'prob': 1.0,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0.0,
    'num_classes': 6
    }

scheduler_args = SimpleNamespace()
scheduler_args.epochs = epochs
scheduler_args.sched = 'cosine'
scheduler_args.lr_cycle_mul = 1.0
scheduler_args.min_lr = 1.0e-5
scheduler_args.decay_rate = 0.1
scheduler_args.warmup_lr = 1.0e-6
scheduler_args.warmup_epochs = 10
scheduler_args.lr_cycle_limit = 1
scheduler_args.seed = 0
scheduler_args.cooldown_epochs = 10

opt_args = SimpleNamespace()
opt_args.weight_decay = 1.0e-4
opt_args.lr = 0.01
opt_args.opt = 'momentum'
opt_args.momentum = 0.9

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

fix_random_seed(seed)

mixup_fn = Mixup(**mixup_args)

if data == 'mnist':
    transforms_channel = transforms.Lambda(lambda x : x.repeat(3, 1, 1))

elif data == 'svhn' or data == 'cifar10' or data == 'tinyimagenet':
    transforms_channel = transforms.Lambda(lambda x : x)

transform_test = transforms.Compose([#transforms.AutoAugment(),
                                      transforms.ToTensor(),
                                      transforms.Resize(224),
                                      transforms_channel,
                                      #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.4),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])

transform_train = transforms.Compose([transforms.AutoAugment(),
                                      transforms.ToTensor(),
                                      #transforms.RandomHorizontalFlip(),
                                      transforms.Resize(224),
                                      transforms_channel,
                                      #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.4),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])

data_path =  'root/'

if data == 'mnist':

    train_data = MNIST(root=data_path, train=True, download=True, transform=transform_train)
    test_data = MNIST(root=data_path, train=False, download=True, transform=transform_test)
    
    idx = torch.zeros_like(train_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(train_data.targets == one_class, idx)
        train_data.targets[train_data.targets == one_class] = j
    train_data.targets = train_data.targets[idx]
    train_data.data  = train_data.data[idx]
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = j
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]

elif data == 'svhn':
    train_data = SVHN(root=data_path, split='train', download=True, transform=transform_train)
    test_data = SVHN(root=data_path, split='test', download=True, transform=transform_test)
    
    idx = np.zeros_like(train_data.labels)
    for j, one_class in enumerate(known_classes):
        idx = np.logical_or(train_data.labels == one_class, idx)
        train_data.labels[train_data.labels == one_class] = j
    train_data.labels = train_data.labels[idx]
    train_data.data  = train_data.data[idx]
    
    idx = np.zeros_like(test_data.labels)
    for j, one_class in enumerate(known_classes):
        idx = np.logical_or(test_data.labels == one_class, idx)
        test_data.labels[test_data.labels == one_class] = j
    test_data.labels = test_data.labels[idx]
    test_data.data = test_data.data[idx]

elif data == 'cifar10':
    train_data = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    
    train_data.targets = torch.tensor(train_data.targets)
    train_data.data = torch.tensor(train_data.data)

    test_data.targets = torch.tensor(test_data.targets)
    test_data.data = torch.tensor(test_data.data)  

    idx = torch.zeros_like(train_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(train_data.targets == one_class, idx)
        train_data.targets[train_data.targets == one_class] = j
    train_data.targets = train_data.targets[idx]
    train_data.data  = train_data.data[idx]
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = j
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]
    
    train_data.targets = train_data.targets.tolist()
    train_data.data = train_data.data.numpy()

    test_data.targets = test_data.targets.tolist()
    test_data.data = test_data.data.numpy()

elif data == 'tinyimagenet':
    from torchvision.datasets import ImageFolder
    
    train_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/train/', transform=transform_train)
    test_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/test_/', transform=transform_test)
    
    for data_t in [train_data, test_data]:
        new_targets = []
        new_samples = []
        for i in range(len(data_t.targets)):
            if data_t.targets[i] in known_classes:
                new_targets.append(known_classes.index(data_t.targets[i]))
                new_samples.append((data_t.samples[i][0], known_classes.index(data_t.targets[i])))
        
        data_t.targets = new_targets
        data_t.samples = new_samples
    

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(len(train_loader), len(test_loader))

def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model
    
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model
    
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model

if model_type == 'deitti':
    model = deit_tiny_patch16_224(num_classes=num_classes)
    
    if pretraining == 'fractal10k':
        ckpt_address = "deitt16_224_fractal10k_lr3e-4_100ep.pth"
    elif pretraining == 'fractal1k':
        ckpt_address = "deitt16_224_fractal1k_lr3e-4_300ep.pth"
    elif pretraining == 'imagenet':
        ckpt_address = "deit_tiny_patch16_224-a1311bcf.pth"

elif model_type == 'deitb':
    model = deit_base_patch16_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "deit_base_patch16_224-b5f2ef4d.pth"

elif model_type == 'vitb16':
    model = vit_base_patch16_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "VITB_16_224.npz"

elif model_type == 'vitb32':
    model = vit_base_patch32_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "vitb_p32_224.npz"

elif model_type == 'vits32':
    model = vit_small_patch32_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "vits_p32_224.npz"

elif model_type == 'vitl32':
    model = vit_large_patch32_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "vitl_p32_224.npz"

elif model_type == 'vitti16':
    model = vit_tiny_patch16_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "vitti_p16_224.npz"

if 'deit' in model_type:
    ckpt = torch.load(ckpt_address, map_location='cpu')
    ckpt_model = ckpt['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f'Remove key [{k}] from pretrained checkpoint')
            del ckpt_model[k]
    
    model.load_state_dict(ckpt_model, strict=False)

elif 'vit' in model_type:
    load_checkpoint(model, ckpt_address)
    
model.cuda()

opt_args.lr = opt_args.lr*batch_size/512

optimizer = create_optimizer(opt_args, model=model)
print(f'Optimizer: \n{optimizer}\n')

scheduler = create_scheduler(scheduler_args ,optimizer=optimizer)
print(f'Scheduler: \n{scheduler}\n')

#criterion = SoftTargetCrossEntropy().cuda()
criterion = LabelSmoothingCrossEntropy(0.0).cuda()
print('SoftTargetCrossEntropy is used for criterion\n')

start_epoch = 1
if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler[0].load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume was loaded from {resume}\n')

for epoch in range(start_epoch, epochs+1):
    
    t1 = time.time()

    model.train()
    
    
    correct = 0
    total = 0

    for j, data in enumerate(train_loader):

        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

                outputs = model(images)
                loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum()
        total += labels.size(0)

    print('Epoch: ', epoch, 'Training acc: ', (correct/total).item(), ' lr: ', optimizer.param_groups[0]['lr'])
    
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for data in test_loader:

            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)

            outputs = model(images)

            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum()
            total += labels.size(0)
        
        t2 = time.time()
        
        print('Epoch: ', epoch, 'Test acc: ', (correct/total).item(), 'Epoch Time:', t2-t1, 's')

    
        scheduler[0].step(epoch)

        if epoch % save_epoch_freq == 0:
              save_path = 'outputs/' + model_type + pretraining + model_name + '-' + str(epoch) + '.pth'
              torch.save({
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler[0].state_dict(),
                  'epoch': epoch
              }, save_path)