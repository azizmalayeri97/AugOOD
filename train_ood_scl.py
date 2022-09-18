from __future__ import print_function

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
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument('--known-classes', default='0 1 2 4 5 9', type=str)
    parser.add_argument('--temp', default=0.1, type=float)
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
model_type = args.model_type
model_name = model_type+data+list_to_str(known_classes)
epochs = args.epochs

print('Known classes:', known_classes)
print('Model name:', model_name)
print('Num classes:', num_classes)


mixup_args = {

    'mixup_alpha': 0.8,
    'cutmix_alpha': 1.0,
    'prob': 1.0,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0.1,
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

transform_test = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize(224, interpolation= transforms.InterpolationMode.BICUBIC ),
                                      transforms_channel,
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])

transform_train = transforms.Compose([transforms.AutoAugment(),
                                      transforms.ToTensor(),
                                      transforms.Resize(224, interpolation= transforms.InterpolationMode.BICUBIC ),
                                      transforms_channel,
                                      #transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                                      #transforms.RandomHorizontalFlip(),
                                      #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                      #transforms.RandomGrayscale(p=0.2),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])

data_path =  'root/'

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

if data == 'mnist':

    train_data = MNIST(root=data_path, train=True, download=True, transform=TwoCropTransform(transform_train))
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
    train_data = SVHN(root=data_path, split='train', download=True, transform=TwoCropTransform(transform_train))
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
    train_data = CIFAR10(root=data_path, train=True, download=True, transform=TwoCropTransform(transform_train))
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
    
    train_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/train/', transform=TwoCropTransform(transform_train))
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

def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model

if model_type == 'deitti':
    model = deit_tiny_patch16_224(num_classes=num_classes)
    ckpt_address = 'deitt16_224_fractal10k_lr3e-4_100ep.pth'
    
    ckpt = torch.load(ckpt_address, map_location='cpu')
    ckpt_model = ckpt['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f'Remove key [{k}] from pretrained checkpoint')
            del ckpt_model[k]
    
    model.load_state_dict(ckpt_model, strict=False)

elif model_type == 'vitb32':
    model = vit_base_patch32_224(num_classes=num_classes)
    pretraining = 'imagenet'
    ckpt_address = "vitb_p32_224.npz"
    
    load_checkpoint(model, ckpt_address)
    

head = (model.head).cuda()
model.head = nn.Sequential()
model.cuda()
#print(f'Checkpoint was loaded from {ckpt_address}\n')

opt_args.lr = opt_args.lr*batch_size/512

optimizer = create_optimizer(opt_args, model=nn.Sequential(*(list(model.children()))))
print(f'Optimizer: \n{optimizer}\n')
scheduler = create_scheduler(scheduler_args ,optimizer=optimizer)
print(f'Scheduler: \n{scheduler}\n')

optimizer_head = create_optimizer(opt_args, model=head)
scheduler_head = create_scheduler(scheduler_args, optimizer=optimizer_head)

#criterion_class = SoftTargetCrossEntropy().cuda()
criterion_class = LabelSmoothingCrossEntropy(0.0).cuda()
print('SoftTargetCrossEntropy is used for criterion\n')

start_epoch = 1
if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler[0].load_state_dict(checkpoint['scheduler'])
        scheduler_head[0].load_state_dict(checkpoint['scheduler_head'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume was loaded from {resume}\n')

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


criterion = SupConLoss(temperature=args.temp)

for epoch in range(start_epoch, epochs+1):
    
    t1 = time.time()

    model.train()
    
    
    correct = 0
    total = 0

    for j, data in enumerate(train_loader):

        images = data[0]
        labels_orig = data[1].cuda(non_blocking=True)
        
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)
        
        if images[0].shape[0]%2 != 0:
            images[0], images[1], labels_orig = images[0][:-1], images[1][:-1], labels_orig[:-1]
            
            
        #a, labels = mixup_fn(images[0], labels_orig)
        _, labels = images[0], labels_orig

        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels_orig.shape[0]
        
        #if images.shape[0]%2 != 0:
                #images, labels_orig = images[:-1], labels_orig[:-1]
        
        
        #images, labels = mixup_fn(images, labels_orig)
        #images, labels = images, labels_orig
        
        #optimizer.zero_grad()
        #with torch.cuda.amp.autocast():

                #outputs = model(images)
                #loss = criterion(outputs, labels)
        #print(loss.item())
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            
            
            model_images = model(images)
            #features = F.normalize(model(images), dim=1)
            f1, f2 = torch.split(model_images, [bsz, bsz], dim=0)
            f3 = F.normalize(f1, dim=1)
            f4 = F.normalize(f2, dim=1)
            features = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)
            loss = criterion(features, labels_orig)
        
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        optimizer_head.zero_grad()
        with torch.cuda.amp.autocast():
        
            out = head(f1.detach())
            loss_head = criterion_class(out, labels)
        
        #optimizer_head.zero_grad()
        loss_head.backward()
        optimizer_head.step()

        _, pred = torch.max(out, dim=1)
        correct += (pred == labels_orig).sum()
        total += labels_orig.size(0)

    print('Epoch: ', epoch, 'Training acc: ', (correct/total).item(), ' lr: ', optimizer.param_groups[0]['lr'])
    
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for data in test_loader:

            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)

            #outputs = F.normalize(model(images), dim=1)
            outputs = model(images)

            _, pred = torch.max(head(outputs), dim=1)
            correct += (pred == labels).sum()
            total += labels.size(0)
        
        t2 = time.time()
        
        print('Epoch: ', epoch, 'Test acc: ', (correct/total).item(), 'Epoch Time:', t2-t1, 's')

        scheduler[0].step(epoch)
        scheduler_head[0].step(epoch)

        if epoch % save_epoch_freq == 0:
              save_path = 'outputs/contrastive-' + model_name + '-' + str(epoch) + '.pth'
              torch.save({
                  'model': model.state_dict(),
                  'head':head.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler[0].state_dict(),
                  'scheduler_head': scheduler_head[0].state_dict(),
                  'epoch': epoch
              }, save_path)