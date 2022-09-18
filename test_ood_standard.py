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

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg, _create_vision_transformer
from timm.models.registry import register_model

from types import SimpleNamespace

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument("--known-classes", default='0 1 2 4 5 9', type=str)
    parser.add_argument("--epoch", default='100', type=str)
    parser.add_argument("--prefix", default='vitb32imagenet', type=str) #model_type + pretraining
    parser.add_argument("--model-type", default='vitb32', type=str)
    
    return parser.parse_args()

args = get_args()

def list_to_str(inputs):
    out = ''
    for element in inputs:
      out += str(element)
    return out

seed = args.seed
batch_size = args.batch_size
num_workers = args.num_workers
known_classes = [int(x) for x in args.known_classes.split(' ')]
num_classes = len(known_classes)
data = args.data
model_name = data+list_to_str(known_classes)
resume = 'outputs/' + args.prefix + model_name + '-' + args.epoch + '.pth'
model_type = args.model_type

print('Known classes:', known_classes)
print('Model name:', model_name)
print('Num classes:', num_classes)

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

fix_random_seed(seed)


if data == 'mnist':
    transforms_channel = transforms.Lambda(lambda x : x.repeat(3, 1, 1))

elif data == 'svhn' or data == 'cifar10' or data == 'tinyimagenet':
    transforms_channel = transforms.Lambda(lambda x : x)

transform_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(224),
                                      transforms_channel,
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])

data_path =  'root/'

if data == 'mnist':
    test_data = MNIST(root=data_path, train=False, download=True, transform=transform_test)
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = j
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]

elif data == 'svhn':
    test_data = SVHN(root=data_path, split='test', download=True, transform=transform_test)

    idx = np.zeros_like(test_data.labels)
    for j, one_class in enumerate(known_classes):
        idx = np.logical_or(test_data.labels == one_class, idx)
        test_data.labels[test_data.labels == one_class] = j
    test_data.labels = test_data.labels[idx]
    test_data.data = test_data.data[idx]

elif data == 'cifar10':
    
    test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    test_data.targets = torch.tensor(test_data.targets)
    test_data.data = torch.tensor(test_data.data)  
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = j
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]

    test_data.targets = test_data.targets.tolist()
    test_data.data = test_data.data.numpy()

elif data == 'tinyimagenet':
    from torchvision.datasets import ImageFolder
    
    test_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/test_/', transform=transform_test)

    new_targets = []
    new_samples = []
    for i in range(len(test_data.targets)):
        if test_data.targets[i] in known_classes:
            new_targets.append(known_classes.index(test_data.targets[i]))
            new_samples.append((test_data.samples[i][0], known_classes.index(test_data.targets[i])))
    
    test_data.targets = new_targets
    test_data.samples = new_samples

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


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
elif model_type == 'deitb':
    model = deit_base_patch16_224(num_classes=num_classes)
elif model_type == 'vitb32':
    model = vit_base_patch32_224(num_classes=num_classes)
elif model_type == 'vits32':
    model = vit_small_patch32_224(num_classes=num_classes)
elif model_type == 'vitti16':
    model = vit_tiny_patch16_224(num_classes=num_classes)
elif model_type == 'vitl32':
    model = vit_large_patch32_224(num_classes=num_classes)
    
model.cuda()

checkpoint = torch.load(resume, map_location='cpu')
model.load_state_dict(checkpoint['model'])
#print(f'Resume was loaded from {resume}\n')
    
    
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
    
    test_acc = (correct/total).item()
    print('Test acc: ', test_acc)
 

data = args.data

if data == 'mnist':
    test_data = MNIST(root=data_path, train=False, download=True, transform=transform_test)
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = 0
    test_data.targets[torch.logical_not(idx)] = 1

elif data == 'svhn':
    test_data = SVHN(root=data_path, split='test', download=True, transform=transform_test)

    idx = np.zeros_like(test_data.labels)
    for j, one_class in enumerate(known_classes):
        idx = np.logical_or(test_data.labels == one_class, idx)
        test_data.labels[test_data.labels == one_class] = 0
    test_data.labels[np.logical_not(idx)] = 1

elif data == 'cifar10':

    test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    test_data.targets = torch.tensor(test_data.targets)
    test_data.data = torch.tensor(test_data.data)  
    
    idx = torch.zeros_like(test_data.targets)
    for j, one_class in enumerate(known_classes):
        idx = torch.logical_or(test_data.targets == one_class, idx)
        test_data.targets[test_data.targets == one_class] = 0
    test_data.targets[torch.logical_not(idx)] = 1

    test_data.targets = test_data.targets.tolist()
    test_data.data = test_data.data.numpy()

elif data == 'tinyimagenet':
    from torchvision.datasets import ImageFolder
    
    test_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/test_/', transform=transform_test)

    new_targets = []
    new_samples = []
    for i in range(len(test_data.targets)):
        if test_data.targets[i] in known_classes:
            new_targets.append(0)
            new_samples.append((test_data.samples[i][0], 0))
        else:
            new_targets.append(1)
            new_samples.append((test_data.samples[i][0], 1))            
    
    test_data.targets = new_targets
    test_data.samples = new_samples


test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

with torch.no_grad():
    scores = []
    true = []
    for data in test_loader:
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        outputs = model(images)
        smax = F.softmax(outputs, dim=1)
        
        scores += (-torch.max(smax, 1)[0]).tolist()
        true += labels.tolist()

fpr, tpr, threshold = metrics.roc_curve(true, scores)
roc_auc = metrics.auc(fpr, tpr)

print('roc_auc: ', model_name, roc_auc)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Test acc = %0.3f' % test_acc)
plt.savefig('ROC_AUC/' + args.prefix + model_name + '-' + args.epoch + '.png')
plt.show()

