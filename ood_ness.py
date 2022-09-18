import torch
import timm
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torch.nn.functional as F
import sklearn.metrics as metrics
import random
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-classes', default=20, type=int)
    return parser.parse_args()
args = get_args()

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

fix_random_seed(args.seed)

data_path =  'root/'
num_classes = args.num_classes
prefix = ''
batch_size = 64
num_workers = 2
augmentation_list = [('noise', 'low'), ('noise', 'mid'), ('noise', 'high'), ('noise', 'rand'),
                      ('autoaugment', 'low'), ('autoaugment', 'mid'), ('autoaugment', 'high'), ('autoaugment', 'rand'), 
                      ('RandAugment', 'low'), ('RandAugment', 'mid'), ('RandAugment', 'high'), ('RandAugment', 'rand'), 
                      ('colorjitter', 'low'), ('colorjitter', 'mid'), ('colorjitter', 'high'),('colorjitter', 'rand'), 
                      ('cutout', 'low'), ('cutout', 'mid'), ('cutout', 'high'),  ('cutout', 'rand'),
                      ('rotate', 'low'), ('rotate', 'mid'), ('rotate', 'high'), ('rotate', 'rand'),
                      ('mixup', 'low'), ('mixup', 'mid'), ('mixup', 'high'), ('mixup', 'rand'), 
                      ('flip', 'low'), ('flip', 'mid'), ('flip', 'high'), ('flip', 'rand'),
                      ('perm', 'low'), ('perm', 'mid'), ('perm', 'high'), ('perm', 'rand'), 
                      ('fgsm', 'low'), ('fgsm', 'mid'), ('fgsm', 'high'), ('fgsm', 'rand'), 
                      ('pgd', 'low'), ('pgd', 'mid'), ('pgd', 'high'),  ('pgd', 'rand'), 
                      ]


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

model = deit_tiny_patch16_224(num_classes=num_classes)
model.cuda()

def convert_list_to_str(lst):
    string = ''
    for elem in lst:
        string += str(elem)
    return string

def augment(images, augmnetation):
    
    if augmnetation == 'noise':
        augmented_images = images + 1.0*torch.randn_like(images)
    
    return augmented_images

#####################

import torchvision.transforms.functional as TF

mu = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
std =  torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def onehot(targets, num_classes):

    #assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes, device='cuda').scatter_(1, targets.view(-1, 1), 1)

def mixup(inputs, targets, num_classes, alpha=2):
    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s)).cuda()
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)
    weight = weight.view(s, 1, 1, 1)
    inputs = weight*x1 + (1-weight)*x2
    weight = weight.view(s, 1)
    targets = weight*y1 + (1-weight)*y2
    return inputs 

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = torch.ones_like(img, device='cuda')*1.0

        for n in range(self.n_holes):

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[:, :, y1: y2, x1: x2] = 0.0

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * mask

        return img

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters=10, restarts=1,
               norm="l_inf", early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def augment(images, labels, augmentation, level):

  augmneted_images = images*std+mu
  
  if augmentation == 'noise':
      
      if level == 'low':
          alpha = 0.1
      elif level == 'mid':
          alpha = 0.25  
      elif level == 'high':
          alpha = 0.5
      elif level == 'rand':
          alpha = random.uniform(0, 0.5)

      augmneted_images += alpha*torch.randn_like(augmneted_images)
  
  elif augmentation == 'rotate':

      if level == 'low':
          alpha = 15
      elif level == 'mid':
          alpha = 45
      elif level == 'high':
          alpha = 90
      elif level == 'rand':
          alpha = random.uniform(-90, 90)
      
      augmneted_images = TF.rotate(augmneted_images, alpha)
  
  elif augmentation == 'flip':
      
      if level == 'rand':
          rand = random.randint(0, 3)
          if rand == 1:
              level = 'low'
          elif rand == 2:
              level = 'mid'
          elif rand == 3:
              level = 'high'
          
          
      if level == 'low':
          augmneted_images = TF.hflip(augmneted_images)
      elif level == 'mid':
          augmneted_images = TF.vflip(augmneted_images)
      elif level == 'high':
          augmneted_images = TF.vflip(TF.hflip(augmneted_images))
            
  elif augmentation == 'perm':

      if level == 'rand':
          rand = random.randint(0, 3)
          if rand == 1:
              level = 'low'
          elif rand == 2:
              level = 'mid'
          elif rand == 3:
              level = 'high'
      
    
      if level == 'low':
          temp = torch.zeros_like(augmneted_images)
          width = augmneted_images.shape[-1]
          temp[:, :, :int(width/2), :int(width/2)], temp[:, :, int(width/2):, :int(width/2)] = augmneted_images[:, :, int(width/2):, :int(width/2)], augmneted_images[:, :, :int(width/2), :int(width/2)]
          augmneted_images = temp
          
      elif level == 'mid':
          temp = torch.zeros_like(augmneted_images)
          width = augmneted_images.shape[-1]
          temp[:, :, :, :int(width/2)], temp[:, :, :, int(width/2):] = augmneted_images[:, :, :, int(width/2):], augmneted_images[:, :, :, :int(width/2)]
          augmneted_images = temp
      
      elif level == 'high':

          temp = torch.zeros_like(augmneted_images)
          width = augmneted_images.shape[-1]  
          temp[:, :, :int(width/2), :int(width/2)], temp[:, :, :int(width/2), int(width/2):], temp[:, :, int(width/2):, :int(width/2)], temp[:, :, int(width/2):, int(width/2):] = augmneted_images[:, :, int(width/2):, int(width/2):], augmneted_images[:, :, int(width/2):, :int(width/2)], augmneted_images[:, :, :int(width/2), int(width/2):], augmneted_images[:, :, :int(width/2), :int(width/2)]
          augmneted_images = temp
    
  elif augmentation == 'colorjitter':
      if level == 'low':
          augmneted_images = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(augmneted_images)
      elif level == 'mid':
          augmneted_images = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)(augmneted_images)
      elif level == 'high':
          augmneted_images = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(augmneted_images)
      elif level == 'rand':
          augmneted_images = transforms.ColorJitter(brightness=random.uniform(0, 0.5), contrast=random.uniform(0, 0.5), saturation=random.uniform(0, 0.5), hue=random.uniform(0, 0.5))(augmneted_images)
  
  elif augmentation == 'mixup':
      if level == 'low':
          augmneted_images = mixup(augmneted_images, labels, num_classes=num_classes, alpha=0.1)
      elif level == 'mid':
          augmneted_images = mixup(augmneted_images, labels, num_classes=num_classes, alpha=0.5)
      elif level == 'high':
          augmneted_images = mixup(augmneted_images, labels, num_classes=num_classes, alpha=1.0)
      elif level == 'rand':
          augmneted_images = mixup(augmneted_images, labels, num_classes=num_classes, alpha=random.uniform(0, 1.0))
  
  elif augmentation == 'cutout':
      if level == 'low':
          augmneted_images = Cutout(n_holes=1, length=32)(augmneted_images)
      if level == 'mid':
          augmneted_images = Cutout(n_holes=2, length=32)(augmneted_images)
      elif level == 'high':
          augmneted_images = Cutout(n_holes=4, length=32)(augmneted_images)
      elif level == 'rand':
          augmneted_images = Cutout(n_holes=random.randint(1, 4), length=random.randint(16, 32))(augmneted_images)
  
  elif augmentation == 'fgsm':
      if level == 'low':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=2/255, alpha=2.5/255, attack_iters=1)
      elif level == 'mid':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=4/255, alpha=5/255, attack_iters=1)
      elif level == 'high':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=8/255, alpha=10/255, attack_iters=1)
      elif level == 'rand':
          rand = random.uniform(0, 8/255)
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=rand, alpha=1.25*rand, attack_iters=1)
  
  elif augmentation == 'pgd':
      if level == 'low':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=2/255, alpha=(2/255)/8, attack_iters=10)
      elif level == 'mid':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=4/255, alpha=(4/255)/8, attack_iters=10)
      elif level == 'high':
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=8/255, alpha=(8/255)/8, attack_iters=10)
      elif level == 'rand':
          rand = random.uniform(0, 8/255)
          augmneted_images += attack_pgd(model, augmneted_images, labels, epsilon=rand, alpha=rand/8, attack_iters=10)
  
  elif augmentation == 'autoaugment':
      if level == 'rand':
          rand = random.randint(1, 3)
          if rand == 1:
              level = 'low'
          elif rand == 2:
              level = 'mid'
          elif rand == 3:
              level = 'high'
              
      if level == 'low':
          t = transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.SVHN)
      elif level == 'mid':
          t = transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10)
      elif level == 'high':
          t = transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.IMAGENET)
      
      augmneted_images = t((255*augmneted_images).to(torch.uint8))/255
  
  elif augmentation == 'RandAugment':
      if level == 'rand':
          rand = random.randint(1, 3)
          if rand == 1:
              level = 'low'
          elif rand == 2:
              level = 'mid'
          elif rand == 3:
              level = 'high'
              
      if level == 'low':
          t = transforms.RandAugment(num_ops= 2, magnitude = 4)
      elif level == 'mid':
          t = transforms.RandAugment(num_ops= 4, magnitude = 8)
      elif level == 'high':
          t = transforms.RandAugment(num_ops= 8, magnitude = 16)
      
      augmneted_images = t((255*augmneted_images).to(torch.uint8))/255
  
  elif augmentation == 'cutout-colorjitter':
      augmneted_images = transforms.ColorJitter(brightness=random.uniform(0, 0.5), contrast=random.uniform(0, 0.5), saturation=random.uniform(0, 0.5), hue=random.uniform(0, 0.5))(augmneted_images)
      augmneted_images = Cutout(n_holes=random.randint(1, 4), length=random.randint(1, 32))(augmneted_images)
      
  augmneted_images = normalize(augmneted_images)
  return augmneted_images
  
#####################
criterion = nn.CrossEntropyLoss(reduction='sum')

for data in ['mnist', 'svhn', 'cifar10']:
    
    if data == 'mnist':
        transforms_channel = transforms.Lambda(lambda x : x.repeat(3, 1, 1))
    elif data == 'svhn' or data == 'cifar10' or data == 'tinyimagenet':
        transforms_channel = transforms.Lambda(lambda x : x)
    
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(224),
                                        transforms_channel,
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])
    
    for known_classes in [[0, 1, 2, 4, 5, 9], [0, 3, 5, 7, 8, 9], [0, 1, 5, 6, 7, 8], [3, 4, 5, 7, 8, 9], [0, 1, 2, 3, 7, 8]]:
           
        if data == 'mnist':
            resume = 'outputs/' + prefix + data + convert_list_to_str(known_classes) + '-20.pth'
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

            test_data = MNIST(root=data_path, train=True, download=False, transform=transform_test)
            
            idx = torch.zeros_like(test_data.targets)
            for j, one_class in enumerate(known_classes):
                idx = torch.logical_or(test_data.targets == one_class, idx)
                test_data.targets[test_data.targets == one_class] = j
            test_data.targets = test_data.targets[idx]
            test_data.data = test_data.data[idx]

        elif data == 'svhn':
            resume = 'outputs/' + prefix + data + convert_list_to_str(known_classes) + '-30.pth'
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            
            test_data = SVHN(root=data_path, split='train', download=False, transform=transform_test)
            
            idx = np.zeros_like(test_data.labels)
            for j, one_class in enumerate(known_classes):
                idx = np.logical_or(test_data.labels == one_class, idx)
                test_data.labels[test_data.labels == one_class] = j
            test_data.labels = test_data.labels[idx]
            test_data.data = test_data.data[idx]
        
        elif data == 'cifar10':
            resume = 'outputs/' + prefix + data + convert_list_to_str(known_classes) + '-40.pth'
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            
            test_data = CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
            
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
            resume = 'outputs/' + prefix +data + convert_list_to_str(known_classes) + '-100.pth'
            checkpoint = torch.load(resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            
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
        

        #with torch.no_grad():
        scores = []
        true = []
        loss_clean = 0
        for datas in test_loader:
            images = datas[0].cuda(non_blocking=True)
            labels = torch.zeros(images.shape[0])
            
            with torch.no_grad():
                outputs = model(images)
                loss_clean += criterion(outputs, datas[1].cuda(non_blocking=True))
                
                smax = F.softmax(outputs, dim=1)  
                #scores += (-torch.max(smax, 1)[0]).tolist()
                scores += torch.gather(-smax, 1, datas[1].cuda(non_blocking=True).unsqueeze(1)).view(-1).tolist()
                true += labels.tolist()
        
        for (augmentation, level) in augmentation_list:
            
            scores_ood = []
            true_ood = torch.ones(len(true)).tolist()
            loss_augmented = 0
            for datas in test_loader:
                images = datas[0].cuda(non_blocking=True)
                labels = datas[1].cuda(non_blocking=True)
                images_ood = augment(images, labels, augmentation, level)
                
                with torch.no_grad():
                    outputs = model(images_ood)
                    loss_augmented += criterion(outputs, labels)
                    
                    smax = F.softmax(outputs, dim=1)
                    #scores_ood += (-torch.max(smax, 1)[0]).tolist()
                    scores_ood += torch.gather(-smax, 1, labels.unsqueeze(1)).view(-1).tolist()
            
            ood2 = (loss_augmented/loss_clean).item()
            fpr, tpr, threshold = metrics.roc_curve(true+true_ood, scores+scores_ood)
            roc_auc = metrics.auc(fpr, tpr)
            
            print('Data:', data, 'Known classes:', convert_list_to_str(known_classes), 'Augmentation:', augmentation, 'Level:', level, 'Roc_AUC:', roc_auc, 'OOD:', ood2)
