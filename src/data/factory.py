import os
import torch as tc
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.datasets import CIFAR10, ImageFolder, SVHN
from src.data.idbh import IDBH

from torch.utils.data import random_split, Subset

from src.utils.printer import dprint

class TinyImageNet(ImageFolder):
    PATH = 'tiny-imagenet-200'
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        path = os.path.join(self.PATH, 'train' if train else 'val')
        root = os.path.join(root, path)
        super().__init__(root, transform, target_transform)
        
DATASETS = {'TIN' : TinyImageNet,
            'CIFAR10': CIFAR10,
            'SVHN': SVHN}

class CropShift(tc.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)
        
    def sample_top(self, x, y):
        x = tc.randint(0, x+1, (1,)).item()
        y = tc.randint(0, y+1, (1,)).item()
        return x, y
            
    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = tc.randint(self.low, self.high, (1,)).item()
        
        w, h = F.get_image_size(img)
        crop_x = tc.randint(0, strength+1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)
        
        img = F.crop(img, top_y, top_x, crop_h, crop_w)
        img = F.pad(img, padding=[crop_x, crop_y], fill=0)
        
        top_x, top_y = self.sample_top(crop_x, crop_y)
        
        return F.crop(img, top_y, top_x, h, w)


def fetch_dataset(dataset, root, train=True, cropshift=(0, 9), idbh='cb', split=False, download=False):
    assert dataset in DATASETS
    
    # hyper-parameter report
    head = 'Training Set' if train else 'Test Set'
    dprint(head, dataset=dataset)

    if train:
        augment = T.Compose([
            T.RandomHorizontalFlip(),
            CropShift(*cropshift),
            IDBH(idbh),
            T.ToTensor()
        ])
    else:
        augment = T.ToTensor()
    
    dataset = DATASETS[dataset](root, train, download=download, transform=augment)

    if split > 1:
        total = len(dataset)
        chunk = total // split
        split = [chunk for i in range(split-1)]
        split = [0] + split + [total-sum(split)]
        split = [sum(split[:i+1]) for i, _ in enumerate(split)]
        indices = list(range(len(dataset)))
        dataset = [Subset(dataset, indices[s:split[i+1]]) for i, s in enumerate(split[:-1])]

    return dataset
