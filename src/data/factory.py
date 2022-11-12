import os
import torch as tc
import torchvision.transforms as T
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



def fetch_dataset(dataset, root, train=True, idbh='cifar10-weak', split=False, download=False):
    assert dataset in DATASETS
    
    # hyper-parameter report
    head = 'Training Set' if train else 'Test Set'
    dprint(head, dataset=dataset)

    if train:
        augment = IDBH(idbh)
    else:
        augment = T.ToTensor()

    if dataset == 'SVHN':
        train = 'train' if train else 'test'
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
