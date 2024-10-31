import PIL
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from filelock import FileLock

import pickle




def pickle_save(file, item):
    with open(file, 'wb') as fp:
        pickle.dump(item, fp)

def pickle_safe_load(file):
    from filelock import FileLock
    with FileLock(f'{file}.lck'):
        with open(file, 'r+b') as pfile:
            data = pickle.load(pfile)
            return data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.__dict__.update(locals())
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform != None:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)
    

class AddUniformNoise(object):
    def __init__(self, mean=0., std=0.05, signed=False):
        self.std = std
        self.mean = mean
        self.signed = signed
        
    def __call__(self, tensor):
        noise = torch.rand(tensor.size()) * self.std + self.mean
        if self.signed:
            noise -= self.std/2
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={0}, std={self.std})'
    

class ToPILImageIfNotAlready(object):
    def __init__(self):
        pass
        
    def __call__(self, tensor):
        if not isinstance(tensor, PIL.Image.Image):
            tensor = T.ToPILImage()(tensor)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'  Applies T.ToPILImage() if it is not already a PIL.Image'