import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import torch.utils.data as data

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

from scipy.spatial import distance


from torchvision.datasets import MNIST, FashionMNIST,CIFAR10, CIFAR100

import os
import pickle
import sys 

import sys
import copy 

import matplotlib.pyplot as plt
import numpy as np 

import torch
import torchvision
from torch import nn
import torchvision.transforms as T
import PIL

from sklearn.model_selection import train_test_split

from my_util.constant_names import *
from PIL import Image

import pickle

def pickle_save(file, item):
    with open(file, 'wb') as fp:
        pickle.dump(item, fp)

def pickle_load(file):
    with open (file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def pickle_safe_load(file):
    with FileLock(f'{file}.lck'):
        with open(file, 'r+b') as pfile:
            data = pickle.load(pfile)
            return data

import math
import random

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])

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
    

class PosterizeIfSpecified(object):
    def __init__(self, bits, posterize):
        self.bits = bits
        self.posterize = posterize
        pass
        
    def __call__(self, tensor):
        if self.posterize:
            tensor = torchvision.transforms.functional.posterize(tensor, self.bits)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'  Applies T.ToPILImage() if it is not already a PIL.Image'
    


class ConvertToKbits(object):
    def __init__(self, bits, active):
        self.bits = bits
        self.bins = 2**(8 - bits)
        self.active = active
        
    def __call__(self, tensor):
        if self.active:
            if self.bits != 8:
                #tensor = torch.round(tensor/self.bins)*self.bins
                tensor = (tensor//self.bins)*self.bins
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={0}, std={self.std})'
    


class PadWithUniformNoiseIfSpecified(object):
    def __init__(self, active, padding=(2, 2, 2, 2), noise_scale=0.025):
        self.active=active
        self.padding=padding
        self.noise_scale=noise_scale
        
    def __call__(self, tensor):
        if self.active:
            tensor = pad_img_with_uniform_noise(tensor, self.padding, self.noise_scale)
            
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'Padding: {self.padding} Uniform Noise Scale: {self.noise_scale}'
    
    
def pad_img_with_uniform_noise(img, padding, noise_scale=1.0):
    ch, h, w = img.shape
    pl, pr, pt, pb = padding
    
    padded_data = torch.nn.functional.pad(img, padding, mode='constant', value=0).float()
    noise = torch.randn(padded_data.shape) * noise_scale

    mask = torch.ones_like(padded_data)
    mask[:, pl:(-pr), pt:(-pb)] = 0

    # Create a tensor for the noisy padded data
    noisy_padded_data = padded_data.clone()
    noisy_padded_data += noise * mask

    # Combine the noisy padded data with the original image
    noisy_data = torch.where(mask == 1, noisy_padded_data, padded_data)

    return noisy_data

# ------------------------------------------------------------------------------------------------------------------------


# Datasets
# ------------------------------------------------------------------------------------------------------------------------
def extract_data_and_targets_from_dataset(ds):
    if isinstance(ds, torch.utils.data.dataset.Subset):
        indices = ds.indices
        data = ds.dataset.data[indices]

        if isinstance(ds.dataset.targets, list):
            targets = torch.tensor(ds.dataset.targets)[indices].tolist()
        else:
            targets = ds.dataset.targets[indices]

    else:
        data = ds.data
        targets = ds.targets

    # Convert to tensor and permute because is numpy
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
        data = torch.permute(data, (0, 3, 1, 2))
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
        
    return data, targets

def stratified_data_targets_split(data, targets, ratio, seed):
    train_idx, valid_idx=train_test_split(
        np.arange(len(targets)),
        test_size=ratio,
        shuffle=True,
        random_state=seed,
        stratify=targets
    )
    
    data_train = data[train_idx]
    targets_train = targets[train_idx]
    
    data_val = data[valid_idx]
    targets_val = targets[valid_idx]
        
    return data_train, targets_train, data_val, targets_val

def extract_k_items_per_class(data, targets, k_items):
    n_classes = len(torch.unique(targets))
    print("this is k items", k_items, data.shape, targets.shape)
    print("this is k items", k_items, data.shape, targets.shape)
    print("this is k items", k_items, data.shape, targets.shape)
    class_indices = [(targets==i).nonzero(as_tuple=True)[0] for i in range(n_classes)]
    class_indices = [elt[torch.randperm(len(elt))] for elt in class_indices]
    class_indices = [elt[:k_items] for elt in class_indices]
    
    data_new = [data[idx] for idx in class_indices]
    targets_new = [targets[idx] for idx in class_indices]
    data_new = torch.cat(data_new)
    targets_new = torch.cat(targets_new)
    
    final_idx = torch.randperm(len(targets_new))
    data_new = data_new[final_idx]
    targets_new = targets_new[final_idx]
    
    return data_new, targets_new

def extract_data_from_ds_dict(ds_dict):
    if "data" in ds_dict.keys():
        data, targets = ds_dict["data"], ds_dict["targets"]
    else:
        data, targets = ds_dict["x"], ds_dict["y"]
    return data, targets


def do_torch_distributed_barrier():
    if not torch.backends.mps.is_available():
        torch.distributed.barrier()


# ------------------------------------------------------------------------------------------------------------------------

        

def load_vae(device, dtype):
    from diffusers.models import AutoencoderKL
    import time
    t1 = time.time()
    print("Started loading vae")
    #vae = AutoencoderKL.from_single_file(path).to(dtype=dtype)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir="../../Models/stabilityai/sd-vae-ft-mse", local_files_only=True).to(dtype=dtype)
    vae = vae.eval()
    vae.train = False
    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.to(device)
    t2 = time.time()
    print("Finished loading vae", t2 - t1)
    return vae


def right_range(tensor):
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor * 0.5) + 0.5
    return tensor

