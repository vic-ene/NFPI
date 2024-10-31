import torch
from torch import nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import sys

import copy
import os
import matplotlib.pyplot as plt

import itertools

import numpy as np 

from tqdm import tqdm

import lightning as L 
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from my_util.dataset_utils import *

from my_util.constant_names import *

from omegaconf import OmegaConf, open_dict, DictConfig

import pickle
from filelock import FileLock

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
        
        
def add_variable_to_hydra_cfg(cfg, var, val):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        OmegaConf.update(cfg, var, val)
    OmegaConf.set_struct(cfg, False)
    

def get_label_encoder_and_update_cfg(cfg):
    # Very important, we seed here so that the random labels encoding for each class will always be the same
    pl.seed_everything(cfg["seed"])

    label_encoder = None

    ch = cfg["dataset_info"][cfg["dataset_name"]]["ch"]
    h  = cfg["dataset_info"][cfg["dataset_name"]]["h"]
    w  = cfg["dataset_info"][cfg["dataset_name"]]["w"]
     
    if cfg["val"]["gmm"]["gaussian_method_name"] in ENCODING_METHODS:
        label_encoder_name = cfg["label_encoder"]["label_encoder_name"]

        if label_encoder_name == "distanced_pixels_encoding_below_image":
            label_encoder = DistancedPixelsEncodingBelowImage(**cfg["label_encoder"]["distanced_pixels_encoding_below_image"], shape=(ch, h, w))
        else:
            sys.exit("do not recognise the label encoder ... ")

        ch, h, w = label_encoder.get_ch_h_w(ch, h, w)

    dim = ch * h * w
    dim_lower = int(dim/2**(cfg["num_levels"]-1))

    if cfg["level_method"]["choice"] in ["all_levels"]:
        dim_lower = dim
    elif cfg["level_method"]["choice"] in ["all_levels_idx"]:
        level_idx = cfg["level_method"]["level_idx"]
        dim_lower = int(dim / 2**(cfg["num_levels"]-1))
        dim_lower = dim_lower * 2**(cfg["num_levels"] - level_idx)
    elif cfg["level_method"]["choice"] in [SPCB]:
        dim_lower = int(dim / 2**(cfg["num_levels"]-1))
    else:
        sys.exit(f"This level method is not known")

    print(f"Dimension with {cfg['level_method']['choice']}", dim_lower, dim)
    print(f"Dimension with {cfg['level_method']['choice']}", dim_lower, dim)
    print(f"Dimension with {cfg['level_method']['choice']}", dim_lower, dim)

    add_variable_to_hydra_cfg(cfg, "ch", ch)
    add_variable_to_hydra_cfg(cfg, "h", h)
    add_variable_to_hydra_cfg(cfg, "w", w)
    add_variable_to_hydra_cfg(cfg, "dim", dim)
    add_variable_to_hydra_cfg(cfg, "dim_lower", dim_lower)

    print("Information regarding the images:", cfg["ch"], cfg["h"], cfg["w"], cfg["dim"])

    return label_encoder


def get_noise_fn_with_name(noise_fn_name, shape, mean, std):
    if noise_fn_name == "uniform":
        return get_uniform_noise_tensor(shape, mean, std)
    if noise_fn_name == "gaussian":
        return get_gaussian_noise_tensor(shape, mean, std)

def get_uniform_noise_tensor(shape, mean, std):
     return mean + torch.rand(shape) * std

def get_gaussian_noise_tensor(shape, mean, std):
    return mean + torch.randn(shape) * std

def cat_3dtensor_and_label_encoding(x, encoding):
    return torch.cat((x, encoding), dim=1)


def bring_back_to_tensor_range(arr):
    min_val = torch.min(arr)
    arr -= min_val
    max_val = torch.max(arr)
    arr /= max_val
    return arr

class DistancedPixelsEncodingBelowImage(ABC):
    def __init__(self, noise_fn_name, mean, std, pxh, k, iters, shape, path=""):
        super().__init__()

        self.__dict__.update(locals())

        def get_distanced_noise_at_once(k, dim, iters=100):
            arr = torch.stack([torch.rand(dim) for i in range(k)])
            #arr = arr.to(torch.float64)

            def dist_grad_full(idx, arr):
                c = arr[idx]
                arr_c = torch.cat((arr[:idx], arr[(idx+1):]))
                diff = -(arr_c - c)
                dist = torch.sqrt(((diff*diff).sum(dim=1)))
                dist = dist.unsqueeze(0).repeat(diff.shape[1], 1)
                dist = torch.transpose(dist, 0, 1)
                grad = ((-diff)/(dist)).sum(dim=0)
                return grad

            for iter_idx in tqdm(range(iters)):
                grad_arr = []
                for i in range(k):
                    grad = dist_grad_full(i, arr) # Distance loss term
                    grad += 1 * 2 * arr[i] # Regularization term
                    grad_arr.append(grad)
                grad_arr = torch.stack(grad_arr)
                arr -= grad_arr
            #arr = arr.to(torch.float32)
                
            return arr

        if self.path == "":
            ch, h, w = self.get_ch_h_w(*(shape))
            encoding = get_distanced_noise_at_once(k, ch*pxh*w, iters=iters)
            encoding = bring_back_to_tensor_range(encoding)
            encoding = encoding.reshape(k, ch, pxh, w)
            save_path = "encoding_1000k.pickle"
            pickle_save(save_path, encoding)
        else:
            encoding = pickle_safe_load(path)
        self.noise_encoding_arr = encoding


    def get_noise_encoding(self , idx):
        encoding = self.noise_encoding_arr[idx]
        return encoding
        
    def get_ch_h_w(self, ch, h, w):
        h += self.pxh
        return ch, h, w
        
    def encode_tensor(self, x, y, enc_value):
        return self.add_pixel_band_encoding_below_image_for_k_classes(
            x, y,
            self.noise_fn_name, self.mean, self.std,
            pxh=self.pxh, k=self.k, enc_value=enc_value
        )

    def get_encoding_label(self, x):
        return self.get_k_class_encoding_result(
            x, pxh=self.pxh, k=self.k
        )

    def remove_encoding_for_batch(self, x):
        if len(x.shape) < 4:
            print("A minibatch should be passed in")
            sys.exit(0)
         
        x = TF.crop(x, 0, 0, x.shape[3], x.shape[3])
        return x
    
    def preprocess_before_plotting_for_batch(self, x):
        if len(x.shape) < 4:
            print("A minibatch should be passed in")
            sys.exit(0)
            
        #x = TF.crop(x, 0, 0, x.shape[3], x.shape[3])
        return x
    
    def add_pixel_band_encoding_below_image_for_k_classes(
            self, 
            x, y, 
            noise_fn, mean, std,
            pxh=8, k=10, enc_value=1.0
        ):
        if y > (k-1):
            print("The label index should be smaller than the : (number_of_classes - 1")
            raise
        
        encoding = self.get_noise_encoding(y).to(x.device)
        # If we are running in test mode, we add random encoding instead
        if enc_value == 0:
            encoding = torch.rand_like(encoding)

        x =  torch.cat((x, encoding), dim=1)
        return x
    

    def get_k_class_encoding_result(self, x, pxh, k):   
        ch, w, h = x.shape
        
        encoding_portion = x[:, x.shape[1] - pxh:]
        diff = self.noise_encoding_arr.to(encoding_portion.device) - encoding_portion
        diff = torch.abs(diff)
        diff = diff.sum(dim=(1, 2, 3))
        label =  torch.argmin(diff)
        return label
    

    
def create_class_label_list(ch, h, w):
    # Add for 1000 classes
    zero = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    one = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]).repeat(ch, 1, 1)
    
    two = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    three = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    four = torch.tensor([
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ]).repeat(ch, 1, 1)
    
    five = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    six = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    seven = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ]).repeat(ch, 1, 1)
    
    eight = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ]).repeat(ch, 1, 1)
    
    nine = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ]).repeat(ch, 1, 1)
    
    padder_0_9 = nn.ZeroPad2d((0, w - 5, 0, 0))
    padder_10_99 = nn.ZeroPad2d((0, w - 10, 0, 0))
    padder_100_199 = nn.ZeroPad2d((0, w - 15, 0, 0))
    
    classes = [zero, one, two, three, four, five, six, seven, eight, nine]
    
    for i in range(1, 10):
        for j in range(0, 10):
            img = torch.cat((classes[i], classes[j]), dim=2)
            classes.append(img)
            # plt.imshow(torch.permute(img, (1, 2, 0)))
            # plt.plot()

    for l in range(1, 10):   
        for i in range(0, 10):
            for j in range(0, 10):
                img = torch.cat((classes[l], classes[i], classes[j]), dim=2)
                classes.append(img)
            
    for i in range(0, 10):
        classes[i] = padder_0_9(classes[i]) 
        
    for i in range(10, 100):
        classes[i] = padder_10_99(classes[i]) 
        
    for i in range(100, 200):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(200, 300):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(300, 400):
        classes[i] = padder_100_199(classes[i]) 
    
    for i in range(400, 500):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(500, 600):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(600, 700):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(700, 800):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(800, 900):
        classes[i] = padder_100_199(classes[i]) 

    for i in range(900, 1000):
        classes[i] = padder_100_199(classes[i]) 
        
    return classes


def add_class_label_into_image(x, y, classes):  
    class_encoding = classes[y]
    class_encoding = class_encoding.to(x.device)
    
    # This should only happen if encoding is along the channel
    if x.shape[0] != class_encoding.shape[0]:
        first_channel = class_encoding[:1, :, :].clone()
        class_encoding = torch.cat((class_encoding, first_channel), dim=0)
        

    x = torch.cat((x, class_encoding), dim=1)
 
    return x
    
    
def append_label_results_to_batch(x, encoding_class, classes):        
    arr = [add_class_label_into_image(elt, encoding_class.get_encoding_label(elt), classes) 
           for elt in x]
    arr = torch.stack(arr)

    return arr



