import torch
from torch import nn 
from torch.distributions import MultivariateNormal
import os 
import sys

import torch
import torch.nn as nn
import numpy as np

from my_util.constant_names import * 
from my_util.dataset_utils import *

import pickle 

def pickle_save(file, item):
    with open(file, 'wb') as fp:
        pickle.dump(item, fp)

def pickle_load(file):
    with open (file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def init_gaussians(cfg, method_name, reparam, num_classes, dtype=torch.float32):

        mu_arr = []
        cov_lower_arr = []
        scale_tril_arr = []

        dim_lower = cfg["dim_lower"]

        training_algorithm = cfg["val"]["gmm"]["training_algorithm"]
        full_cov =  cfg["val"]["gmm"]["fpi_info"]["full_cov"]
        if method_name in ["one", "one_conditioned_on_images"]:
            for i in range(num_classes):
                mu = torch.zeros(dim_lower, dtype=dtype)
                cov_lower = torch.ones(dim_lower, dtype=dtype)

                if ("fpi" in training_algorithm) and (full_cov): 
                    cov_lower = torch.diag_embed(cov_lower)
                    scale_tril = torch.linalg.cholesky(cov_lower)
                else:
                    scale_tril = torch.sqrt(cov_lower)
                
                mu_arr.append(mu) 
                cov_lower_arr.append(cov_lower) 
                scale_tril_arr.append(scale_tril)

        elif method_name in ["one_per_class_all_in_center"]:
            for i in range(num_classes):
                mu = torch.zeros(dim_lower, dtype=dtype)
                cov_lower = torch.ones(dim_lower, dtype=dtype)
                cov_lower *= reparam.get_init_cov_value()
                
                if ("fpi" in training_algorithm) and (full_cov): 
                    cov_lower = torch.diag_embed(cov_lower)
                    scale_tril = torch.linalg.cholesky(cov_lower)
                else:
                    scale_tril = torch.sqrt(cov_lower)
                
                mu_arr.append(mu) 
                cov_lower_arr.append(cov_lower) 
                scale_tril_arr.append(scale_tril)


        elif method_name in GAUSSIAN_CHECKPOINTS_METHODS:
            print("Loading the gaussians")
            print("Loading the gaussians")

            
            gaussians_load_path = cfg["gaussians_load_path"]
            if gaussians_load_path == "":
                sys.exit("No Checkpoint given for the gaussians, expecting one")

            ckpt = torch.load(gaussians_load_path, map_location=torch.device("cpu"))
            state_dict = ckpt["state_dict"]

            mu_arr = []
            cov_lower_arr = []
            scale_tril_arr = []

            for key, val in state_dict.items():
                if ("flow.mu_arr" in key) and not ("ema" in key):
                    mu_arr.append(val)
                if ("flow.cov_lower_arr" in key) and not ("ema" in key):
                    cov_lower_arr.append(val)
                if ("flow.scale_tril_arr" in key) and not ("ema" in key):
                    scale_tril_arr.append(val)

        else:
            print(method_name)
            sys.exit("Wrong name for the gaussian method")

        # disable gradients from the gaussians
        for i in range(len(mu_arr)):
            mu_arr[i] = nn.Parameter(mu_arr[i], requires_grad=False)
            cov_lower_arr[i] = nn.Parameter(cov_lower_arr[i], requires_grad=False)
            scale_tril_arr[i] = nn.Parameter(scale_tril_arr[i], requires_grad=False)

        return mu_arr, cov_lower_arr, scale_tril_arr

        

class GaussianAdam(nn.Module):
    def __init__(self, size, beta1=0.9, beta2=0.999, eps=1e-8, active=True):
        super().__init__()
        self.__dict__.update(locals())
        
        self.t = 0
        self.register_buffer(name='m_t', tensor=torch.zeros(self.size))
        self.register_buffer(name='v_t', tensor=torch.zeros(self.size))

    def reset(self):
        self.t = 0
        self.m_t[:] = 0
        self.v_t[:] = 0
    
    def get_grad(self, gt):
        if not self.active:
            return gt

        self.t += 1
        
        self.m_t = self.beta1 * self.m_t  + ((1 - self.beta1) * (gt))
        self.v_t = self.beta2 * self.v_t  + ((1 - self.beta2) * (gt**2))
        m_t_hat = self.m_t / (1 - (self.beta1**self.t))
        v_t_hat = self.v_t / (1 - (self.beta2**self.t))
        
        adam_grad = (m_t_hat) / (torch.sqrt(v_t_hat) + self.eps)
        return adam_grad
    


class ManualMultiLinearScheduler(nn.Module):
    def __init__(self, lrs, steps):
        super().__init__()
                
        self.register_buffer(name="current_epoch", tensor=torch.tensor([0]))
        self.register_buffer(name="current_iteration", tensor=torch.tensor([0]))
        
        lr_arr = []
        for i in range(len(lrs) - 1):
            lr_step_arr = torch.linspace(lrs[i], lrs[i+1], steps[i+1] - steps[i])
            lr_arr.append(lr_step_arr)
        lr_arr = torch.cat(lr_arr)

        self.lr_arr = lr_arr
        #self.register_buffer(name="lr_arr", tensor=lr_arr) 
    
    def step(self):
        self.current_epoch = self.current_epoch + 1

    def step_iteration(self):
        self.current_iteration = self.current_iteration + 1

    def get_current_epoch(self):
        return self.current_epoch.item()

    def get_current_iteration(self):
        return self.current_iteration.item()
    
    def get_lr(self):
        return self.lr_arr[self.current_epoch.item()]
    

class ManualMultiGeometricScheduler(nn.Module):
    def __init__(self, lrs, steps):
        super().__init__()
                
        self.register_buffer(name="current_epoch", tensor=torch.tensor([0]))
        self.register_buffer(name="current_iteration", tensor=torch.tensor([0]))
       
        lr_arr = []
        for i in range(len(lrs) - 1):
            lr_beg = lrs[i]
            lr_end = lrs[i+1]
            iters = int((steps[i+1] - steps[i]))
            if iters == 1:
                q = (lr_end / lr_beg)**(1/(iters))
            elif iters > 1:
                q = (lr_end / lr_beg)**(1/(iters-1))
            lr_step_arr = [lr_beg * (q**i) for i in range(iters)]
            
            lr_arr.append(torch.tensor(lr_step_arr))
        lr_arr = torch.cat(lr_arr)
        self.lr_arr = lr_arr

    def step(self):
        self.current_epoch = self.current_epoch + 1

    def step_iteration(self):
        self.current_iteration = self.current_iteration + 1

    def get_current_epoch(self):
        return self.current_epoch.item()

    def get_current_iteration(self):
        return self.current_iteration.item()
    
    def get_lr(self):
        return self.lr_arr[self.current_epoch.item()]
        