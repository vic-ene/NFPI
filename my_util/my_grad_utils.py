import torch
from functools import partial
import sys
import numpy as np 

from my_util.logging_utils import *
from torch import nn

from my_util.my_plot_utils import *
from models.prior_loss import *

import copy


def get_min_max_mean(x):
    res = str(torch.min(x)) + " " +  str(torch.max(x)) + " " +  str(torch.mean(x))
    return res


def mvn_grad_pytorch_formula(mu, cov_lower, z, nll=True):
    """
    Assumes only the diagonal to avoid huge matrices and out of memory error on CUDA
    """
    
    # We return 0 as gradient if there are no elements in the array
    if len(z) == 0:
        grad_m = torch.zeros_like(mu).type_as(mu)
        grad_s = torch.zeros_like(mu).type_as(mu)
        return grad_m, grad_s
            
    # Calculate inverse and differences only once
    sigma_inv = 1 / cov_lower
    z_minus_mu = z - mu
    
    # Calculate gradients for the mean
    # Do not sum them yet to reuse for the gradient calculation in the covariance matrix
    grad_m = z_minus_mu * sigma_inv

    # Calculate gradients for the covariance matrix
    # It can be written as the element-wise square of the mu gradient minus sigma_inv
    grad_s = 0.5 * (grad_m * grad_m - sigma_inv)
    
    # Add the gradients
    grad_m = torch.sum(grad_m, dim=0)
    grad_s = torch.sum(grad_s, dim=0)
    
    # Multiply by -1 to get the correct sign takes into account the -log
    grad_m *= -1
    grad_s *= -1

    return grad_m, grad_s

def kl_div(mu_q, std_q, mu_p, std_p):
    """
    Computes the KL divergence between the two given variational distribution.\
    This computes KL(q||p), which is not symmetric. It quantifies how far is\
    The estimated distribution q from the true distribution of p.
    
    
    Taken from: https://discuss.pytorch.org/t/how-to-efficiently-compute-a-pairwise-kl-divergence-matrix-of-a-batch-of-gaussian-distributions/192519/3
    """
    k = mu_q.size(-1)
    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=-1)
    logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=-1)
    fs = torch.sum(torch.div(std_q**2, std_p**2), dim=-1) + torch.sum(
        torch.div(mu_diff_sq, std_p**2), dim=-1
    )
    kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
    return kl_divergence

@torch.enable_grad()
def get_kld_loss_and_grad_and_min_max(mu_params, std_params, chunk_size=100, min_max_mode="pairwise_sum_class_avg"):
    K = len(mu_params)
    D = len(mu_params[0])
    if chunk_size > K:
        chunk_size = K
    chunk_ratio = K // chunk_size
    mu = torch.stack([elt.detach().clone() for elt in mu_params])
    std = torch.stack([elt.detach().clone() for elt in std_params])
    mu.requires_grad=True
    std.requires_grad=True


    kld_loss_detach_arr = []
    for c1 in range(chunk_ratio):
        c1_start = c1 * chunk_size
        c1_end = (c1 + 1) * chunk_size
        temp_kld_loss_detach_arr = []
        for c2 in range(chunk_ratio):
            c2_start = c2 * chunk_size
            c2_end = (c2 + 1) * chunk_size

            mu_p1 = mu[c1_start:c1_end]
            std_p1 = std[c1_start:c1_end]
            
            mu_p2 = mu[c2_start:c2_end]
            std_p2 = std[c2_start:c2_end]

            kld_loss = -kl_div(mu_p1.unsqueeze(1), std_p1.unsqueeze(1), mu_p2.unsqueeze(0), std_p2.unsqueeze(0))
            kld_loss_detach = kld_loss.detach().clone()
            temp_kld_loss_detach_arr.append(kld_loss_detach)
            
            kld_loss = torch.sum(kld_loss)
            kld_loss.backward()
            
        temp_kld_loss_detach_arr = torch.cat(temp_kld_loss_detach_arr, dim=1)
        kld_loss_detach_arr.append(temp_kld_loss_detach_arr)
    kld_loss_detach_arr = torch.cat(kld_loss_detach_arr, dim=0)

    # Calculate the total loss (it is detached so cannot / SHOULD not backward again on it)
    total_kld_loss = torch.sum(kld_loss_detach_arr) / ((K) * (K-1))
    
    # Calculate min and max for the kld loss
    if min_max_mode in ["pairwise_sum_class_avg", "pairwise_sum_avg_class_avg"]:
        kld_loss_detach_arr = kld_loss_detach_arr + kld_loss_detach_arr.t()
        kld_loss_detach_arr = torch.sum(kld_loss_detach_arr, dim=1)
        kld_loss_detach_arr /=  (K - 1) 
        # if min_max_mode == "pairwise_sum_avg_class_avg":
        #     kld_loss_detach /= 2
        min_max_loss = torch.tensor([torch.min(kld_loss_detach_arr), torch.max(kld_loss_detach_arr)])
    else:
        sys.exit("Not implemented yet")


    """
        Convert the gradients with respect to the standard deviation into gradients with respect to the variance using chain rule
        --------------------------------------------------------------------------------------------------------------------------
    """

    std.grad = std.grad / (2 * std.detach())


    """
        --------------------------------------------------------------------------------------------------------------------------
    """

    # Store the gradients for mu and sigma in here
    grad_m, grad_s = mu.grad, std.grad
    kld_grad_arr = [
        (mu_grad, sigma_grad) for mu_grad, sigma_grad in zip(grad_m, grad_s)
    ]

    return total_kld_loss, kld_grad_arr, min_max_loss


# Return the KLD losses and the min and the max
def get_kld_losses_and_min_max(mu_arr, scale_tril_arr):
    n_gaussians = len(mu_arr)
    device_ = mu_arr[0].device
    
    distr_list  = [
        get_distribution(mu_arr[i], scale_tril_arr[i])
        for i in range(n_gaussians)
    ]
        
    kld_list = [
        - torch.distributions.kl.kl_divergence(
            distr_list[i], distr_list[j]
        )
        for i in range(n_gaussians) for j in range(n_gaussians)
    ]
    
    kld_tensor = torch.tensor(kld_list).to(device_)
    kld_tensor = kld_tensor.reshape(n_gaussians, n_gaussians)

    # Kld is calculated along row and columns for a single gaussian
    kld_tensor_1 = torch.clone(kld_tensor)
    kld_tensor_2 = torch.transpose(torch.clone(kld_tensor), 0, 1)

    # Note: Diagonal elements on KLD are equal to 0 (because gaussians are the same)
    kld_tensor_1 = torch.sum(kld_tensor_1, dim=1) 
    kld_tensor_2 = torch.sum(kld_tensor_2, dim=1) 


    #kld_tensor = (kld_tensor_1 + kld_tensor_2) / (2 * (n_gaussians - 1))
    kld_tensor = (kld_tensor_1 + kld_tensor_2) / (n_gaussians - 1)

    min_max_loss = torch.tensor([torch.min(kld_tensor), torch.max(kld_tensor)])
    return kld_tensor, min_max_loss

def get_pairwise_kld_losses(mu_arr, scale_tril_arr):
    with torch.no_grad():
        n_gaussians = len(mu_arr)
        device_ = mu_arr[0].device
        mu_arr.requires_grad=False
        scale_tril_arr.requires_grad=False
        
        if mu_arr[0].shape == scale_tril_arr[0].shape:
            K = len(mu_arr)
            D = len(mu_arr[0])
            chunk_size=100
            if chunk_size > K:
                chunk_size = K
            chunk_ratio = K // chunk_size
            mu = torch.stack([elt.detach().clone() for elt in mu_arr])
            std = torch.stack([elt.detach().clone() for elt in scale_tril_arr])

            kld_loss_detach_arr = []
            for c1 in range(chunk_ratio):
                c1_start = c1 * chunk_size
                c1_end = (c1 + 1) * chunk_size
                temp_kld_loss_detach_arr = []
                for c2 in range(chunk_ratio):
                    c2_start = c2 * chunk_size
                    c2_end = (c2 + 1) * chunk_size

                    mu_p1 = mu[c1_start:c1_end]
                    std_p1 = std[c1_start:c1_end]
                    
                    mu_p2 = mu[c2_start:c2_end]
                    std_p2 = std[c2_start:c2_end]

                    kld_loss = kl_div(mu_p1.unsqueeze(1), std_p1.unsqueeze(1), mu_p2.unsqueeze(0), std_p2.unsqueeze(0))
                    kld_loss_detach = kld_loss.detach().clone()
                    temp_kld_loss_detach_arr.append(kld_loss_detach)
                                    
                temp_kld_loss_detach_arr = torch.cat(temp_kld_loss_detach_arr, dim=1)
                kld_loss_detach_arr.append(temp_kld_loss_detach_arr)
            kld_loss_detach_arr = torch.cat(kld_loss_detach_arr, dim=0)

            return kld_loss_detach_arr
        
        else:
            distr_list  = [
                get_distribution(mu_arr[i], scale_tril_arr[i])
                for i in range(n_gaussians)
            ]
                
            kld_list = [
                torch.distributions.kl.kl_divergence(
                    distr_list[i], distr_list[j]
                )
                for i in range(n_gaussians) for j in range(n_gaussians)
            ]
            
            kld_tensor = torch.tensor(kld_list).to(device_)
            kld_tensor = kld_tensor.reshape(n_gaussians, n_gaussians)
            
            return kld_tensor

def norm_grad_max_min_loss(g, min_max):
    min_loss = min_max[0]
    max_loss = min_max[1]

    if min_loss == max_loss == 0:
        # The first time the KLD runs, all the gaussians are overlapping
        # This means their gradient is equal to 0 and their loss is also equal to 0
        # To avoid dividing by 0, we return the unmodified gradient (which should be 0 in this case)
        # It is also a sanity check
        #print("Did not normalise the gradients with the min and max loss because both were equal to 0")
        return g
    
    g = g / (max_loss - min_loss)
    return g

def update_mvn_params(cfg, trainer, i, mu, cov_lower, nf_mu_grad, nf_cov_lower_grad, kld_mu_grad, kld_cov_lower_grad,  
                      lr_mu, lr_sigma, lambda_nf_mu, lambda_nf_sigma, lambda_kld_mu, lambda_kld_sigma, adam_mu, adam_sigma, reparam, min_max_nf_loss, min_max_kld_loss,
                      info_dict):
    
    
    # Save the previous values for mu and sigma to calculate the norm of the new values
    # ---------------------------------------------------------------------------------------------
    prev_mu = torch.clone(mu)
    prev_cov_lower = torch.clone(cov_lower)
    # ---------------------------------------------------------------------------------------------

    # Calculate the gradients for mu
    # ---------------------------------------------------------------------------------------------
    if not cfg["val"]["gmm"]["kld_blank_mu"]:
        nf_mu_grad = lambda_nf_mu * norm_grad_max_min_loss(nf_mu_grad, min_max_nf_loss)
        kld_mu_grad = lambda_kld_mu * norm_grad_max_min_loss(kld_mu_grad, min_max_kld_loss)


        norm_mu_nf_l1 = torch.norm(nf_mu_grad, p=1).item()
        norm_mu_kld_l1 = torch.norm(kld_mu_grad, p=1).item()
        if norm_mu_kld_l1 != 0:
            mu_grad_norm_nf_div_kld_l1 = norm_mu_nf_l1 / norm_mu_kld_l1
            info_dict["mu_grad_norm_nf_div_kld_l1"] = mu_grad_norm_nf_div_kld_l1
           

    mu_grad = nf_mu_grad + kld_mu_grad
    adam_mu_grad = adam_mu.get_grad(mu_grad)
    mu = mu - lr_mu * adam_mu_grad


    # Calculate the gradients for Sigma
    # ---------------------------------------------------------------------------------------------
    ## Convert Sigma to the LATENT representation 
    # ---------------------------------------------------------------------------------------------
    cov_lower_latent = reparam.latent_representation(cov_lower) 
    # ---------------------------------------------------------------------------------------------
    d_theta_div_d_theta_hat = reparam.derivative(cov_lower_latent)
    nf_sigma_grad = nf_cov_lower_grad * d_theta_div_d_theta_hat
    kld_sigma_grad = kld_cov_lower_grad * d_theta_div_d_theta_hat

    if not cfg["val"]["gmm"]["kld_blank_sigma"]:
        nf_sigma_grad = lambda_nf_sigma * norm_grad_max_min_loss(nf_sigma_grad, min_max_nf_loss)
        kld_sigma_grad = lambda_kld_sigma * norm_grad_max_min_loss(kld_sigma_grad, min_max_kld_loss)

        norm_sigma_nf_l1 = torch.norm(nf_sigma_grad, p=1).item()
        norm_sigma_kld_l1 = torch.norm(kld_sigma_grad, p=1).item()     
        if norm_sigma_kld_l1 != 0:
            sigma_grad_norm_nf_div_kld_l1 = norm_sigma_nf_l1 / norm_sigma_kld_l1
            info_dict["sigma_grad_norm_nf_div_kld_l1"] = sigma_grad_norm_nf_div_kld_l1
            

    sigma_grad = nf_sigma_grad + kld_sigma_grad
    adam_sigma_grad = adam_sigma.get_grad(sigma_grad)
    cov_lower_latent = cov_lower_latent - lr_sigma * adam_sigma_grad


    ## Convert Sigma to the AMBIENT representation 
    # ---------------------------------------------------------------------------------------------
    cov_lower = reparam.ambient_representation(cov_lower_latent)


    # Convert to ParameterList so that gradients can be calculated for the gaussians if specified
    # ---------------------------------------------------------------------------------------------
    mu = nn.Parameter(mu, requires_grad=False)
    cov_lower = nn.Parameter(cov_lower, requires_grad=False)
    # ---------------------------------------------------------------------------------------------

    return mu, cov_lower, adam_mu_grad, adam_sigma_grad


