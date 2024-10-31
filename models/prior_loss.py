import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Independent

import numpy as np 
import sys

def get_distribution(mu, scale_tril):
    if len(mu.flatten()) != len(scale_tril.flatten()):
        distr = MultivariateNormal(
            loc = mu,
            scale_tril = scale_tril
        )
    else:
        distr = Independent(
            Normal(mu, scale_tril), 1
        )
    return distr
    

from my_util.constant_names import *

class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    cfg:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k, num_classes, cfg):
        super(NLLLoss, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.cfg = cfg
    
    
    def forward(self, z, y, sldj, log_p, mu_arr, scale_tril_arr, distr_arr=None):
        z_arr = []
        prior_ll_arr = []

        bs, ch, w, h = z.shape  # Store these variables before reshaping
        z = z.reshape(bs, -1) # Reshape image into 1D


        bincount =   torch.bincount(y.cpu())      
        all_indices = bincount.nonzero(as_tuple=False).flatten().to(z.device)
        if self.cfg["val"]["gmm"]["gaussian_method_name"] in ["one", "one_conditioned_on_images"]:
            distr = get_distribution(mu_arr[0], scale_tril_arr[0])
            prior_ll = distr.log_prob(z)
        else:
            if not self.cfg["level_method"]["choice"] in EXTRA_LOG_P_METHODS:
                for i in all_indices:
                    y_i_index = (y == i).nonzero(as_tuple=True)[0]
                    z_i = z[y_i_index]

                    distr = get_distribution(mu_arr[i], scale_tril_arr[i])  if (distr_arr == None) else distr_arr[i]
                    log_pz = distr.log_prob(z_i)
                    prior_ll_arr.append(log_pz)
                    
                    # Store z inside of arr to return it
                    z_arr.append(z_i)
                prior_ll = torch.cat(prior_ll_arr, dim=0)
                
            else:
                for i in all_indices:
                    y_i_index = (y == i).nonzero(as_tuple=True)[0]
                    z_i = z[y_i_index]
                    log_p_i = log_p[y_i_index]

                    #print("Here pof", mu_arr[i].shape, scale_tril_arr[i].shape)
                    distr = get_distribution(mu_arr[i], scale_tril_arr[i])  if (distr_arr == None) else distr_arr[i]
                    log_pz = distr.log_prob(z_i)
                    if self.cfg["level_method"]["choice"] in EXTRA_LOG_P_METHODS:
                        log_pz = log_pz + log_p_i
                    prior_ll_arr.append(log_pz)
                    
                    # Store z inside of arr to return it
                    z_arr.append(z_i)
                prior_ll = torch.cat(prior_ll_arr, dim=0)
        z = z.reshape(bs, ch, h, w)


        if not self.cfg["vae"]["is_wrapping"]:
            dequant_term = - np.log(self.k / 2.0) * np.prod(self.cfg["dim"])
        else:
            dequant_term = - np.log(self.cfg["vae"]["max_abs_train"]) * np.prod(self.cfg["dim"])

        nll = -(prior_ll + sldj + dequant_term)
        with torch.no_grad():
            min_max_loss = torch.tensor([torch.min(nll.detach()), torch.max(nll.detach())])

        nll = nll.mean()

        return nll, z_arr, min_max_loss
    

    
def bits_per_dim(nll, dim):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    cfg:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    bpd = nll / (np.log(2) * dim)

    return bpd