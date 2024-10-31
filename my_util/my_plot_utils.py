import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import io
from PIL import Image
import torchvision.transforms as T


from string import ascii_uppercase, digits
import random 
import time
from glob import glob
import os
from os.path import basename, normpath
import math 

import sys

import torch
import torchvision
from torchvision.transforms import ToTensor, ToPILImage

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred):
    conf_mat=confusion_matrix(y_true, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(
      cmap="jet", 
      include_values=False, 
      colorbar=True
    )
    

def get_plot_as_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(io.BytesIO(buf.getvalue()))
    image = ToTensor()(img).unsqueeze(0)
    image = image[0]
    return image

# Save the current image plotted with matplotlib to tensorboard
def save_to_tensorboard(trainer, file_grp, step, img=None, title=None):
    if title != None:
        plt.title(title)

    if img == None:
        img = get_plot_as_img()

    trainer.logger.experiment.add_image(file_grp, img, global_step=step)
    plt.clf()
    plt.cla()
    plt.close()
    sns.reset_defaults()

# Used to show images from a batch
def make_img_grid(imgs, nrows=8):
    is_int = imgs.dtype == torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype == torch.int32
    imgs = torchvision.utils.make_grid(imgs, nrow=nrows, padding=2, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    return imgs

def plot_and_sort_tensor(t, plot=False):
    plt.plot(t, label="not sorted")
    t_sort, ind = torch.sort(t)
    plt.plot(t_sort, label="sorted")
    plt.legend(loc="upper left")
    #plt.gca().set_box_aspect(1)
    #plt.gca().set_ylim(torch.min(t), torch.max(t))
    if plot:
        plt.plot()

PLOT_SIZE = 5

# Scatter a list of points with different colors
def multi_scatter(arr, colors=["seagreen", "dodgerblue", "salmon"], title="Title", info_arr=None, xy_lim=None, ds_name="cifar10"):
    if len(arr) != len(colors):
        # Choose custom color in some cases
        colors = ["crimson", "coral", "gold", "seagreen", "turquoise", "dodgerblue", "slateblue", "violet", "pink", "chocolate"]
        # If there are still not enough, we pick them randomly
        if len(arr) > len(colors):
            colors = ["#"+''.join([random.choice('0123456789ABCDEF') 
                                  for j in range(6)])
                                  for i in range(len(arr))]

    # Fix if classes are stored in side of a dict ... 
    if isinstance(info_arr, dict):
        info_arr = [elt for elt in info_arr.keys()]

    for i, elt in enumerate(arr):
        if info_arr == None:
            plt.scatter(elt[:, 0], elt[:, 1], c=colors[i], alpha=0.5, s=PLOT_SIZE)
        elif info_arr != None:
            plt.scatter(elt[:, 0], elt[:, 1], c=colors[i], alpha=0.5, s=PLOT_SIZE, label="{}: {}".format(i, info_arr[i]))
            
            # To display the Jaccard index which is the last in the list
            if(len(info_arr) > len(arr)) and ((i+1) == len(arr)):
                plt.scatter(0, 0, c="black", alpha=0, label="%: {}".format(info_arr[-1]))
            
            
            plt.tight_layout()
            if ds_name in ["mnist", "fashionmnist", "cifar10"]:
                ax = plt.subplot(111)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title)
    
    # Set xlim and ylim to a certain range
    if xy_lim != None:
        plt.xlim(xy_lim)
        plt.ylim(xy_lim)


def plot_class_centroid(arr, ds, cfg):
    m_arr = [np.mean(elt, axis=0) for elt in arr]
    plt.gcf().set_size_inches(10, 10, forward=True)
    for m, name in zip(m_arr, ds):
        plt.scatter(m[0], m[1])

        if cfg["dataset_name"] in ["tinyimagenet", "imagenet"]:
            plt.annotate(name, (m[0], m[1]), fontsize=6, ha='center', va='center')
        else:
            plt.annotate(name, (m[0], m[1]), ha='center', va='center')
        

import seaborn as sns
import numpy as np

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def replace_diagonal_with_zero(matrix):
    """
    Replace the diagonal elements of a PyTorch tensor with zeros.

    Parameters:
    - matrix: 2D PyTorch tensor

    Returns:
    - modified_matrix: Matrix with diagonal elements replaced by zeros
    """
    matrix = torch.tensor(matrix)
    diagonal_mask = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    modified_matrix = matrix - (diagonal_mask)
    modified_matrix = modified_matrix.numpy()
    return modified_matrix

def plot_cov(cov):
    cov = cov.cpu().numpy()
    cor = correlation_from_covariance(cov)
    cor = torch.tensor(cor)
    cor = torch.abs(cor)
    cor = cor.numpy()
    cor = replace_diagonal_with_zero(cor)

    sns.set(font_scale=3.0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor)
    plt.show()
        
        


