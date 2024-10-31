
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch import Tensor 
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import shutil
import cv2

import lightning as L 
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from sklearn.manifold import TSNE

from models.utils import *

import gc
from tqdm import tqdm

from my_util.label_encoder import *

import sys 
import os

from models.gaussians_util import *
from models.prior_loss import *

from my_util.logging_utils import *
from my_util.my_grad_utils import *
from my_util.quick_calculations import *
from my_util.my_plot_utils import *
from my_util.dataset_utils import *
from my_util.constant_names import *
from my_util.reparam_classes import *

from models.datamodules.datautils_and_transforms import *
from models.ema_pytorch.ema_pytorch import *
from models.ema_pytorch.post_hoc_ema import *


import time
import psutil

# torch.set_printoptions(edgeitems=5)
torch.set_printoptions(linewidth=300)
i_samples = 3

from diffusers.models import AutoencoderKL

def load_vae(device, dtype):
    t1 = time.time()
    print("Started loading vae")
    #vae = AutoencoderKL.from_single_file(path).to(dtype=dtype)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir="../Models/stabilityai/sd-vae-ft-mse", local_files_only=True).to(dtype=dtype)
    vae = vae.eval()
    vae.train = False
    for param in vae.parameters():
        param.requires_grad = False
    if device != None:
        vae = vae.to(device)
    t2 = time.time()
    print("Finished loading vae", t2 - t1)
    #vae = torch.compile(vae)
    return vae

# Function used to load the NF from pytorch lightning checkpoint directly inside of the pl class
def load_state_dict_from_pl(state_dict):
    layers_to_remove = []
    bad_words = ["adam", "sched", "min_max", "ema", "pw_kld_factors"]
    for key, val in state_dict.items():
        if any(word in key for word in bad_words):
            layers_to_remove.append(key)

    for key in layers_to_remove:
        del state_dict[key]

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        bad_name = "flow."
        if bad_name in k:
            name = k.replace(bad_name, "")  
        else:
            name = k
        new_state_dict[name] = v


    state_dict = new_state_dict 

    return state_dict

def replace_last_dim_with_mean(tensor):
    shape_size = len(tensor.shape)
    shape = torch.ones(shape_size-1).to(torch.int64).tolist()
    shape.append(tensor.shape[-1])
    tensor = torch.mean(tensor, dim=-1, keepdim=True).repeat(*shape)
    return tensor

def add_Id(mat, eps=1e-7):
    Id = torch.eye(mat.shape[0]).to(mat.device)
    mat = mat + Id * eps
    return mat

def get_off_diagonal_elements(M):
    return M[~torch.eye(*M.shape,dtype = torch.bool)]

def nan_sys_exit(tensor, msg):
    if torch.isnan(tensor).any():
        sys.exit(msg)
    
class PL_MEF(pl.LightningModule):
    def __init__(self, cfg, flow, label_encoder=None):
        super().__init__()
        self.automatic_optimization = cfg["automatic_optimization"]

        self.save_hyperparameters(ignore=['flow'])

        self.cfg = cfg
        self.flow = flow

        self.label_encoder = label_encoder
        
        self.num_classes = cfg["dataset_info"][cfg["dataset_name"]]["num_classes"]
        cfg["num_classes"] = self.num_classes
       
        #self.loss_fn = torch.compile(NLLLoss(2 ** cfg["bits"], self.num_classes, cfg), mode="default")
        self.loss_fn = NLLLoss(2 ** cfg["bits"], self.num_classes, cfg)

        self.register_buffer(name="nf_loss_min_max", tensor=torch.tensor([0.0, 0.0]))
        self.register_buffer(name="kld_loss_min_max", tensor=torch.tensor([0.0, 0.0]))

        cfg_gmm = self.hparams["cfg"]["val"]["gmm"]

        self.mu_adam_arr = nn.ModuleList([GaussianAdam(size=self.cfg["dim_lower"], beta1=cfg["val"]["gmm"]["adam_beta1"], beta2=cfg["val"]["gmm"]["adam_beta2"], active=cfg["val"]["gmm"]["use_adam_for_gradients"]) 
            for _ in range(len(self.flow.mu_arr))])
        self.cov_lower_adam_arr = nn.ModuleList([GaussianAdam(size=self.cfg["dim_lower"], beta1=cfg["val"]["gmm"]["adam_beta1"], beta2=cfg["val"]["gmm"]["adam_beta2"], active=cfg["val"]["gmm"]["use_adam_for_gradients"]) 
            for _ in range(len(self.flow.cov_lower_arr))])
        
    
        # Configure steps and decays for the learning rate of the gaussians
        # ----------------------------------------------------------------------------------------------------------------
        self.gmm_train_epochs = cfg["val"]["gmm"]["train_epochs"]

        def get_steps_and_lr_steps(lrs, steps, train_epochs):
            if train_epochs > steps[-1]:
                lrs.append(lrs[-1])
                steps.append(train_epochs)
                
            lrs = [float(elt) for elt in lrs]
            steps = [int(elt) for elt in steps]
            
            res_dict = {
                "lrs": lrs,
                "steps": steps,
            }
            if len(steps) != len(lrs):
                sys.exit("Config file not initialized correctly")

            return res_dict
        
        manual_sched_dict = {
            "lin": ManualMultiLinearScheduler,
            "geom": ManualMultiGeometricScheduler,
        }

        cfg_gmm["lambda_kld_mu_arr"][0] = cfg_gmm["kld"] 
        cfg_gmm["lambda_kld_sigma_arr"][0] = cfg_gmm["kld"] 
        print(cfg_gmm["lambda_kld_mu_arr"], cfg_gmm["lambda_kld_sigma_arr"])
        print(cfg_gmm["lambda_kld_mu_arr"], cfg_gmm["lambda_kld_sigma_arr"])

        self.lr_mu_sched = manual_sched_dict[cfg_gmm["manual_sched_lr_mu"]](
            **get_steps_and_lr_steps(cfg_gmm["mu_lr_arr"], cfg_gmm["mu_steps_arr"], self.gmm_train_epochs)
        )
        self.lr_sigma_sched = manual_sched_dict[cfg_gmm["manual_sched_lr_sigma"]](
            **get_steps_and_lr_steps(cfg_gmm["sigma_lr_arr"], cfg_gmm["sigma_steps_arr"], self.gmm_train_epochs)
        )
        self.lambda_kld_mu_sched =  manual_sched_dict[cfg_gmm["manual_sched_lambda_kld_mu"]](
            **get_steps_and_lr_steps(cfg_gmm["lambda_kld_mu_arr"], cfg_gmm["lambda_kld_mu_steps"], self.gmm_train_epochs)
        )
        self.lambda_kld_sigma_sched = manual_sched_dict[cfg_gmm["manual_sched_lambda_kld_sigma"]](
            **get_steps_and_lr_steps(cfg_gmm["lambda_kld_sigma_arr"], cfg_gmm["lambda_kld_sigma_steps"], self.gmm_train_epochs)
        )
        self.lambda_nf_mu_sched =  manual_sched_dict[cfg_gmm["manual_sched_lambda_nf_mu"]](
            **get_steps_and_lr_steps(cfg_gmm["lambda_nf_mu_arr"], cfg_gmm["lambda_nf_mu_steps"], self.gmm_train_epochs)
        )
        self.lambda_nf_sigma_sched = manual_sched_dict[cfg_gmm["manual_sched_lambda_nf_sigma"]](
            **get_steps_and_lr_steps(cfg_gmm["lambda_nf_sigma_arr"], cfg_gmm["lambda_nf_sigma_steps"], self.gmm_train_epochs)
        )

        # Stores kld ponderations
        self.K_arr_denominator = nn.Parameter(torch.zeros(self.num_classes, self.num_classes), requires_grad=False)
        self.K_arr = nn.Parameter(torch.zeros(self.num_classes, self.num_classes), requires_grad=False)
        self.tau_arr = nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)


        self.manual_sched_arr = [self.lr_mu_sched, self.lr_sigma_sched, self.lambda_kld_mu_sched, self.lambda_kld_sigma_sched,  self.lambda_nf_mu_sched,  self.lambda_nf_sigma_sched]

        self.val_step = 0
        self.val_epoch = 0
        # ----------------------------------------------------------------------------------------------------------------

        # To save full covariance matrices in float64 ... Need them because cannot reconstruct from scale_tril in float32
        self.full_cov_arr = []

        # To gather gradients from 1 gmm epoch to the other one
        self.init_gather_arr()


        self.current_noise_arr = None
        self.checkpoint_path_for_classification = None
        self.gpu_timer = GPUTimer(active=cfg["gpu_timer_active"])


        bs, ch, h, w, dim = self.get_bs_ch_h_w_dim()
        print(bs, ch, h, w, dim)
        self.classes_label_list = create_class_label_list(ch, h, w)

        self.ds_train_rank = None
        self.dl_train_rank = None

        if cfg["vae"]["is_wrapping"]:
            self.vae = [load_vae(None, torch.float32)] # Need to wrap so that it is not considered as a child module
        else:
            self.vae = None

        self.init_ema()


    def init_ema(self):
        cfg = self.cfg

        self.ema = None
        self.ema_name = cfg["train"]["ema_name"]
        if self.ema_name != "":
            if self.ema_name == "ema":
                self.ema = EMA(
                    self.flow,
                    ignore_startswith_names={"mu_arr", "cov_lower_arr", "scale_tril_arr"},
                    **cfg["train"][self.ema_name],
                )
                self.ema.ema_model.eval()
                self.ema = self.ema.to(self.device)

    def ema_if_possible(self):
        if self.ema == None:
            #print("ema has not been inited so returning false")
            return False
        
        if self.ema != None:
            if self.ema.initted.item():
                #print("ema has been inited so returning true")
                return True
            else:
                #print("ema has not been inited so returning false") 
                return False
            
    def log_accuracy_and_min_max_mean_kld(self, z, y, step):
        if z != None:
            preds, _ = self.get_predicted_labels(torch.clone(z).reshape(z.shape[0], -1), y)
            acc = ((preds == y).sum() / y.size(0)).item()
        else:
            preds = None
            acc = -1.0

        pw_kld = get_off_diagonal_elements(
            get_pairwise_kld_losses(self.flow.mu_arr, self.flow.scale_tril_arr)
        )
        min_kld, max_kld, mean_kld = torch.min(pw_kld).item(), torch.max(pw_kld).item(), torch.mean(pw_kld).item()
        self.logger.experiment.add_scalars(f"kld_and_acc_info",{
            "acc": acc,
            "min_kld": min_kld,
            "max_kld": max_kld,
            "mean_kld": mean_kld
        }, global_step=step)
        return preds, pw_kld

            
    def calculate_K_arr__and_tau_arr(self, kld_factor, mode="sym"):
        cfg = self.cfg
        fpi_info = cfg["val"]["gmm"]["fpi_info"]
        with torch.no_grad():
            pw_kld = get_pairwise_kld_losses(self.flow.mu_arr, self.flow.scale_tril_arr)
            

            if mode== "sym":
                if  fpi_info["exp_den_update"]:
                    pw_kld_scale = pw_kld * cfg["val"]["gmm"]["fpi_info"]["pw_kld_factors_scale"]
                    K_arr = torch.exp(-pw_kld / pw_kld_scale) / pw_kld_scale
                    K_arr = K_arr.fill_diagonal_(0.0) # Remove nan from zero division on the diagonal
                    tau_arr = kld_factor * (K_arr - K_arr.t()).sum(dim=1) # division by the number of classes is already in the kld_factor
                else:
                    if (torch.count_nonzero(self.K_arr_denominator).item() == 0) or (1):
                        pw_kld_scale = pw_kld * cfg["val"]["gmm"]["fpi_info"]["pw_kld_factors_scale"]
                        self.K_arr_denominator = nn.Parameter(torch.ones_like(pw_kld_scale) * (torch.mean(pw_kld_scale[pw_kld_scale != 0]).item()), requires_grad=False)
                    print("using the real exponential")
                    K_arr = torch.exp(-pw_kld / self.K_arr_denominator) / self.K_arr_denominator
                    K_arr = K_arr.fill_diagonal_(0.0) # Remove nan from zero division on the diagonal
                    tau_arr = kld_factor * (K_arr - K_arr.t()).sum(dim=1) # division by the number of classes is already in the kld_factor
                        
            else:
                sys.exit("this mode is no longer implemented for the kld formuala ...")
                
            nan_sys_exit(K_arr, "Found a nan for the K_arr")
            return K_arr, tau_arr
        
    
    def vae_wrap_encode(self, x, y=None):
        with torch.no_grad():
            if self.cfg["vae"]["is_wrapping"]:
                x  = self.vae[0].encode(x).latent_dist.sample()
            if self.cfg["val"]["gmm"]["gaussian_method_name"] in ENCODING_METHODS:
                x = torch.stack([self.label_encoder.encode_tensor(elt_x, elt_y, 1.0) for (elt_x, elt_y) in zip(x, y)])

            return x
    
    def vae_wrap_decode(self, x):
        with torch.no_grad():
            if self.cfg["val"]["gmm"]["gaussian_method_name"] in ENCODING_METHODS:
                x = x[:, :, :32, :32]
            if self.cfg["vae"]["is_wrapping"]:
                x  = self.vae[0].decode(x).sample
                x = right_range(x)
            return x


    def get_dtype(self):
        if self.cfg["trainer_precision"] == 32:
            return torch.float32
        else:
            return torch.float64
        
    def update_gaussian_params_arr_idx(self, idx, mu, cov, eps=None):
        mu = mu.data
        cov = cov.data
        
        if len(cov.shape) > 1: # we have a dense covariance matrix
            if (eps != None):
                cov = add_Id(cov, eps)                
            scale_tril = torch.linalg.cholesky(cov)
        else: 
            scale_tril = torch.sqrt(cov)
        
        mu = mu.to(self.get_dtype())
        cov = cov.to(self.get_dtype())
        scale_tril = scale_tril.to(self.get_dtype())

        self.flow.mu_arr[idx] = nn.Parameter(mu, requires_grad=False)
        self.flow.cov_lower_arr[idx] = nn.Parameter(cov, requires_grad=False)
        self.flow.scale_tril_arr[idx] = nn.Parameter(scale_tril, requires_grad=False)

    def load_hyperparams_from_checkpoint(self):
        cfg = self.hparams["cfg"]

        # We load gaussians from a pickle file
        if cfg["gaussian_hyperparams_pickle_file"] != "":
            print("Loaded gaussian_hyperparams_pickle_file")
            print("Loaded gaussian_hyperparams_pickle_file")
            hyperparam_dict = pickle_safe_load(cfg["gaussian_hyperparams_pickle_file"])

            if cfg["gaussian_hyperparams_method"] in ["mu", "mu_scale_diag", "mu_cov_diag", "mu_cov_full"]:
                    self.flow.mu_arr = nn.ParameterList(hyperparam_dict["mu"]).to(self.get_dtype())

            if cfg["gaussian_hyperparams_method"] in ["scale_diag", "mu_scale_diag"]:
                    self.flow.sigma_diag_arr = nn.ParameterList(hyperparam_dict["scale_diag"]).to(self.get_dtype())

            if cfg["gaussian_hyperparams_method"] in ["cov_diag", "mu_cov_diag"]:
                self.flow.sigma_diag_arr = nn.ParameterList(hyperparam_dict["cov_diag"]).to(self.get_dtype())

            if cfg["gaussian_hyperparams_method"] in ["cov_full", "mu_cov_full"]:
                self.flow.sigma_diag_arr = nn.ParameterList(hyperparam_dict["scale_tril"]).to(self.get_dtype())

            hyperparam_dict = None
    
    def init_gather_arr(self):
        cfg = self.hparams["cfg"]

        self.gather_nf_grad_m = [torch.zeros(cfg["dim_lower"]).to(self.device) for _ in range(len(self.flow.mu_arr))]
        self.gather_nf_grad_s = [torch.zeros(cfg["dim_lower"]).to(self.device) for _ in range(len(self.flow.mu_arr))]
        self.gather_kld_grad_m = [torch.zeros(cfg["dim_lower"]).to(self.device) for _ in range(len(self.flow.mu_arr))]
        self.gather_kld_grad_s = [torch.zeros(cfg["dim_lower"]).to(self.device) for _ in range(len(self.flow.mu_arr))]
        self.gather_z_arr = []
        self.gather_y_arr = []
        self.gather_log_det_arr = []
        self.gather_log_p_arr = []


    
    def configure_optimizers(self, only_optim=False):
        # Custom optimizers dict (stored inside of optim folder)
        optim_dict = {
            "adamax": torch.optim.Adamax,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }

        optimizer = None
        cfg_train = self.cfg["train"]
        optimizer = optim_dict[self.cfg["train"]["optimizer"]](
            [{'params': self.flow.parameters()}], lr=cfg_train["lr"],
        )
        
        # Used to generate / load the optimizer in the PL_Classifier module
        if only_optim:
            return optimizer

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg_train["step_size"], cfg_train["lr_decay"])
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }

        return  [optimizer], [scheduler_dict]
    
    def forward(self, x, y, reverse=False, init=False, use_recorded_outputs=False, ema_if_possible=False):
        # We update the standard model, not the ema one (typically for training)
        if (not ema_if_possible):
            #print("not forwarding with ema")
            return self.flow(x, y, reverse=reverse, init=init, use_recorded_outputs=use_recorded_outputs)
        
        if ema_if_possible:
            if self.ema == None:
                sys.exit("Tried doing ema, but it did not exist, check if code logic is fine")
            else:
                #print("forwarding with ema")
                #print(self.ema.ema_model.training)
                #print(self.ema.ema_model.requires_grad)
                return self.ema(x, y, reverse=reverse, init=init, use_recorded_outputs=use_recorded_outputs)

        
        
    
    
    def run_init(self):
        cfg = self.hparams["cfg"]

        pl.seed_everything(cfg["seed"])
        self.flow.eval()
        with torch.no_grad():
            init_data_arr = []
            count = 0
            for i, (init_data, y) in enumerate(self.trainer.datamodule.dl_init): 
                init_data = init_data.to(self.device)
                init_data = self.vae_wrap_encode(init_data, y)
                init_data_arr.append(init_data)
                count += init_data.shape[0]
                if cfg["dataset_name"] in ["imagenet"]:
                    if count >= cfg["train"]["init_size"]:
                        print("breaking the loop for data data_init arr at count", count)
                        break

            init_data_arr = torch.cat(init_data_arr)
            _, _, _ = self.forward(init_data_arr, y, init=True, ema_if_possible=False)


        self.flow.train()


    def on_train_start(self):
        cfg = self.hparams["cfg"]

        self.checkpoint_folder = os.path.join(self.logger.log_dir, "", "checkpoints")
        
        if self.vae != None:
            self.vae[0] = self.vae[0].to(self.device)

        if cfg["nf_ckpt_path"] == None:
            self.run_init()

            if cfg["nf_weights_load_path"] != "":
                print("Also loading NF weights from checkpoint not strict")
                print("Also loading NF weights from checkpoint not strict")
                ckpt = torch.load(cfg["nf_weights_load_path"], map_location=self.device)['state_dict']
                ckpt = load_state_dict_from_pl(ckpt)
                self.flow.load_state_dict(ckpt, strict=False)

            if "fpi" in cfg["val"]["gmm"]["training_algorithm"]:
                print("running the gaussian init on training data")
                print("running the gaussian init on training data")
                if (cfg["gaussians_load_path"] == ""):
                    self.run_gaussian_training_init()

            if self.ema:
                self.ema.ema_model.level_out_shape_arr = copy.deepcopy(self.flow.level_out_shape_arr)

            # -----------------------------------------------------------------------------------------------------------------

        else:
            self.trainer.check_val_every_n_epoch = cfg["val"]["gmm"]["every_n_epochs"]
            self.trainer.reload_dataloaders_every_n_epochs=100000
        

        self.classes_list = self.trainer.datamodule.classes_list
        del self.trainer.datamodule.dl_init
        del self.trainer.datamodule.ds_init

    def training_step(self, batch, batch_idx):   
        #print(self.flow.mu_arr[0])
        cfg = self.cfg
        if cfg["train"]["center_warmup_epochs"] == -1:
            #print('skipping training step for the first epoch for fpi method')
            return
         
        self.distr_arr = [get_distribution(self.flow.mu_arr[i], self.flow.scale_tril_arr[i]) for i in range(self.num_classes)] if (batch_idx == 0) else self.distr_arr
        if len(batch) == 2:
            x, y = batch; bs = x.size(0); x = self.vae_wrap_encode(x, y)
        elif len(batch) == 3:
            x, y, shard_key = batch; bs = x.size(0); x = self.vae_wrap_encode(x, y)
            print(f"\n shard_key {self.global_rank} ----", shard_key)

        x, log_det, log_p = self.forward(x, y, ema_if_possible=False)
        loss, z_arr, min_max_nf_loss = self.loss_fn(x, y, log_det, log_p, self.flow.mu_arr, self.flow.scale_tril_arr, distr_arr=self.distr_arr)
        bpd = bits_per_dim(loss, self.cfg["dim"])

        # Log the training accuracy
        if (self.current_epoch + 1) % self.cfg["save_every_n_epochs"] == 0:
            with torch.no_grad():
                preds, _ = self.get_predicted_labels(x.detach().clone().reshape(x.shape[0], -1), y)
                acc_val = ((preds == y).sum() / y.size(0))
                self.log("1_1 train_acc", acc_val, on_epoch=True, sync_dist=True)
            
        #print(torch.cuda.mem_get_info())
        if not self.automatic_optimization:
            is_last_batch = ((batch_idx + 1) == self.trainer.num_training_batches)
            

            opt = self.optimizers()
            sched = self.lr_schedulers()

            # Manualy do the learning rate warmup 
            # ----------------------------------------------------------------------------------------
            current_step = batch_idx + self.current_epoch * self.trainer.num_training_batches
            #print("current steps here", batch_idx, self.trainer.num_training_batches, self.current_epoch, current_step)
            if current_step <= self.hparams["cfg"]["train"]["warmup_steps"]:
                lr_scale = ((current_step+1) / self.hparams["cfg"]["train"]["warmup_steps"]) * self.hparams["cfg"]["train"]["lr"]
                lr_scale = self.hparams["cfg"]["train"]["lr"] if lr_scale > self.hparams["cfg"]["train"]["lr"] else lr_scale
                for param_group in opt.param_groups:
                    param_group['lr'] = lr_scale
            # ----------------------------------------------------------------------------------------

                
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=False)
            opt.step()


            # if self.ema != None:
            #     if ((self.trainer.current_epoch) > 400):
            #         self.ema.update()

            if self.ema != None:
                if ((self.trainer.current_epoch) >= self.cfg["val"]["gmm"]["merge_datasets_after"]):
                    self.ema.update()

            if is_last_batch:
                sched.step()

        self.log("1_0 train_bpd", bpd, on_step=True, on_epoch=True, sync_dist=False, logger=True, prog_bar=True)
        self.log("1_1 train_loss", loss, on_step=False, on_epoch=True, sync_dist=False, logger=True, prog_bar=False)
        
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
       
        res_dict = {
            "loss": loss
        }
        return res_dict


    def on_train_epoch_start(self):
        cfg = self.hparams["cfg"]

        # Save the configfile inside of the logger
        dir_number = self.logger.log_dir.split("_")[-1]
        save_cfg_full_path = os.path.join(self.logger.log_dir, "", f"cfg_reload_{dir_number}.yaml")
        if not os.path.isfile(save_cfg_full_path):
            shutil.copy(cfg["cfg_full_path"], save_cfg_full_path)

    def on_train_epoch_end(self):
        cfg = self.hparams["cfg"]

        # Save the center warmup model     
        # ------------------------------------------------------------------------------------------------
        if (self.current_epoch == cfg["train"]["center_warmup_epochs"]) and (cfg["train"]["center_warmup_epochs"] != 0) :
            warmup_file = os.path.join(self.checkpoint_folder, "", f"warmup_{self.current_epoch}e.ckpt")
            torch.save({
                'state_dict': self.flow.state_dict(),
            }, warmup_file)
        # ------------------------------------------------------------------------------------------------

        # Used to validate / train the gaussians at the right moments
        # ------------------------------------------------------------------------------------------------
        if self.current_epoch == 0:
            self.trainer.check_val_every_n_epoch = cfg["val"]["gmm"]["every_n_epochs"]
        # ------------------------------------------------------------------------------------------------

        # Plots and saves images inside of TensorBoard
        # ------------------------------------------------------------------------------------------------
        if ((self.current_epoch + 1) % cfg["save_every_n_epochs"]) == 0 \
        or (self.current_epoch == cfg["train"]["epochs"] - 1):
            if self.global_rank == 0:
                self.show_results(self.current_epoch, T=cfg["T"], task_id=f"generated_images", noise_arr=self.current_noise_arr)
            # ------------------------------------------------------------------------------------------------

        # Run testing during the training to see how it does
        # ------------------------------------------------------------------------------------------------
        # if (self.current_epoch > 0) and (self.current_epoch % cfg["save_every_n_epochs"] == 0):
        if ((self.current_epoch+1) % cfg["test_every_n_epochs"] == 0):
            if self.global_rank == 0:
                if not cfg["val"]["gmm"]["gaussian_method_name"] in ONE_GAUSSIAN_METHODS:
                    self.run_testing()
        # ------------------------------------------------------------------------------------------------

  

    def on_train_end(self):
        cfg = self.hparams["cfg"]
        if self.global_rank == 0:
            
            if cfg["temperature_sampling_for_results"]:
                T_arr = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][::-1]
                # Temperature sampling for the default model
                for i, T_ in enumerate(T_arr):
                    # print("Sampling at", T_)
                    self.show_results(i, T=T_, task_id=f"generated_images_t", noise_arr=None)
                    

        # Needed --> do not delete 
        do_torch_distributed_barrier()

       
        # Store the checkpoint file for the classification
        def newest_file(path):
            files = os.listdir(path)
            paths = [os.path.join(path, basename) for basename in files]
            return max(paths, key=os.path.getctime)

        checkpoint_folder = os.path.join(self.logger.log_dir, "", "checkpoints")
        checkpoint_file = newest_file(checkpoint_folder)
        self.checkpoint_path_for_classification = checkpoint_file

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):   
    
        cfg = self.hparams["cfg"]

        if cfg["val"]["gmm"]["gaussian_method_name"] in NO_VAL_METHODS \
        or self.current_epoch < cfg["train"]["center_warmup_epochs"] \
        or self.current_epoch > cfg["val"]["gmm"]["merge_datasets_after"]:
            # Will skip the learning of the gaussian parameters once this condition is reached
            print("Skipped the validation / learning of gaussians", self.current_epoch)
            print("Skipped the validation / learning of gaussians", self.current_epoch)
            return
        

        self.flow.eval()
        # Initialize the gradients for the EM at the beginning of every epoch so that they can be accumulated and gathered
        if batch_idx == 0: self.init_gather_arr()

        is_last_batch = (batch_idx + 1) == (index_if_list_else_nbr(self.trainer.num_val_batches, dataloader_idx)) 

        # Forward all the data points x to the latent space x --> z 
        # At the end, also gather them on every gpu
        # ----------------------------------------------------------------------------

        if len(batch) == 2:
            x, y = batch; 
        elif len(batch) == 3:
            x, y, s_key = batch;
        bs = x.size(0); x = self.vae_wrap_encode(x, y)
        z, log_det, log_p = self.forward(x, y, ema_if_possible=False)

        out_ch, out_h, out_w = z.shape[1:] # Get the unflattened image shape
        z = z.reshape(z.shape[0], - 1) # Flatten the image shape
        self.gather_z_arr.append(z)
        self.gather_y_arr.append(y)
        self.gather_log_det_arr.append(log_det)
        self.gather_log_p_arr.append(log_p)

        if is_last_batch:
            self.gather_z_arr = torch.cat(self.gather_z_arr)
            self.gather_y_arr = torch.cat(self.gather_y_arr)
            self.gather_log_det_arr = torch.cat(self.gather_log_det_arr)
            self.gather_log_p_arr = torch.cat(self.gather_log_p_arr)

            if cfg["num_gpus"] != 1:
                z_arr_gather = self.all_gather(self.gather_z_arr)
                y_arr_gather = self.all_gather(self.gather_y_arr)
                log_det_arr_gather = self.all_gather(self.gather_log_det_arr)
                log_p_arr_gather = self.all_gather(self.gather_log_p_arr)
                self.gather_z_arr = torch.flatten(z_arr_gather, 0, 1)
                self.gather_y_arr = torch.flatten(y_arr_gather, 0, 1)
                self.gather_log_det_arr = torch.flatten(log_det_arr_gather, 0, 1)
                self.gather_log_p_arr = torch.flatten(log_p_arr_gather, 0, 1)

            z = self.gather_z_arr
            y = self.gather_y_arr
            log_det = self.gather_log_det_arr
            log_p = self.gather_log_p_arr


            print("shape after gather", z.shape, y.shape)
            print("shape after gather", z.shape, y.shape)
            num_classes = len(torch.unique(y))
            if (num_classes != len(self.flow.mu_arr)):
                sys.exit(f"Number of classes different from number of gaussians {num_classes} != {len(self.flow.mu_arr)}")

            preds, pw_kld = self.log_accuracy_and_min_max_mean_kld(z, y, 2*self.val_epoch)

            if cfg["val"]["gmm"]["training_algorithm"] == "glem":
                # Get the learning rates and the lambda coefficients
                # ----------------------------------------------------------------------------
                lr_mu = self.lr_mu_sched.get_lr()
                lr_sigma = self.lr_sigma_sched.get_lr()
                lambda_kld_mu = self.lambda_kld_mu_sched.get_lr()
                lambda_kld_sigma = self.lambda_kld_sigma_sched.get_lr()
                lambda_nf_mu = self.lambda_nf_mu_sched.get_lr()
                lambda_nf_sigma =  self.lambda_nf_sigma_sched.get_lr()
                # ----------------------------------------------------------------------------

                for loop_idx in range(cfg["val"]["gmm"]["n_loops"]):
                    # Calculate min and max losses
                    # ----------------------------------------------------------------------------------------------------------------
                    z = z.reshape(z.shape[0], out_ch, out_h, out_w)
                    loss, z_arr, min_max_nf_loss = self.loss_fn(z, y, log_det, log_p, self.flow.mu_arr, self.flow.scale_tril_arr)
                    bpd = bits_per_dim(loss, self.cfg["dim"])
                    z = z.reshape(z.shape[0], -1)

                    min_max_mode = "pairwise_sum_avg_class_avg" if cfg["dataset_name"] == IMAGENET else "pairwise_sum_class_avg"
                    kld_loss, kld_grad_arr, min_max_kld_loss = get_kld_loss_and_grad_and_min_max(self.flow.mu_arr, self.flow.scale_tril_arr, min_max_mode=min_max_mode)
                    # ------------------------------------------------------------------------------------------

                    # Using min max normalization for the Likelihood loss and KLD loss to better balance them
                    # ------------------------------------------------------------------------------------------------
                    nf_min_loss, nf_max_loss = min_max_nf_loss[0], min_max_nf_loss[1]
                    kld_min_loss, kld_max_loss = min_max_kld_loss[0],  min_max_kld_loss[1]

                    # Increase the normalization range of the loss to be safe, 
                    # We use abs because the sign of the loss is negative and we don't want to swap min and max with the loss_epsilon
                    loss_eps = 0.2
                    nf_min_loss -= torch.abs(nf_min_loss) * loss_eps
                    nf_max_loss += torch.abs(nf_max_loss) * loss_eps
                    kld_min_loss -= torch.abs(kld_min_loss) * loss_eps
                    kld_max_loss += torch.abs(kld_max_loss) * loss_eps
                    
                    dim_used = cfg["dim"]

                    # Store those losses in the hyperparameters
                    self.nf_loss_min_max[0] = nf_min_loss * cfg["val"]["gmm"]["total_bs_val"]
                    self.nf_loss_min_max[1] = nf_max_loss * cfg["val"]["gmm"]["total_bs_val"]
                    self.kld_loss_min_max[0] = kld_min_loss * ((len(self.flow.mu_arr)) * dim_used)
                    self.kld_loss_min_max[1] = kld_max_loss * ((len(self.flow.mu_arr)) * dim_used)

                    # ------------------------------------------------------------------------------------------------

                    # Log the min max losses
                    self.logger.experiment.add_scalars(f"xyz_min_max_nf_loss",{
                        "min_nf_loss_chosen": self.nf_loss_min_max[0] / cfg["val"]["gmm"]["total_bs_val"],
                        "max_nf_loss_chosen": self.nf_loss_min_max[1] / cfg["val"]["gmm"]["total_bs_val"],
                        "min_nf_loss_real": nf_min_loss,
                        "max_nf_loss_real": nf_max_loss,
                    }, global_step=self.val_step)

                    self.logger.experiment.add_scalars(f"xyz_min_max_kld_loss",{
                        "min_kld_loss_chosen": self.kld_loss_min_max[0] / ((len(self.flow.mu_arr)) * dim_used),
                        "max_kld_loss_chosen": self.kld_loss_min_max[1] / ((len(self.flow.mu_arr)) * dim_used),
                        "min_kld_loss_real": kld_min_loss,
                        "max_kld_loss_real": kld_max_loss,
                        "mean_kld_loss": torch.mean(kld_loss).item()
                    }, global_step=self.val_step)
                    # ----------------------------------------------------------------------------

                    # Calculate the gradients for the NF transport loss for every class
                    # ----------------------------------------------------------------------------------------------------
                    for class_idx in range(num_classes):
                        c_idx = (y==class_idx).nonzero(as_tuple=True)
                        z_i = z[c_idx]

                        grad_m_i, grad_s_i = mvn_grad_pytorch_formula(
                            self.flow.mu_arr[class_idx], self.flow.cov_lower_arr[class_idx], z_i, nll=True              
                        )
                        self.gather_nf_grad_m[class_idx] += grad_m_i
                        self.gather_nf_grad_s[class_idx] += grad_s_i
                        nf_grad_arr = [
                            (grad_m_elt, grad_s_elt) for _, (grad_m_elt, grad_s_elt) in enumerate(zip(self.gather_nf_grad_m, self.gather_nf_grad_s))
                        ]
                    # ----------------------------------------------------------------------------------------------------

                    # Iterates on the gradients on every gaussians and updates the gaussians after manipulating the gradients
                    for i, gathered_grads in enumerate(zip(nf_grad_arr, kld_grad_arr)):
                        nf_grads, kld_grads = gathered_grads[0], gathered_grads[1]

                        nf_grad_m, nf_grad_s = nf_grads[0], nf_grads[1]
                        kld_grad_m, kld_grad_s = kld_grads[0], kld_grads[1]
                        
                        # ----------------------------------------------------------------------------------------------------------------------------
                        # Actually update the gaussians parameters
                        # ----------------------------------------------------------------------------------------------------------------------------
                        info_dict={
                            "mu_grad_norm_nf_div_kld_l1": None,
                            "sigma_grad_norm_nf_div_kld_l1": None,
                            "grad_mu_nf_div_kld": None,
                            "grad_sigma_nf_div_kld": None
                        }
                        mu_arr_new_i, cov_lower_arr_new_i, adam_grad_mu, adam_grad_sigma = update_mvn_params(
                            cfg,
                            self.trainer, i,
                            self.flow.mu_arr[i], self.flow.cov_lower_arr[i],
                            nf_grad_m, nf_grad_s, kld_grad_m, kld_grad_s, 
                            lr_mu, lr_sigma, 
                            lambda_nf_mu , lambda_nf_sigma, 
                            lambda_kld_mu, lambda_kld_sigma, 
                            self.mu_adam_arr[i], self.cov_lower_adam_arr[i],
                            self.flow.reparam,
                            self.nf_loss_min_max, self.kld_loss_min_max,
                            info_dict
                        )
                        self.update_gaussian_params_arr_idx(i, mu_arr_new_i, cov_lower_arr_new_i)
                        # ----------------------------------------------------------------------------------------------------------------------------
                        
                        # Logging of gradients to understand what is happening with the gaussians hyperparameter
                        # ----------------------------------------------------------------------------------------------------------------------------
                        if not cfg["val"]["gmm"]["gaussian_method_name"] in ONE_GAUSSIAN_METHODS:
                            if self.global_rank == 0 and (i < i_samples) and (adam_grad_mu != None and adam_grad_sigma != None):
                                self.logger.experiment.add_scalars(f"z_grad_info{i}",{
                                    "adam_norm_mu": torch.norm((adam_grad_mu), p=1),
                                    "adam_norm_sigma": torch.norm((adam_grad_sigma), p=1),
                                }, global_step=self.val_step)

                    # We reset the adam after the first iteration so that the kld is also taken into account
                    if (self.current_epoch == 0) and (loop_idx == 0):
                        if cfg["val"]["gmm"]["reset_adam_after_first_iteration"]:
                            if self.val_step == 0:
                                for elt1, elt2 in zip(self.mu_adam_arr, self.cov_lower_adam_arr):
                                    elt1.reset()
                                    elt2.reset()

                    # ----------------------------------------------------------------------------------------------------------------------------
                    


                self.log("1_0 val_bpd", bpd, on_step=True, on_epoch=True, sync_dist=False, logger=True, prog_bar=True)
                self.log("1_1 val_loss", loss, on_step=True, on_epoch=True, sync_dist=False, logger=True, prog_bar=False)
                self.log("lr_mu", lr_mu, on_step=False, on_epoch=True, sync_dist=False, logger=True)
                self.log("lr_sigma", lr_sigma, on_step=False, on_epoch=True, sync_dist=False, logger=True)
                self.log("lambda_kld_mu", lambda_kld_mu, on_step=False, on_epoch=True, sync_dist=False, logger=True)
                self.log("lambda_kld_sigma", lambda_kld_sigma, on_step=False, on_epoch=True, sync_dist=False, logger=True)

                # ------------------------------------------------------------------------------------------------------------------------
                # Update the schedulers step (corresponds to epoch)
                for manual_sched in self.manual_sched_arr:
                    manual_sched.step_iteration()
                self.val_step = self.lr_mu_sched.get_current_iteration()

                # ------------------------------------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------------------------------------

            if cfg["val"]["gmm"]["training_algorithm"] == "fpi_nf":
                    fpi_info = cfg["val"]["gmm"]["fpi_info"]

                    # See the accuracy on the validation set
                    preds, _ = self.get_predicted_labels(torch.clone(z).reshape(z.shape[0], -1), y)
                    acc_val = ((preds == y).sum() / y.size(0))  
                    self.logger.experiment.add_scalars(f"_-_acc_val",{
                        "acc_val": acc_val.item(),
                    }, global_step=self.val_step)
                    
                    for class_idx in range(num_classes):
                        y_i_index = (y == class_idx).nonzero(as_tuple=True)[0]
                        z_i = z[y_i_index]
                    
                        data_mean_i = torch.mean(z_i, dim=0)
                        data_var_i = torch.var(z_i, dim=1) + torch.ones_like(data_var_i)

                        self.update_gaussian_params_arr_idx(class_idx, data_mean_i, data_var_i)

            print("before fpi nf kld")
            if cfg["val"]["gmm"]["training_algorithm"] == "fpi_nf_kld":
                    fpi_info = cfg["val"]["gmm"]["fpi_info"]

                    dtype = torch.float64 # Needed for more precise calculations
                    if cfg["dataset_name"] in ["imagenet"]:
                        dtype = torch.float32 # Needed for more memory efficiency

                    z = z.to(dtype) 
                    print(z.shape, z.dtype)
                    print(z.shape, z.dtype)
                    print(self.flow.mu_arr[0].shape[0])
                    print(self.flow.mu_arr[0].shape[0])

                    full_cov = fpi_info["full_cov"]     # Whether to learn full covariance using FPI  (it is not implemented yet) 
                    beta_chol = fpi_info["beta_chol"]  

                    # Those invervals give the possibility to train the means and covariance at different intervals / epochs
                    # In practice, it is used to disable the training of one of them (mean or covariance)
                    # ------------------------------------------------------------------------------------------------------------------------
                    interval_mu = fpi_info["interval_mu"]; start_mu, end_mu = interval_mu[0], interval_mu[1]
                    interval_sigma = fpi_info["interval_sigma"]; start_sigma, end_sigma = interval_sigma[0], interval_sigma[1]
                    train_mu = (self.current_epoch >= start_mu) and (self.current_epoch <= end_mu)
                    train_sigma = (self.current_epoch >= start_sigma) and (self.current_epoch <= end_sigma)
                    # ------------------------------------------------------------------------------------------------------------------------


                    # Extract lambda factor for the FPI
                    # Lambda should be identical for mean and covariance, otherwise the code will exit
                    # ------------------------------------------------------------------------------------------------------------------------
                    lambda_fpi_mu = fpi_info["lambda_fpi_mu"]
                    lambda_fpi_sigma = fpi_info["lambda_fpi_sigma"]
                    
                    kld_eq = fpi_info["kld_eq"]
                    kld_normalization_factor = (num_classes) * (num_classes - 1)
                    kld_factor_mu = lambda_fpi_mu / kld_normalization_factor
                    kld_factor_sigma = lambda_fpi_sigma / kld_normalization_factor

                    print("print those are the kld factors", kld_factor_mu, kld_factor_sigma)
                    print("print those are the kld factors", kld_factor_mu, kld_factor_sigma)
                    # ------------------------------------------------------------------------------------------------------------------------

                    # (k * d) matrix
                    # Each row is the mean of the datapoints of a class
                    data_mean_arr = torch.stack([
                        torch.mean(z[(y == class_idx).nonzero(as_tuple=True)[0]], dim=0)
                        for class_idx in range(num_classes)
                    ])

                    # (k * d) matrix
                    # Each row is the variance of the datapoints of a class
                    data_var_arr = torch.stack([
                        torch.var(z[(y == class_idx).nonzero(as_tuple=True)[0]], dim=0)
                        for class_idx in range(num_classes)
                    ])

                    del z
                    del y
                    torch.cuda.empty_cache()
                    z = None
                    y = None
                    # -------------------------------------------

                    for repet_idx in (pbar := tqdm(range(fpi_info["max_iters"]))):
                        K_arr, tau_arr = self.calculate_K_arr__and_tau_arr(kld_factor=kld_factor_sigma, mode=fpi_info["kld_eq"])
                        self.K_arr = nn.Parameter(K_arr, requires_grad=False)
                        self.tau_arr = nn.Parameter(tau_arr, requires_grad=False)

                        K_arr = torch.clone(self.K_arr)
                        tau_arr = torch.clone(self.tau_arr)

                        max_diff_mu = 0
                        max_diff_var = 0
                        max_diff_cov = 0
                                                    
                        if not full_cov:
                            # CLONE AND RESHAPE DATA
                            # ---------------------------------------------
                            mu_clone = torch.stack([torch.clone(self.flow.mu_arr[i]).to(dtype) for i in range(num_classes)])
                            var_clone = torch.stack([torch.clone(self.flow.cov_lower_arr[i]).to(dtype) for i in range(num_classes)])
                            var_inv_clone = torch.stack([1 / elt.to(dtype) for elt in var_clone ])

                            # (k * k * d) matrix
                            # Each row is the difference between mu_i - m_j (that's why using transpose)
                            mu_diffs_arr = torch.transpose(torch.stack([
                                mu_clone - mu_clone[class_idx]
                                for class_idx in range(num_classes)
                            ]), 0, 1)
                            
                            # (k * k * d) matrix
                            # Each row is the diagonal of the outer product of  (mu_i - m_j) (mu_i - m_j)^{T)
                            mu_diffs_arr_square = torch.square(mu_diffs_arr)
                            if cfg["val"]["gmm"]["fpi_info"]["sum_mu_diffs_arr_square"]:
                                mu_diffs_arr_square = torch.square(mu_diffs_arr).sum(dim=1)
                            print("mu_diffs_arr_square Shape here", mu_diffs_arr_square.shape)
                            print("mu_diffs_arr_square Shape here", mu_diffs_arr_square.shape)

                            # (k * k * d) matrix  # Each row is the variance for i to j
                            var_clone_arr = torch.stack([ var_clone for _ in range(num_classes)])
                            # (k * k * d) matrix # Each row is the inverse of variance i to j
                            var_inv_arr = torch.stack([ var_inv_clone for _ in range(num_classes)])


                            # Fixed point iteration calculations for all the means 
                            # ----------------------------------------------------------------
                            # (k * k * 1) matrix
                            K_ij = torch.clone(K_arr).unsqueeze(-1)
                            K_ji = torch.transpose(torch.clone(K_ij), 0, 1) if (kld_eq == "sym") else torch.zeros_like(K_ij)
                            if train_mu:
                                c_var_inv_arr = torch.clone(var_inv_arr)
                                c_var_clone = torch.clone(var_clone)

                                # Replace variance with mean of variance to stabilize ... (avoids spiking mean amplitude on a single dimension when labmda is high)
                                if cfg["val"]["gmm"]["fpi_info"]["use_isotropic_variance_when_training_mean"]:
                                    c_var_inv_arr = replace_last_dim_with_mean(c_var_inv_arr) 
                                    c_var_clone = replace_last_dim_with_mean(c_var_clone) 
                                else:
                                    pass
                                c_data_mean_arr = torch.clone(data_mean_arr)

                                kld_mu = (K_ij * c_var_inv_arr) + (K_ji * torch.transpose(c_var_inv_arr, 0, 1))
                                kld_mu = kld_mu * mu_diffs_arr
                                kld_mu = torch.sum(kld_mu, dim=1)
                                kld_mu = c_var_clone * kld_mu
                                mu_new = c_data_mean_arr + kld_factor_mu * kld_mu

                            else:
                                max_diff_mu = 0
                                mu_new  = torch.clone(mu_clone)
                            # ----------------------------------------------------------------

                          
                            # Fixed point iteration calculations for all the variances
                            # ----------------------------------------------------------------
                            # (k * k * 1) matrix
                            if train_sigma:
                                B = + data_var_arr \
                                    + kld_factor_sigma * var_clone * torch.sum(K_ij * var_inv_arr, dim=1) * var_clone
                                

                                A = var_clone * (
                                    + (1 + tau_arr.unsqueeze(-1)) * var_inv_clone \
                                    + kld_factor_sigma * var_inv_clone * torch.sum(K_ji * mu_diffs_arr_square,  dim=1) * var_inv_clone \
                                    + kld_factor_sigma * var_inv_clone * torch.sum(K_ji * var_clone_arr, dim=1) * var_inv_clone
                                ) * var_clone

                                BA = B*A
                                BA_05 = torch.pow(BA, 0.5)
                                var_new = BA_05 * (1/A)
                            else:
                                max_diff_var = 0
                                var_new = torch.clone(var_clone)
                            # ----------------------------------------------------------------
                            
                            nan_sys_exit(var_new, "nan sys exit in var_new")

                            diff_mu = mu_clone - mu_new
                            diff_var = var_clone - var_new

                            # max_norm1_mu = torch.max(torch.norm(diff_mu, dim=1, p=float('inf')))
                            # max_norm1_var = torch.max(torch.norm(diff_var, dim=1, p=float('inf')))

                            # max_diff_mu = max_norm1_mu
                            # max_diff_var = max_norm1_var

                            max_norm2_mu = torch.max(torch.norm(diff_mu, dim=1, p=2))
                            max_norm2_var = torch.max(torch.norm(diff_var, dim=1, p=2))

                            max_diff_mu = max_norm2_mu
                            max_diff_var = max_norm2_var


                            nan_sys_exit(max_diff_mu, "nan max diff mu")
                            nan_sys_exit(max_diff_var, "nan max diff var")

                            for class_idx in range(num_classes):
                                self.update_gaussian_params_arr_idx(class_idx, mu_new[class_idx], var_new[class_idx], None)

                        pbar.set_description(f"Max diff mu {max_diff_mu} -----  Max diff var {max_diff_var}")

                        self.logger.experiment.add_scalars(f"l2_max_difference",{
                            "xyzz_max_diff_l2_mu_final": max_diff_mu,
                            "xyzz_max_diff_l2_var_final": max_diff_var,
                        }, global_step=self.val_step)
                        self.val_step += 1

                        do_torch_distributed_barrier()

                        epsilon_convergence = fpi_info["epsilon_convergence"]
                        if not full_cov:
                            if ((max_diff_mu <= epsilon_convergence) and (max_diff_var <= epsilon_convergence)):
                                break
                        else:
                            sys.exit("not implemented for full covariance")
            

            preds, pw_kld = self.log_accuracy_and_min_max_mean_kld(z, y, 2*self.val_epoch+1)

            # Plot pairwise minimal kld factors
            k = num_classes * (num_classes - 1)
            k1 = int(k // 10)
            k2 = int(k)

            top_k_pw, _ = torch.topk(pw_kld, k=k1, largest=False)
            plt.plot(top_k_pw.cpu())
            TASK_ID = f"pairwise_kld_factors_{k1}"
            step = self.current_epoch
            save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=step, img=None, title=None)

            top_k_pw, _ = torch.topk(pw_kld, k=k2, largest=False)
            plt.plot(top_k_pw.cpu())
            TASK_ID = f"pairwise_kld_factors_{k2}"
            step = self.current_epoch
            save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=step, img=None, title=None)

            
                                
                                
        # Switch the flow back to training
        self.flow.train()

        res_dict = {
            "loss": None
        }
        return res_dict


    def on_validation_end(self):
        cfg = self.hparams["cfg"]

        if cfg["val"]["gmm"]["gaussian_method_name"] in NO_VAL_METHODS or \
        self.current_epoch < cfg["train"]["center_warmup_epochs"]:
            return

        # We only step the epochs here to remain consistent in several loops in the GMM
        for manual_sched in self.manual_sched_arr:
            manual_sched.step()
        self.val_epoch = self.lr_mu_sched.get_current_epoch()

        # # We create a new noise sample to see the progress during training
        if self.current_epoch < int(cfg["val"]["gmm"]["train_epochs"] / 1):
            bs, ch, h, w, dim = self.get_bs_ch_h_w_dim()
            self.current_noise_arr = self.sample_noise_arr(bs, cfg)

    
    def run_testing(self):
        with torch.no_grad():
            self.flow.eval()
            self.ema.ema_model.eval()
            
            self.test_tsne_arr = []
            self.test_tsne_y_arr = []

            bpd_avg = AverageMeter()
            acc_avg = AverageMeter()
            top5_acc_avg = AverageMeter()
 
            dl_test =  self.trainer.datamodule.test_dataloader()[0]
            for i, batch in tqdm(enumerate(dl_test)):
                x, y = batch; bs = x.size(0)
                x = x.to(self.device)
                y = y.to(self.device)
                x = self.vae_wrap_encode(x, y)
                
                x, log_det, log_p = self.forward(x, y, ema_if_possible=self.ema_if_possible())
                loss, z_arr, min_max_nf_loss = self.loss_fn(x, y, log_det, log_p, self.flow.mu_arr, self.flow.scale_tril_arr)
                bpd = bits_per_dim(loss, self.cfg["dim"])

                preds, _ = self.get_predicted_labels(x.reshape(x.size(0), -1), y)
                acc = ((preds == y).sum() / y.size(0))

                top5_acc = self.top5_acc(x.reshape(x.size(0), -1), y)

                bpd_avg.update(bpd)
                acc_avg.update(acc)
                top5_acc_avg.update(top5_acc)

            bpd = bpd_avg.avg
            acc = acc_avg.avg
            top5_acc = top5_acc_avg.avg


            self.logger.experiment.add_scalars(f"3_0 test during train bpd",{
                "bpd": bpd
            }, global_step=self.current_epoch)

            self.logger.experiment.add_scalars(f"3_0 test during train acc",{
                "acc": acc
            }, global_step=self.current_epoch)

            self.logger.experiment.add_scalars(f"3_0 test during train top5_acc",{
                "acc": top5_acc
            }, global_step=self.current_epoch)


            self.flow.train()
            self.flow.zero_grad()
            self.ema.ema_model.zero_grad()

    def gather_data_from_gpus(self):
        with torch.no_grad():
            self.flow.eval()
            num_gpus = self.cfg["num_gpus"]
             
            if self.dl_train_rank == None:
                
                if not self.cfg["dataset_name"] in ["imagenet"]:
                    data = self.trainer.datamodule.ds_train.data
                    targets = self.trainer.datamodule.ds_train.targets
                    transform_init = self.trainer.datamodule.ds_train.transform
                else:
                    print("doing data init for imagenet")
                    data = self.trainer.datamodule.ds_init.data
                    targets = self.trainer.datamodule.ds_init.targets
                    transform_init = None

                num = len(data) // num_gpus
                # if (len(data) % num_gpus) != 0:
                #     new_size = (len(data) // num_gpus) * num_gpus
                #     print("This is the new size", new_size)
                #     sys.exit("Did not have exact division in training init, should implement this in a better way")
                rank_start = self.global_rank * num
                rank_end = (self.global_rank + 1) * num

                data_rank = data[rank_start:rank_end]
                targets_rank = targets[rank_start:rank_end]
                ds_train_rank = MyDataset(data_rank, targets_rank, transform_init)
                dl_train_rank = torch.utils.data.DataLoader(
                    ds_train_rank,
                    batch_size=1000,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=self.cfg["num_workers"],
                    persistent_workers=False,
                    prefetch_factor=3,
                )

            
            z_arr = []
            y_arr = []
            for i, batch in tqdm(enumerate(dl_train_rank)):
                x, y = batch; bs = x.size(0)
                x = x.to(self.device)
                y = y.to(self.device)
                
                z, log_det, log_p = self.forward(x, y, ema_if_possible=False)
                z_arr.append(z)
                y_arr.append(y)
            z = torch.cat(z_arr)
            y = torch.cat(y_arr)
            print(z.shape, y.shape)
            
            if self.cfg["num_gpus"] != 1:
                z = self.all_gather(z)
                y = self.all_gather(y)
                z = torch.flatten(z, 0, 1)
                y = torch.flatten(y, 0, 1)
            print("final init shape", z.shape)
            self.flow.train()
            
        return z, y

    def run_gaussian_training_init(self):
        for _ in range(3): print("Initializing the gaussians on the training data")

        cfg = self.cfg
        fpi_info = cfg["val"]["gmm"]["fpi_info"]

        full_cov = fpi_info["full_cov"]
        identity_diag = fpi_info["identity_diag"]
        beta_chol = fpi_info["beta_chol"]

        self.flow.eval()
        with torch.no_grad():
            z, y = self.gather_data_from_gpus()
            z = z.reshape(z.shape[0], -1)
            
            for class_idx in range(len(torch.unique(y))):
                idx = (y == class_idx).nonzero(as_tuple=True)
                z_idx = z[idx]

                mean = torch.mean(z_idx, dim=0)
                cov = torch.cov(z_idx.t())
                if identity_diag:
                    cov = torch.eye(len(self.flow.mu_arr[0])).to(self.device) * cfg["val"]["gmm"]["reparam_method"]["sigmoid"]["a"]

                if not full_cov:
                    cov = torch.diag(cov)
                
                self.update_gaussian_params_arr_idx(class_idx, mean, cov, beta_chol)

            preds, pw_kld = self.log_accuracy_and_min_max_mean_kld(z, y, -1)


        self.flow.train()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            self.flow.eval()
            self.ema.ema_model.eval()

            cfg = self.hparams["cfg"]
            if batch_idx == 0:
                self.test_tsne_arr = []
                self.true_y_arr = []
                self.pred_y_arr = []
                self.log_probs_arr = []

            x, y = batch; bs = x.size(0); x = self.vae_wrap_encode(x, y)
            x, log_det, log_p = self.forward(x, y, ema_if_possible=self.ema_if_possible())
            loss, z_arr, min_max_nf_loss = self.loss_fn(x, y, log_det, log_p, self.flow.mu_arr, self.flow.scale_tril_arr)
            bpd = bits_per_dim(loss, self.cfg["dim"])

            preds, log_probs = self.get_predicted_labels(x.reshape(x.size(0), -1), y, scale_gaussian_params=True)
            self.test_tsne_arr.append(x.detach().cpu())
            self.true_y_arr.append(y.detach().cpu())
            self.pred_y_arr.append(preds)
            self.log_probs_arr.append(log_probs)
            
            acc = ((preds == y).sum() / y.size(0))

            self.log("1_0 test_bpd", bpd, on_step=False, on_epoch=True, sync_dist=False, logger=True, prog_bar=True)
            self.log("1_1 test_loss", loss, on_step=False, on_epoch=True, sync_dist=False, logger=True, prog_bar=False)
            self.log("1_1 test_acc", acc, on_step=False, on_epoch=True, sync_dist=False, logger=True, prog_bar=True)
            

            res_dict = {
                "loss": loss
            }
            return res_dict



    def get_bs_ch_h_w_dim(self):
        cfg = self.hparams["cfg"]
        bs = 64

        return (bs, cfg["ch"], cfg["h"], cfg["w"], cfg["dim"])
    

    def get_predicted_labels(self, x, y, scale_gaussian_params=False):
        cfg = self.hparams["cfg"]
        log_prob_arr = []
        for i in range(len(self.flow.mu_arr)):
            distr = get_distribution(
                self.flow.mu_arr[i], 
                self.flow.scale_tril_arr[i]
            )
            log_prob_arr.append(distr.log_prob(x))  
            
        log_probs = torch.stack(log_prob_arr)
        log_probs = torch.transpose(log_probs, 0, 1)
        preds = log_probs.argmax(dim=1)
        return preds, log_probs
    
    def top5_acc(self, x, y, scale_gaussian_params=False):
        cfg = self.hparams["cfg"]
        log_prob_arr = []
        for i in range(len(self.flow.mu_arr)):
            distr = get_distribution(
                self.flow.mu_arr[i], 
                self.flow.scale_tril_arr[i]
            )
            log_prob_arr.append(distr.log_prob(x))  

        log_probs = torch.stack(log_prob_arr)
        log_probs = torch.transpose(log_probs, 0, 1)
        # Use topk to get the top 5 predictions
        top5_values, top5_preds = log_probs.topk(5, dim=1)
        
        # Check if the true labels are in the top 5 predictions
        correct_preds = y.unsqueeze(1).expand_as(top5_preds) == top5_preds
        top5_acc = correct_preds.any(dim=1).float().mean().item()  # Calculate the top-5 accuracy
        
        return top5_acc
    
    def sample_noise_arr(self, bs, cfg, T=1, bs_forced=None):
        # To speed up calculations
        if len(self.classes_list) > 100:
            bs = 9
        
        if bs_forced:
            bs = bs_forced

        noise_arr = []
        for i in range(len(self.flow.mu_arr)):
            distr = get_distribution(
                self.flow.mu_arr[i], 
                torch.clone(self.flow.scale_tril_arr[i] * T)
            )
            noise = distr.sample(sample_shape=(bs,))
            noise_arr.append(noise)

        # If there are several gaussians, merge them and put them inside of the same batch
        noise_arr = torch.stack(noise_arr)
        noise_arr = torch.split(noise_arr, self.num_classes)
        noise_arr = torch.cat(noise_arr, dim=1)
        noise_arr = [elt for elt in noise_arr]

        return noise_arr
    

    def make_imgs_from_noise(self, bs, ch, h, w, T, noise_arr, max_noise=10):
        with torch.no_grad():
            self.flow.eval()

            noise_arr = self.sample_noise_arr(bs, self.cfg, T=T)

            z_arr = []
            for i, noise in enumerate(noise_arr[:max_noise]):
                y = torch.full((noise.shape[0],), i)
                z, _, _ = self.forward(noise, y, reverse=True, ema_if_possible=self.ema_if_possible())
                z_arr.append(z)

            self.flow.train()
        return z_arr, noise_arr
    


    def sample_noise_and_label_for_sampling(self, mu, sigma, distr, bs, cfg):
        z = distr.sample(sample_shape=(bs,))
        y = torch.full((z.shape[0],), 0)
        return z, y


    def one_conditionned_on_images_sample(self, bs, ch, h, w, T, cfg):
        with torch.no_grad():
            self.flow.eval()
            
            counts = torch.zeros(self.num_classes).to(self.device)
            z_arr = [[] for _ in range(self.num_classes)]
            y_arr = [[] for _ in range(self.num_classes)]

            
            mu_ = self.flow.mu_arr[0]
            sigma_ = self.flow.scale_tril_arr[0] * self.cfg["T"]
            distr = get_distribution(mu_, sigma_)
            while torch.min(counts) < bs:
                #print(f"Resampled in on conditioned on images min is {torch.min(counts)}")
                z, y = self.sample_noise_and_label_for_sampling(mu_, sigma_, distr, bs, cfg)
                z, _, _ = self.forward(z, y, reverse=True, ema_if_possible=self.ema_if_possible())

                y_arr_temp = [self.label_encoder.get_encoding_label(z_elt) for z_elt in z]
                y_arr_temp = torch.tensor(y_arr_temp).to(z.device)
                bin_counts = torch.tensor(np.bincount(y_arr_temp.cpu(), minlength=len(counts))).to(z.device)
                counts += bin_counts

                # We crop a square corresponding to the width shape --> remove the encoding portion
                z = self.label_encoder.remove_encoding_for_batch(z)

                for i in range(self.num_classes):
                    idx = (y_arr_temp == i).nonzero(as_tuple=True)[0]
                    z_temp = z[idx]
                    targets = torch.full((len(idx),), i).to(z.device)
                    
                    z_arr[i].append(z_temp)
                    y_arr[i].append(targets)

            # Concatenate tensors in the list of the same class
            z_arr = [torch.cat(elt) for elt in z_arr]
            y_arr = [torch.cat(elt) for elt in y_arr]

            # Only keep k imgs and targets from each class
            z_arr = [elt[:bs] for elt in z_arr]
            y_arr = [elt[:bs] for elt in y_arr]

            # Concatenate the entire the tensor with all the classes into a single one
            z_arr = torch.cat(z_arr)
            y_arr = torch.cat(y_arr)

            # Shuffle the images and the targets
            indices = torch.randperm(z_arr.size()[0]).to(z_arr.device)
            z_arr = z_arr[indices]
            y_arr = y_arr[indices]

            return z_arr, y_arr
    
    def sample_for_batch(self, bs, ch, h, w, T, cfg):
        # This case needs special sampling, to ensure all the classes are balanced
        if self.cfg["val"]["gmm"]["gaussian_method_name"] == "one_conditioned_on_images":
            if self.cfg["gaussian_hyperparams_pickle_file"] == "":
                z_arr, y_arr = self.one_conditionned_on_images_sample(bs, ch, h, w, T, cfg)
            else:
                sys.exti("should not provide gaussians reestimated on training data here")
            return z_arr, y_arr
        
        with torch.no_grad():
            self.flow.eval()
            z_arr = []
            y_arr = []

            for i in range(len(self.flow.mu_arr)):
                
                mu_ = torch.clone(self.flow.mu_arr[i])
                sigma_ = torch.clone(self.flow.scale_tril_arr[i])  * self.cfg["T"]
                distr = get_distribution(mu_, sigma_)
                z, y = self.sample_noise_and_label_for_sampling(mu_, sigma_, distr, bs, cfg)
                z, _, _ = self.forward(z, y, reverse=True, ema_if_possible=self.ema_if_possible())
                z = z.cpu()
                y = y.cpu()
                
                # Create the correct labels depending on the method used
                if not self.cfg["val"]["gmm"]["gaussian_method_name"] in ENCODING_SEVERAL_SAMPLING_TECHNIQUES:
                    y_arr.append(torch.full((bs,), i).to(z.device))

                else:
                    sys.exit("removed this, this sampling is not longer possible")

                z_arr.append(z)
            
            z_arr = torch.cat(z_arr)
            y_arr = torch.cat(y_arr)

            indices = torch.randperm(z_arr.size()[0]).to(z_arr.device)
            z_arr = z_arr[indices]
            y_arr = y_arr[indices]   
                  
        return z_arr, y_arr
    
    def create_tensors_for_dataset(self, bs_arr, ch, h, w, T, sample_seed, og_seed, cfg):
        ds_info = cfg["dataset_info"][cfg["dataset_name"]]
        imgs_per_class = int(ds_info["train_size"] // ds_info["num_classes"] * cfg["num_imgs_scale"])

        with torch.no_grad():
            pl.seed_everything(sample_seed)
            x_arr = []
            y_arr = []
            for i, bs in enumerate(bs_arr):
                x, y = self.sample_for_batch(bs, ch, h, w, T, cfg)
                x_arr.append(x.detach().cpu())
                y_arr.append(y.detach().cpu())
            x_arr = torch.cat(x_arr)
            y_arr = torch.cat(y_arr)
            print("y_arr bincount", np.bincount(y_arr.cpu()))
            pl.seed_everything(og_seed)

            return x_arr, y_arr
    
    def show_results(self, step, T=1, task_id="gen_imgs", noise_arr=None):
        cfg = self.hparams["cfg"]

        bs, ch, h, w, dim = self.get_bs_ch_h_w_dim()

        if self.current_epoch >= cfg["save_start_epoch"]: # To make sure does not crash because of sampling
            samples, noise_arr = self.make_imgs_from_noise(bs=bs, ch=ch, h=h, w=w, T=T, noise_arr=noise_arr)
            # The images are inside of an array for each prior so we iterate over them
            for i, sample_ in enumerate(samples):
                #print("T i", i)
                if cfg["val"]["gmm"]["gaussian_method_name"] in ENCODING_METHODS:
                    sample_ = append_label_results_to_batch(sample_, self.label_encoder, self.classes_label_list)
                    sample_ = self.label_encoder.preprocess_before_plotting_for_batch(sample_)

                sample_ = self.vae_wrap_decode(sample_)
                nrows = 8
                img_grid = make_img_grid(sample_, nrows=nrows)
                save_to_tensorboard(self.trainer, file_grp=f"{task_id}_{i}", step=step, img=img_grid, title=None)


        if task_id != "generated_images_t":
            for i, mu in enumerate(self.flow.mu_arr):
                if i < i_samples:
                    TASK_ID = f"mu_plot_{i}"
                    step = self.current_epoch
                    plot_and_sort_tensor(mu.clone().detach().cpu(), plot=False)
                    save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=step, img=None, title=None)
            
            for i, cov_lower in enumerate(self.flow.cov_lower_arr):
                if i < i_samples:
                    TASK_ID = f"var_plot_{i}"
                    step = self.current_epoch
                    if len(cov_lower.shape) > 1:
                        var = torch.diag(cov_lower)
                    else:
                        var = cov_lower
                    plot_and_sort_tensor(var.clone().detach().cpu(), plot=False)
                    save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=step, img=None, title=None)
                    
                    # if (self.current_epoch > 0) and ((self.current_epoch % 100) == 0):
                    #     TASK_ID = f"var_cov_plot_{i}"
                    #     step = self.current_epoch
                    #     plot_cov(cov_lower)
                    #     save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=step, img=None, title=None)


        # We run a TSNE on the points sampled from the gaussians
        bs, ch, h, w, dim = self.get_bs_ch_h_w_dim()

        bs_forced = None
        if cfg["dataset_name"] in ["imagenet"]:
            bs_forced = 2
        noise = self.sample_noise_arr(bs, self.cfg, bs_forced=bs_forced)
        noise = torch.cat(noise)
        noise = noise.reshape(noise.shape[0], -1)
        print("Before Starting TSNE")
        noise_emb = TSNE(n_components=2, verbose=0, random_state=0).fit_transform(noise.detach().cpu())
        print("Finished TSNE") 
        noise_emb = np.array_split(noise_emb, self.num_classes)

        print("before tsne scatter")
        multi_scatter(noise_emb, info_arr=self.classes_list, ds_name=cfg["dataset_name"])
        print("after tsne scatter")
        TASK_ID = f"2_0 tsne noise"
        save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=self.current_epoch, img=None, title=None)  

        print("before tsne scatter name")
        plot_class_centroid(noise_emb, self.classes_list, cfg)
        print("after tsne scatter name")
        TASK_ID = f"2_0 tsne noise names"
        save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=self.current_epoch, img=None, title=None) 
        
        return noise_arr 
    



    