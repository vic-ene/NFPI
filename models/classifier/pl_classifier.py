import os
import sys
from time import time
from pathlib import Path
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import lightning as L 
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from torchvision.datasets import MNIST, FashionMNIST,CIFAR10, CIFAR100
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import copy
import time 

from tqdm import tqdm
import time

from PIL import Image

from models.mef import *
from models.pl_mef import *

from my_util.logging_utils import *
from my_util.dataset_utils import *
from my_util.constant_names import *

from my_util.label_encoder import *




class LitResNetClassifier(pl.LightningModule):
    def __init__(self, cfg, gen_model_path, sample_id, train_id, ds_folder):
        
        super().__init__()
        self.save_hyperparameters(ignore=['gen_model_path'])
        cfg = self.hparams["cfg"]

      
        self.gen_model_path = gen_model_path
        self.sample_id = sample_id
        self.train_id = train_id
        self.ds_folder = ds_folder
    
        self.__dict__.update(locals())
        resnets = {
            18: torchvision.models.resnet18, 
            34: torchvision.models.resnet34,
            50: torchvision.models.resnet50, 
            101: torchvision.models.resnet101,
            152: torchvision.models.resnet152
        }

        self.automatic_optimization = False
        
        self.criterion = nn.CrossEntropyLoss()

        # instantiate model
        pl.seed_everything(cfg["seed"])
        self.resnet_model = resnets[cfg["classification"]["resnet_version"]]()
        
        dataset_name = cfg["dataset_name"]
        num_classes = cfg["dataset_info"][dataset_name]["num_classes"]
        in_ch = 1 if dataset_name in ONE_CH_DATASETS else 3
        if dataset_name in SMALL_FIRST_RESNET_CONVOLUTION_DATASETS:
            self.resnet_model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet_model.maxpool = nn.Identity()
        else:
            sys.exit("This dataset is not setup yet")
    
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        # Load a checkpointed model that was fine tuned
        if cfg["cnn_load_path"] != "":
            self.resnet_model = torch.load(cfg["cnn_load_path"], map_location=torch.device("cpu"))



        self.init_dataloaders()

        self.gpu_timer = GPUTimer(active=1)



    def define_bs_arr(self):
        cfg = self.cfg
        num_gpus = cfg["num_gpus"]
        dataset_name = cfg["dataset_name"]
        dataset_info = cfg["dataset_info"][dataset_name]
        num_imgs = dataset_info["train_size"] // 1
        num_imgs = int(num_imgs * cfg["num_imgs_scale"])
        
        if self.gen_model_path != None:
            if self.cfg["classifier_nf_strategy"] in CHANGE_DS_SIZE_METHODS:
                num_imgs = int(num_imgs * self.cfg["mixing_nbr"])
                

        num_classes = dataset_info["num_classes"]
        imgs_per_class = num_imgs // num_classes
        ch, h , w = dataset_info["ch"], dataset_info["h"], dataset_info["w"]
        img_per_class_per_gpu = (num_imgs/num_gpus)//num_classes // 1

        # Find the sampling Batch Size (bs)
        # ------------------------------------------------------------------------------------
        if dataset_name in [MNIST_NAME, FASHIONMNIST_NAME]:
            if num_gpus <= 4:
                bs = 1500
            if num_gpus == 8:
                bs = 750
        elif dataset_name in [CIFAR10_NAME, CIFAR100_1CH_NAME, SVHN_NAME, CINIC10_NAME]:
            if num_gpus <= 4:
                bs = 1250
            if num_gpus == 8:
                bs = 625
        elif dataset_name in [CIFAR100_NAME, CIFAR100_1CH_NAME]:
            if num_gpus <= 4:
                bs = 125
            if num_gpus == 8:
                bs = 62
        elif dataset_name in [TINYIMAGENET_NAME, TINYIMAGENET_32_NAME]:
            if num_gpus <= 4:
                bs = 125
            if num_gpus == 8: 
                bs = 62
        elif dataset_name in [DOTA_V1_32]:
            if num_gpus <= 4:
                bs = 1250
            if num_gpus == 8:
                bs = 625
        elif dataset_name in [EUROSAT]:
            if num_gpus <= 4:
                bs = 1250
            if num_gpus == 8:
                bs = 625
        else:
            sys.exit("Not implemented yet")
        # ------------------------------------------------------------------------------------

        num_normal_bs = int(img_per_class_per_gpu // bs)
        smaller_final_bs = int(img_per_class_per_gpu % bs)
        bs_arr = [bs for _ in range(num_normal_bs)]
        if smaller_final_bs > 0:
            bs_arr.append(smaller_final_bs)
        tot_imgs_per_class = int(sum(bs_arr) * num_gpus) 
        if tot_imgs_per_class < imgs_per_class:
            extra = (imgs_per_class - tot_imgs_per_class) // num_gpus
            print("Those are the extra images added", extra)
            bs_arr[-1] = bs_arr[-1] + extra

        # We add one extra value to be sure there are enough samples .... (code above could use refactoring)
        for i in range(len(bs_arr)):
            bs_arr[i] += 1
        self.bs_arr = bs_arr
        
        print("This is the bs_arr", self.bs_arr)
        


    def forward(self, X):
        logits = self.resnet_model(X)
        return logits
    
    
    def run_calculations(self, batch, batch_idx, task_name):
        if task_name == "test" and batch_idx == 0:
            self.true_y_arr = []
            self.pred_y_arr = []

        x, y = batch; bs = x.size(0)

        logits = self(x)
        loss = self.criterion(logits, y)
        pred_y = torch.argmax(logits.detach().clone(), dim=1)
            

        if task_name == "test":
            self.true_y_arr.append(y)
            self.pred_y_arr.append(pred_y)

        acc = (y == pred_y).type(torch.FloatTensor).mean()
            
        self.log(f"{task_name}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log(f"{task_name}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        res_dict = {
            "loss": loss,
            "acc": acc
        }
        return res_dict
    

    def manual_step_schedulers(self, is_last_batch):
        optimizer_name = self.cfg["classification"]["optimizer_name"] 
        sched = self.lr_schedulers()

        if optimizer_name == "adam" and is_last_batch:
            sched.step()

        if optimizer_name == "sgd":
            sched.step()
    
    def training_step(self, batch, batch_idx):
        is_last_batch = ((batch_idx+1) == self.trainer.num_training_batches)

        res_dict = self.run_calculations(batch, batch_idx, "train")
        loss = res_dict["loss"]
        opt = self.optimizers().optimizer
        opt.zero_grad()
        self.manual_backward(loss, retain_graph=False)
        opt.step()

        self.manual_step_schedulers(is_last_batch)

        return res_dict
    

    def val_dataloader(self):
        return self.dl_test
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.run_calculations(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.run_calculations(batch, batch_idx, "test")
    
    def on_test_end(self):
        # Plots the confusion matrix
        y_true = torch.cat(self.true_y_arr).detach().cpu()
        y_pred = torch.cat(self.pred_y_arr).detach().cpu()
        TASK_ID = "confusion_matrix_cnn"
        plot_confusion_matrix(y_true, y_pred)
        save_to_tensorboard(self.trainer, file_grp=TASK_ID, step=0, img=None, title=None)


    def on_train_end(self):
        ckpt_folder = os.path.join(self.logger.log_dir, "", "checkpoints")
        os.makedirs(ckpt_folder, exist_ok=True)
        ckpt_folder = os.path.join(ckpt_folder, "ckpt.ckpt")
        torch.save(self.resnet_model, ckpt_folder)

    def configure_optimizers(self):   
        cfg = self.hparams["cfg"]

        dataset_name = self.cfg["dataset_name"]
        optimizers = {'adam': Adam, 'sgd': SGD} 
        optimizer_name = self.cfg["classification"]["optimizer_name"]   

        if optimizer_name == "adam":
            optimizer = optimizers["adam"](
                self.parameters(), 
                lr=self.cfg["classification"][optimizer_name]["lr"]
            )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                self.cfg["classification"][optimizer_name]["steps"], 
                self.cfg["classification"][optimizer_name]["steps_decay"]
            )

        if optimizer_name == "sgd":
            optimizer = optimizers["sgd"](
                self.parameters(), 
                lr=self.cfg["classification"][optimizer_name]["lr"],
                momentum=self.cfg["classification"][optimizer_name]["momentum"],
                weight_decay=self.cfg["classification"][optimizer_name]["weight_decay"]
            )

            n_imgs_og = self.cfg["dataset_info"][dataset_name]["train_size"]
            n_imgs_gen = int(self.cfg["dataset_info"][dataset_name]["train_size"] * cfg["num_imgs_scale"])
            n_imgs = n_imgs_gen

            
             # If we sample our own dataset and merge it, we must increase the dataset accordingly size to be taken into account by the optimizer
            if self.gen_model_path != None:
                if self.cfg["classifier_nf_strategy"] in CHANGE_DS_SIZE_METHODS:
                    n_imgs += n_imgs_og + int(n_imgs_gen * self.cfg["mixing_nbr"])

            steps_per_epoch = (n_imgs // self.cfg["num_gpus"]) // self.cfg["classification"][dataset_name]["batch_size"]
            if (n_imgs // self.cfg["num_gpus"]) % self.cfg["classification"][dataset_name]["batch_size"] != 0 \
            and not self.cfg["classification"]["drop_last"]:
                    steps_per_epoch += 1


            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            )
        
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler_dict]
    
    def make_dl_train(self, ds_train, dataset_name):
        cfg = self.hparams["cfg"]

        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=cfg["classification"][dataset_name]["batch_size"],
            shuffle=True,
            drop_last=cfg["classification"]["drop_last"],
            pin_memory=True,
            num_workers=cfg["num_workers"],
            persistent_workers=True,
            prefetch_factor=3,
        )
        return dl_train
    
    def make_dl_test(self, ds_test, dataset_name):
        cfg = self.hparams["cfg"]

        dl_test = torch.utils.data.DataLoader(
            ds_test,
            batch_size=100,
            shuffle=False, 
            drop_last=False,
            pin_memory=True,
            num_workers=cfg["num_workers"],
            persistent_workers=True,
            prefetch_factor=3,
        )
        return dl_test

    def get_transforms(self, dataset_name):
        cfg = self.hparams["cfg"]
        dataset_name = cfg["dataset_name"]
        ds_info = cfg["dataset_info"][dataset_name]
        ch = ds_info["ch"]

        if dataset_name.lower() in [MNIST_NAME, FASHIONMNIST_NAME, SVHN_NAME]:
           
            transform_train = T.Compose([
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.Pad(2),
                T.ToTensor(), 
                T.Normalize((0.5), (0.5)),
            ])

            transform_test = T.Compose([
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.Pad(2),
                T.ToTensor(), 
                T.Normalize((0.5), (0.5)),
            ])

        if dataset_name.lower() in [CIFAR10_NAME, CIFAR100_NAME, CINIC10_NAME]:
            transform_train =  T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            transform_test =  T.Compose([
                #ConvertToKbits(cfg["bits"], active=cfg["classifier_nf_strategy"] != "do_not_use"),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if dataset_name.lower() in [TINYIMAGENET_NAME]:
            transform_train =  T.Compose([
                T.RandomCrop(64, padding=4),
                T.RandomHorizontalFlip(),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            transform_test =  T.Compose([
                #ConvertToKbits(cfg["bits"], active=cfg["classifier_nf_strategy"] != "do_not_use"),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if dataset_name.lower() in [TINYIMAGENET_32_NAME]:
            transform_train =  T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            transform_test =  T.Compose([
                #ConvertToKbits(cfg["bits"], active=cfg["classifier_nf_strategy"] != "do_not_use"),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if dataset_name.lower() in [DOTA_V1_32]:
            transform_train =  T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                ToPILImageIfNotAlready(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            transform_test =  T.Compose([
                ToPILImageIfNotAlready(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if dataset_name.lower() in [EUROSAT]:
            transform_train =  T.Compose([
                T.RandomCrop(64, padding=4),
                T.RandomHorizontalFlip(),
                ToPILImageIfNotAlready(),
                PosterizeIfSpecified(cfg["bits"], cfg["do_k_bits_posterize_transform"]),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            transform_test =  T.Compose([
                ToPILImageIfNotAlready(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


        return transform_train, transform_test
    
    
    def init_dataloaders(self):
        cfg = self.hparams["cfg"]

        dataset_name = cfg["dataset_name"]
        transform_train, transform_test = self.get_transforms(dataset_name)

        # We load custom dataset files (theyr are usualy the same as those in pytorch but in tensor format)
        ds_folder = os.path.join(cfg["pickle_data_path"], "", dataset_name)
        ds_dict_path = os.path.join(ds_folder, "", cfg["dataset_classification"] + ".pickle")
        ds_dict = pickle_safe_load(ds_dict_path)

        # If there is no "train" key in the dict, it means the pickle dataset created by the NF for the CNN is directly used
        # In this case we need to fetch the test dataset differently
        # And also add the classes to the original ds dict
        if not "train" in ds_dict:
            print("Using dataset from CNN")
            og_ds_path = os.path.join(ds_folder, "", f"{dataset_name}.pickle")
            ds_dict_og = pickle_safe_load(og_ds_path)
            ds_dict_test = ds_dict_og["test"],
            ds_dict_test = ds_dict_test[0]
            ds_dict_train = copy.deepcopy(ds_dict)
            ds_dict["classes"] = ds_dict_og["classes"]
        else:
            ds_dict_train, ds_dict_test = ds_dict["train"], ds_dict["test"]


        data_train, targets_train = ds_dict_train["data"], ds_dict_train["targets"]
        data_test, targets_test = ds_dict_test["data"], ds_dict_test["targets"]
        data_classes = ds_dict["classes"]

        # Dirty to change size of image dataset to still run the class
        if ds_dict_path.endswith("custom.pickle"):
            cfg["dataset_info"][dataset_name]["train_size"] = int(len(data_train))
            cfg["num_imgs_scale"] = 1.0
            print("using custom combined dataset so need to change number of images")

        # Extract only k ims per class for the classification if specified
        ipc = cfg["classification"][dataset_name]["imgs_per_class"]
        if ipc != 0:
            pl.seed_everything(cfg["seed"])
            data_train, targets_train = extract_k_items_per_class(data_train, targets_train, ipc)
        self.classes_list = data_classes

        ds_train = MyDataset(data_train, targets_train, transform=transform_train) 
        ds_test = MyDataset(data_test, targets_test, transform=transform_test) 

        # Store the original datasets
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.dl_train = self.make_dl_train(ds_train, dataset_name)
        self.dl_test = self.make_dl_test(ds_test, dataset_name)

        self.define_bs_arr()

        
    def train_dataloader(self):
        cfg = self.hparams["cfg"]
        if self.cfg["classifier_nf_strategy"] == "do_not_use":
            return self.dl_train
        
        dataset_name = cfg["dataset_name"]
        dataset_info = cfg["dataset_info"][dataset_name]
        imgs_per_class = dataset_info["train_size"] // dataset_info["num_classes"]
        ch, h , w, T = dataset_info["ch"], dataset_info["h"], dataset_info["w"], cfg["T"]  

        sample_seed_start = cfg["sample_seed_start"] if "sample_seed_start" in cfg else 0
        sample_seed = self.global_rank * self.trainer.max_epochs + self.trainer.current_epoch + sample_seed_start
        
        def sample_x_y_tensors(sample_seed):
            #print(self.global_rank, f"This is the sample seed {sample_seed}")
            self.gpu_timer.start()
            gen_model = self.load_nf_model(self.gen_model_path, (self.sample_id>0))
            label_encoder = get_label_encoder_and_update_cfg(cfg)
            gen_model.label_encoder = label_encoder
            ch, h, w = cfg["ch"], cfg["h"], cfg["w"]

            x, y = gen_model.create_tensors_for_dataset(self.bs_arr, ch, h, w, T, sample_seed, cfg["seed"], cfg)
            
            gen_model = None
            self.gpu_timer.stop("Dataset Sampling Finished")

            self.gpu_timer.start()
            tensor_arr = [(x, y)]

            # We only gather if there are a lot of gpus here
            # We do this because if we try to sample 1M images, the code crashes on 1gpu (out of mem cuda) even if we move the data to cpu ???  
            # Don't realy know how to fix it and it is easier to do it like that
            if cfg["num_gpus"] != 1:
                tensor_arr = self.all_gather(tensor_arr)[0]
                # Condition below results in : If gathering was performed with several devices (extra dimension 5th dimension was added at index 0), we remove it 
                if len(tensor_arr[0].shape) == 5:
                    x = tensor_arr[0].flatten(0, 1)
                    y = tensor_arr[1].flatten(0, 1)
            self.gpu_timer.stop("Tensor Gathering Finished")

            return x, y
        
        def convert_x_y_tensors_to_same_format_as_dataset(x, y, dataset_name):
            if dataset_name in [MNIST_NAME, FASHIONMNIST_NAME]:
                #x = F.interpolate(x, size=(28, 28), mode='bicubic', align_corners=False)
                x = TF.center_crop(x, 28)
                x = torch.clamp(x*256, 0, 255)
                x = torch.round(x).to(torch.uint8)
                x = torch.flatten(x, 0, 1)
                x = x.detach().cpu()
                y = y.detach().cpu()

            elif dataset_name in [CIFAR10_NAME, CIFAR100_NAME, SVHN_NAME, EUROSAT]:
                x = torch.clamp(x*256, 0, 255).detach().cpu()
                x = torch.round(x).to(torch.uint8)
                x = x.detach().cpu()
                y = y.detach().cpu()

            return x, y
        

        # Pickle file that will contain / contains the dataset
        ds_file = os.path.join(self.ds_folder, "", f"ds_{self.trainer.current_epoch}.pickle")

        # train_id = 0 if we are running a single classification task so we always go inside of this statement
        x, y = sample_x_y_tensors(sample_seed)
        x, y = extract_k_items_per_class(x, y, int(imgs_per_class * cfg["num_imgs_scale"]))
        x, y = convert_x_y_tensors_to_same_format_as_dataset(x, y, dataset_name)
        # If we run all the classification, we store the dataset to avoid resampling it
        ds_dict = {"data": x, "targets": y}

        if self.global_rank == 0:
            if cfg["do_all_classifications"]:
                pickle_save(ds_file, ds_dict)

    
    def test_dataloader(self):
        return self.dl_test
    
    def load_nf_model(self, path, swap=False):
        cfg = self.hparams["cfg"]
        label_encoder = get_label_encoder_and_update_cfg(cfg)
        nf_flow = MEF(cfg, cfg["num_levels"], cfg["num_flows"], cfg["conv_type"], cfg["flow_type"], cfg["num_blocks"], cfg["hidden_channels"], 
                  cfg["h"], cfg["w"], in_channels=cfg["ch"])
        nf_model = PL_MEF.load_from_checkpoint(path, flow=nf_flow, cfg=cfg, label_encoder=label_encoder)
        nf_model.load_hyperparams_from_checkpoint()
        nf_model = nf_model.to(self.device)

        return nf_model