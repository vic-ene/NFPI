from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import pickle

import lightning as L
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
import numpy as np

import copy
import sys
from models.datamodules.datautils_and_transforms import *
from sklearn.model_selection import train_test_split
import os

import torchvision
from torchvision import transforms

import webdataset as wds

sys.path.append("..")
from my_util.constant_names import *

def identity(x):
    return x




class WebdatasetDatamodule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_transforms(self):
        cfg = self.cfg
        dataset_name = cfg["dataset_name"]
       
        if dataset_name.lower() in ["imagenet"]:
            if not cfg["vae"]["is_wrapping"]:
                sys.exit("Imagenet is only setup for vae")

            transform_train = None
            transform_val = None
            transform_test = None

        return transform_train, transform_val, transform_test

    def prepare_data(self):
        pass



    def setup(self, stage: str):
        cfg = self.cfg
        shard_folder = cfg["data_path"]
        print(shard_folder)
        print(shard_folder)
        
        url_start = str(0).zfill(6)
        ipc_train = cfg["dataset_info"]["items_per_class_train"]
        ipc_val = cfg["dataset_info"]["items_per_class_val"]
        train_url_end = str(ipc_train - 1).zfill(6)
        val_url_end = str(ipc_val - 1).zfill(6)
        
        train_urls = "train_{" + url_start + ".." + train_url_end  + "}.tar"
        train_urls = os.path.join(shard_folder, "", train_urls)
        print("Final train url", train_urls)

        val_urls = "train_{" + url_start + ".." + val_url_end  + "}.tar"
        val_urls = os.path.join(shard_folder, "", val_urls)
        print("Final val url", val_urls)


        transform_train, transform_val, transform_test = self.get_transforms()

        bs_train = self.cfg["train"]["batch_size"]
        bs_val = self.cfg["val"]["batch_size"]
        print(shard_folder)
        print(shard_folder)
        items_per_shard = int(shard_folder.split("/")[-1].split("_")[-1])
        self.batches_per_epoch_train =  int((items_per_shard // bs_train) * ipc_train / cfg["num_gpus"]) * 1
        self.batches_per_epoch_val =  int((items_per_shard // bs_val) * ipc_val / cfg["num_gpus"]) * 1

        print("This is bs train", bs_train)
        print("This is bs train", bs_train)
        print(self.batches_per_epoch_train)
        print(self.batches_per_epoch_val)


        self.ds_train = wds.WebDataset(train_urls, resampled=True)\
            .shuffle(cfg["shuffle_webdataset"], initial=cfg["initial_webdataset"])\
            .decode("torch", handler=lambda x: (transforms.ToTensor()(x[0]), x[1], x[2]))\
            .to_tuple("input.pyd", "cls", "__key__")\
            .map_tuple(transform_train, identity, identity)\
            .batched(16)\
            
        self.ds_val = wds.WebDataset(val_urls, resampled=True)\
            .shuffle(cfg["shuffle_webdataset"], initial=cfg["initial_webdataset"])\
            .decode("torch", handler=lambda x: (transforms.ToTensor()(x[0]), x[1], x[2]))\
            .to_tuple("input.pyd", "cls", "__key__")\
            .map_tuple(transform_train, identity, identity)\
            .batched(16)\
            



        ds_dict_test = pickle_safe_load(cfg["data_path_test_pickle"])
        data_test, targets_test = ds_dict_test["val"]["data"], ds_dict_test["val"]["targets"]
        self.ds_test = MyDataset(data_test, targets_test, transform=None)
        

        data_init_ds = pickle_safe_load(cfg["data_path_init_pickle"])
        x_init = data_init_ds["train"]["data"]
        y_init = data_init_ds["train"]["targets"]
        self.ds_init = MyDataset(x_init, y_init, transform=None)
        self.dl_init = torch.utils.data.DataLoader(self.ds_init, batch_size=cfg["train"]["init_batch_size"], shuffle=True)

        self.classes_list = data_init_ds["classes"]

    
    def train_dataloader(self):
        # https://github.com/webdataset/webdataset/issues/250
        dl = wds.WebLoader(
            self.ds_train, 
            batch_size=None, 
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        ).unbatched().shuffle(self.cfg["shuffle_webdataset"]).batched(self.cfg["train"]["batch_size"])\
        .with_epoch(self.batches_per_epoch_train)\
        .with_length(self.batches_per_epoch_train)\
        .repeat(self.cfg["train"]["epochs"])

        if ((self.trainer.current_epoch) >= self.cfg["val"]["gmm"]["merge_datasets_after"]):
            self.trainer.check_val_every_n_epoch = 10000
            self.trainer.reload_dataloaders_every_n_epochs=10000
            
        return dl

    
    def val_dataloader(self):
        # https://github.com/webdataset/webdataset/issues/250
        dl = wds.WebLoader(
            self.ds_val, 
            batch_size=None, 
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        ).unbatched().shuffle(self.cfg["shuffle_webdataset"]).batched(self.cfg["val"]["batch_size"])\
        .with_epoch(self.batches_per_epoch_val)\
        .with_length(self.batches_per_epoch_val)\
        .repeat(self.cfg["train"]["epochs"])
        return dl

    def get_test_datasets(self):
            return [self.ds_test]


    def test_dataloader(self): 
        ds_test_arr = self.get_test_datasets()
        dl_arr = []
        for ds in ds_test_arr:
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=self.cfg["test"]["batch_size"], 
                shuffle=False, 
                drop_last=False,
                pin_memory=True, 
                num_workers=self.cfg["num_workers"],
                persistent_workers=True,
                prefetch_factor=3,
            )
            dl_arr.append(dl)
        return dl_arr