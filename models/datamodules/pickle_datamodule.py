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

sys.path.append("..")
from my_util.constant_names import *


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


class PickleDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


    def get_transforms(self):
        cfg = self.cfg
        dataset_name = cfg["dataset_name"]
        uniform_noise_value = cfg["uniform_noise_value"]
        signed_noise=0

        if dataset_name.lower() in ["mnist", "fashionmnist"]:
            transform_train = T.Compose([
                    ToPILImageIfNotAlready(),
                    T.Pad(2),
                    T.Pad(8, padding_mode='edge'),
                    T.RandomRotation(12),
                    T.CenterCrop(36),
                    T.RandomCrop(32),
                    T.ToTensor(), 
                    AddUniformNoise(mean=0.0, std=uniform_noise_value, signed=signed_noise)
            ])
            transform_val = copy.deepcopy(transform_train)
            transform_test = T.Compose([
                ToPILImageIfNotAlready(),
                T.Pad(2),
                T.ToTensor(),
                AddUniformNoise(mean=0.0, std=uniform_noise_value, signed=signed_noise)
            ])


        elif dataset_name.lower() in ["svhn"]:
            transform_train =  T.Compose([
                ToPILImageIfNotAlready(),
                T.ColorJitter(0.1, 0.1, 0.05),
                T.Pad(8, padding_mode='edge'),
                T.RandomRotation(12),
                T.CenterCrop(36),
                T.RandomCrop(32),
                T.ToTensor(),
            ])
            transform_val = copy.deepcopy(transform_train)
            transform_test = T.Compose([
                ToPILImageIfNotAlready(), 
                T.ToTensor(), 
            ])
        

        elif dataset_name.lower() in ["cifar10", "cifar100", "cinic10", "imagenet_32"]:
            signed_noise=False
            transform_train =  T.Compose([
                ToPILImageIfNotAlready(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1, 0.1, 0.05),
                T.Pad(8, padding_mode='edge'),
                T.RandomRotation(12),
                T.CenterCrop(36),
                T.RandomCrop(32),
                T.ToTensor(),
            ])
            transform_val = copy.deepcopy(transform_train)
            transform_test = T.Compose([
                ToPILImageIfNotAlready(), 
                T.ToTensor(), 
            ])

        elif dataset_name.lower() in ["tinyimagenet"]:
            signed_noise=False
            transform_train =  T.Compose([
                ToPILImageIfNotAlready(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.05, 0.05, 0.025),
                T.Pad(8, padding_mode='edge'),
                T.RandomRotation(2),
                T.CenterCrop(68),
                T.RandomCrop(64),
                T.ToTensor(),
            ])
            transform_val = copy.deepcopy(transform_train)
            transform_test = T.Compose([
                ToPILImageIfNotAlready(), 
                T.ToTensor(), 
            ])

        else:
            sys.exit("Did not find transforms for this dataset")

        return transform_train, transform_val, transform_test
    

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if self.trainer != None:
            print("This is the epoch from the datamodule", self.trainer.current_epoch)

        cfg = self.cfg
        dataset_name = cfg["dataset_name"]
        ds_info = cfg["dataset_info"][dataset_name]
        
        # We load custom dataset files (theyr are usualy the same as those in pytorch but in tensor format)
        ds_folder = os.path.join(cfg["data_path"], "", dataset_name)
        ds_dict_path = os.path.join(ds_folder, "", cfg["dataset_nf"] + ".pickle")
        print("this is the ds_dict_path", ds_dict_path)
        self.ds_dict_path = ds_dict_path
        ds_dict = pickle_safe_load(ds_dict_path)
        self.classes_list = ds_dict["classes"]
        if isinstance(self.classes_list, torch.Tensor):
            self.classes_list = self.classes_list.tolist()

        data_train, targets_train = ds_dict["train"]["data"], ds_dict["train"]["targets"]
        data_test, targets_test = ds_dict["test"]["data"], ds_dict["test"]["targets"]
        self.classes_list = ds_dict["classes"]
        transform_train, transform_val, transform_test = self.get_transforms()

        if not cfg["val"]["gmm"]["use_train_dataset"]:
            if cfg["val"]["gmm"]["do_it"] \
            and cfg["val"]["gmm"]["gaussian_method_name"] not in NO_VAL_METHODS:
                data_train, targets_train, data_val, targets_val = stratified_data_targets_split(data_train, targets_train,  (1 - ds_info["train_ratio"]), cfg["seed"])
            else:
                data_val = torch.clone(data_train[:16])
                targets_val = torch.clone(targets_train[:16])
        else:
            data_val, targets_val = torch.clone(data_train), torch.clone(targets_train)

        # Only take k items per class to run the code faster / make small experiments
        if cfg["dataset_info"]["items_per_class_train"] != 0:
            pl.seed_everything(cfg["seed"])
            data_train, targets_train = extract_k_items_per_class(data_train, targets_train, cfg["dataset_info"]["items_per_class_train"])
            
        if cfg["dataset_info"]["items_per_class_val"] != 0:
            pl.seed_everything(cfg["seed"])
            data_val, targets_val = extract_k_items_per_class(data_val, targets_val, cfg["dataset_info"]["items_per_class_val"])
                   
        # Keep track of the total validation batch size on each gpu
        cfg["val"]["gmm"]["total_bs_val"] = int(len(targets_val))
        
        pl.seed_everything(cfg["seed"])
        ds_train = MyDataset(data_train, targets_train, transform_train)
        ds_val   = MyDataset(data_val  , targets_val  , transform_val  )
        ds_test  = MyDataset(data_test , targets_test , transform_test )

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        # Initialize the init data
        init_data_size = cfg["train"]["init_batch_size"]
        init_data = torch.clone(data_train[:init_data_size])
        init_targets = torch.clone(targets_train[:init_data_size])
        self.ds_init = MyDataset(init_data, init_targets, transform_train)
        self.dl_init = torch.utils.data.DataLoader(self.ds_init, batch_size=init_data_size)
        
    def train_dataloader(self) -> torch.Any:
        cfg = self.cfg

        if cfg["val"]["gmm"]["gaussian_method_name"] not in NO_VAL_METHODS:
            if ((self.trainer.current_epoch) >= cfg["val"]["gmm"]["merge_datasets_after"]):
                if not cfg["val"]["gmm"]["use_train_dataset"]:
                    cfg["val"]["gmm"]["gaussian_method_name"] = get_equivalent_method_after_merge_dataset(cfg["val"]["gmm"]["gaussian_method_name"])
                    print("This is the new gaussian method name", cfg["val"]["gmm"]["gaussian_method_name"])
                    print("This is the new gaussian method name", cfg["val"]["gmm"]["gaussian_method_name"])
                    data_train, targets_train = self.ds_train.data, self.ds_train.targets
                    data_val, targets_val = self.ds_val.data, self.ds_val.targets
                    data_new = torch.cat((data_train, data_val), dim=0)
                    targets_new = torch.cat((targets_train, targets_val), dim=0)
                    pl.seed_everything(cfg["seed"])
                    idx = torch.randperm(targets_new.size()[0])
                    data_new = data_new[idx]
                    targets_new = targets_new[idx]
                    self.ds_train.data = data_new
                    self.ds_train.targets = targets_new
                    print("Reloading without train dataset")
                else:
                    print("reloading with train dataset")
                    self.trainer.check_val_every_n_epoch = 10000
                self.trainer.reload_dataloaders_every_n_epochs=10000
                print("No longer reloading datasets")
                print("No longer reloading datasets")



        dl_train = torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg["num_workers"],
            persistent_workers=True,
            prefetch_factor=3,
        )
        return dl_train
    
    def val_dataloader(self):
        dl_val = torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.cfg["val"]["batch_size"], 
            shuffle=False, 
            drop_last=False,
            pin_memory=False,
            num_workers=self.cfg["num_workers"],
            persistent_workers=True,
            prefetch_factor=3,
        )
        return dl_val
    
    
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
