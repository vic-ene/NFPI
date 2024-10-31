import sys
import shutil

import math
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

import sys

import json
import argparse


import json 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import lightning as L 
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging



from my_util.dataset_utils import *

from models.mef import *
from models.pl_mef import *
from models.classifier.pl_classifier import *
from models.datamodules.pickle_datamodule import *
from models.datamodules.webdataset_datamodule import *


from my_util.label_encoder import *

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def get_nf_trainer(cfg, name):
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg["output_dir"], name=name)
        checkpoint_callback = ModelCheckpoint(save_on_train_epoch_end=True, every_n_epochs=cfg["save_model_ckpt_every_n_epochs"])
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')

        callbacks = [
            TQDMProgressBar(cfg["refresh_rate"]), 
            checkpoint_callback,
            lr_monitor_callback,
        ]     

        if torch.cuda.is_available():
            trainer = pl.Trainer(
                accelerator = "gpu",
                num_nodes = cfg["num_nodes"],
                devices = cfg["num_devices"],
                strategy = "ddp_find_unused_parameters_false",
                precision = cfg["trainer_precision"], 
                max_epochs = cfg["train"]["epochs"],
                logger = tb_logger,
                callbacks = callbacks,
                num_sanity_val_steps = 0,
                check_val_every_n_epoch=1,
                deterministic=cfg["deterministic"],
                reload_dataloaders_every_n_epochs=cfg["val"]["gmm"]["merge_datasets_after"],
            )

            trainer_test = pl.Trainer(
                accelerator = "gpu",
                num_nodes = 1,
                devices = 1,
                precision = cfg["trainer_precision"], 
                logger = tb_logger,
                deterministic=cfg["deterministic"],
            )
        else:
            print("Did not find cuda trainer for nf !!!")
            print("Did not find cuda trainer for nf !!!")
            print("Did not find cuda trainer for nf !!!")


        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None  
        return trainer, trainer_test


def get_classifier_trainer(cfg, name):
    dataset_name = cfg["dataset_name"]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg["output_dir"], name=name)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg["classification"][dataset_name]["epochs"])
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    callbacks = [
        TQDMProgressBar(cfg["refresh_rate"]), 
        checkpoint_callback,
        lr_monitor_callback,
    ]     
    

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator = "gpu",
            num_nodes = cfg["num_nodes"],
            devices = cfg["num_devices"],
            strategy = "ddp",
            precision = cfg["trainer_precision"], 
            max_epochs = cfg["classification"][dataset_name]["epochs"],
            logger = tb_logger,
            callbacks = callbacks,
            num_sanity_val_steps = 0,
            reload_dataloaders_every_n_epochs=1 if cfg["classifier_nf_strategy"] in RELOAD_DATASET_EVERY_EPOCH_METHODS else 0,
            deterministic=cfg["deterministic"],
            plugins=None if not cfg["sync_batch_norm"] else [L.pytorch.plugins.TorchSyncBatchNorm()]
        )

        trainer_test = pl.Trainer(
                accelerator = "gpu",
                num_nodes = 1,
                devices = 1,
                precision = cfg["trainer_precision"], 
                logger = tb_logger,
                deterministic=cfg["deterministic"],
            )
    else:
        print("Did not find cuda trainer for cnn !!!")
        print("Did not find cuda trainer for cnn !!!")
        print("Did not find cuda trainer for cnn !!!")

        

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None  
    return trainer, trainer_test


def run_training(cfg):
    torch_version = torch.__version__
    print("This is the pytorch version", torch_version)

    torch.set_float32_matmul_precision('high')
    #torch.set_float32_matmul_precision('medium')
    torch.use_deterministic_algorithms(cfg["deterministic"])

    dataset_name = cfg["dataset_name"]

    # Load a certain model and keep and keep or training 
    cfg["nf_ckpt_path"] = None if cfg["nf_ckpt_path"] == "" else cfg["nf_ckpt_path"]
   
    if os.environ.get('SLURM_NNODES'):
        cfg["num_nodes"] = int(os.environ['SLURM_NNODES'])
        print("num nodes here",  cfg["num_nodes"])
    if os.environ.get('SLURM_GPUS_ON_NODE'):
        cfg["num_devices"] = int(os.environ['SLURM_GPUS_ON_NODE'])
        print("num gpus here", cfg["num_devices"])
    if os.environ.get('SLURM_JOB_GPUS'):
         extra_var = os.environ.get('SLURM_JOB_GPUS')
         print("This is extra variables", extra_var)

    cfg["num_gpus"] = cfg["num_nodes"] * cfg["num_devices"]
    print("Counted number of gpus by me: ", cfg["num_gpus"])
    print("Printed counted devices by torch.cuda.device.count()", torch.cuda.device_count())

    
    # Ensure not loading a NF checkpoint for the classifier if training the NF
    if cfg["train_nf"] == 1:
        cfg["nf_load_path"] =  ""

    # Fix batch size if there are several gpus
    if cfg["num_gpus"] >= 1:
        cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"] / cfg["num_gpus"])
        cfg["val"]["batch_size"] = int(cfg["val"]["batch_size"] / cfg["num_gpus"])
        cfg["classification"][dataset_name]["batch_size"] = int(cfg["classification"][dataset_name]["batch_size"] / cfg["num_gpus"])

    # Small fix to ensure a checkpoint is saved no matter what
    if cfg["save_every_n_epochs"] > cfg["train"]["epochs"]:
        cfg["save_every_n_epochs"] = cfg["train"]["epochs"]

    if not cfg["classifier_nf_strategy"] in cfg["classifier_nf_strategy_choices"]:
        sys.exit("Please provide a right Classification value")

    if cfg["save_model_ckpt_every_n_epochs"] > cfg["train"]["epochs"]:
        cfg["save_model_ckpt_every_n_epochs"] = None

    # Make sure running fpi, even if increase the number of center warmup
    if (cfg["train"]["center_warmup_epochs"]) >= (cfg["val"]["gmm"]["merge_datasets_after"]):
        merge_datasets_after = cfg["train"]["center_warmup_epochs"] + 1
        print("increased merge_datasets_after to", merge_datasets_after)
        print("increased merge_datasets_after to", merge_datasets_after)
        cfg["val"]["gmm"]["merge_datasets_after"] = merge_datasets_after


    # Debug the code
    if cfg["debug"]:
        import debugpy  
        debugpy.listen(5678)
        print("waiting for debbuger")
        debugpy.wait_for_client()
        print("Attached ! ")


    # Add variable to count the number of steps epoch 
    print("Those are the batch sizes", cfg["train"]["init_batch_size"], cfg["train"]["batch_size"], cfg["val"]["batch_size"])
# ------------------------------------------------------------------------------------------------------------------------
    
    # ----------------------------- Setup and train the NF ----------------------------------------------------------
    label_encoder = get_label_encoder_and_update_cfg(cfg)
        
    pl.seed_everything(cfg["seed"])
    nf_trainer, nf_trainer_test = get_nf_trainer(cfg, cfg["output_dir"])
    pl.seed_everything(cfg["seed"])
    nf_flow = MEF(cfg,cfg["num_levels"], cfg["num_flows"], cfg["conv_type"], cfg["flow_type"], cfg["num_blocks"], cfg["hidden_channels"], 
                  cfg["h"], cfg["w"], in_channels=cfg["ch"])
    pl.seed_everything(cfg["seed"])
    nf_model = PL_MEF(cfg, flow=nf_flow, label_encoder=label_encoder)

    if ("not_pickle_datapath" in cfg.keys()) or (cfg["pickle_data_path"] == ""):
        data_path = cfg["data_path"]
        nf_datamodule = WebdatasetDatamodule(cfg)
    else:
        nf_datamodule = PickleDataModule(cfg)
    
    if cfg["train_nf"]:
        if cfg["torch_compile"]:
            nf_flow_comp = torch.compile(nf_flow, mode=cfg["torch_compile_mode"])
            nf_model = PL_MEF(cfg, flow=nf_flow_comp, label_encoder=label_encoder)
        
        pl.seed_everything(cfg["seed"])
        nf_trainer.fit(nf_model, ckpt_path=cfg["nf_ckpt_path"], datamodule=nf_datamodule)  
        pl.seed_everything(cfg["seed"]) 
        nf_trainer_test.test(nf_model, datamodule=nf_datamodule)  
    # ----------------------------------------------------------------------------------------------------------------


    # Retrieve the checkpoint file of the NF --> (Used by the classifier to sample data)
    # ---------------------------------------------------------------------------------------
    gen_model_path = None
    if cfg["train_nf"] == 1:
        gen_model_path = nf_model.checkpoint_path_for_classification
        cfg["nf_load_path"] = gen_model_path
        cfg["gaussians_load_path"] = gen_model_path
    else:
        gen_model_path = cfg["nf_load_path"] if cfg["nf_load_path"] != "" else gen_model_path  
    print("here is the path retrieved", gen_model_path) 
    # ---------------------------------------------------------------------------------------

    # Test the model with different gaussians from a checkpoint (used inside or outside of the pipeline)
    # ---------------------------------------------------------------------------------------
    if cfg["test_nf_alone"]:
        acc_arr = []
        nf_load_path = cfg["nf_load_path"]
        for i in range(cfg["num_test_alone"]):
            nf_dirname = os.path.dirname(nf_load_path)
            nf_flow = MEF(cfg,cfg["num_levels"], cfg["num_flows"], cfg["conv_type"], cfg["flow_type"], cfg["num_blocks"], cfg["hidden_channels"], 
                  cfg["h"], cfg["w"], in_channels=cfg["ch"])
            nf_model = PL_MEF.load_from_checkpoint(nf_load_path, flow=nf_flow, cfg=cfg, label_encoder=label_encoder)
            nf_model.load_hyperparams_from_checkpoint()
            pl.seed_everything(int(cfg["seed"] + i ))

            result = nf_trainer_test.test(nf_model, datamodule=nf_datamodule)
            acc_arr.append(result[0]["1_1 test_acc"])
            print("done this one")
        acc_arr = torch.tensor(acc_arr)

        writer = SummaryWriter(log_dir=nf_dirname)
        mean_ = torch.mean(acc_arr, dim=0)
        std_ =  torch.std(acc_arr, dim=0)
        print(mean_, std_)
        print(mean_, std_)
        writer.add_scalars("1_1 test_acc_avg", {
            "acc_mean": mean_,
            "acc_std":  std_
        }, 0)
    # ---------------------------------------------------------------------------------------

    # ----------------------------- Setup and train the Classifier --------------------------------------------------
    # We usualy run only a single classifcation experiment, because thediffusers accelerate transformersvae is only a single way of sampling the data
    # However, when we use the encoding and multiple gaussians, there can be 3 ways of sampling the data
    if cfg["train_classifier"]:

        cfg["val"]["gmm"]["n_classification_experiments"] = 1
        ALL_CLASSIFICATIONS = [cfg["classifier_nf_strategy"]]

        n_sample_strategies = 1
        for sample_id in range(n_sample_strategies):
            for exp_id in range(cfg["val"]["gmm"]["n_classification_experiments"]):
                cfg["val"]["gmm"]["current_classification_experiment"] = exp_id

                T_arr = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7]
                if cfg["only_T"]:
                    T_arr = [cfg["T"]]

                if cfg["classifier_nf_strategy"] == "do_not_use":
                    T_arr = [1.0]

                print("This is the T_arr", T_arr)
                for T_ in T_arr:
                    cfg["T"] = T_

                    # Create a folder that will contain the pickle datasets to avoid resampling every time
                    ds_folder = os.path.join(cfg["output_dir"], "", cfg["output_dir"], f"{sample_id}_exp_{exp_id}_T{T_}", "", "pickle_datasets")
                    os.makedirs(ds_folder, exist_ok=True)

                    for train_id, classification_method in enumerate(ALL_CLASSIFICATIONS):
                        cfg["classifier_nf_strategy"] = classification_method

                        pl.seed_everything(cfg["seed"])
                        cl_exp_name = os.path.join(cfg["output_dir"], "", f"{sample_id}_exp_{exp_id}_T{T_}", "", classification_method)
                        cl_trainer, cl_trainer_test = get_classifier_trainer(cfg, cl_exp_name)
                        pl.seed_everything(cfg["seed"])
                        classifier =  LitResNetClassifier(cfg, gen_model_path, sample_id, train_id, ds_folder)
                        
                        pl.seed_everything(cfg["seed"])
                        cl_trainer.fit(classifier)
                        pl.seed_everything(cfg["seed"])
                        cl_trainer_test.test(classifier)



# Define main code loop
# ------------------------------------------------------------------------------------------------------------------------
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="___configs", config_name="cifar10_short")
def my_app(cfg):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(hydra_cfg)

    output_dir = hydra_cfg['runtime']['output_dir']
    add_variable_to_hydra_cfg(cfg, "output_dir", output_dir)
    print("This is the output dir", output_dir)
    output_dir_hydra = os.path.join(output_dir, "", ".hydra")
    print(output_dir_hydra)
    custom_save_cfg_file = os.path.join(output_dir_hydra, "", "cfg.yaml")
    #print(cfg)

    config_name_retrieved = hydra_cfg['job']['config_name']
    config_dir_retrieved = hydra_cfg['runtime']['config_sources'][1]["path"]
    config_path_retrieved = os.path.join(config_dir_retrieved, "", config_name_retrieved + ".yaml")

    # Copy the original config file in hydra dir
    shutil.copy(config_path_retrieved, custom_save_cfg_file)
    # Save it to also save it in lightning_logs/version_0
    add_variable_to_hydra_cfg(cfg, "cfg_full_path", custom_save_cfg_file)

    print(config_path_retrieved)
    print(config_path_retrieved)
    print(config_path_retrieved)


    # To know whether or not running in IDRIS -> doing a required import
    HPC = not os.getcwd().startswith("/home/")
    if HPC:
        print("running on HPC")
        import idr_torch

    # We pass cfg og to save them in the right folder
    run_training(cfg)
# ------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    my_app()




