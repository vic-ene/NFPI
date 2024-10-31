# NFPI
Code for the paper Learning conditionally untangled latent spaces using Fixed Point Iteration accepted at BMVC 2024.




# Installation

Create a conda environment, and install the requires packages.

```bash
conda create --name NFPI
conda install pip
# Install these 3 packages first
python -m pip install torch torchvision torchaudio
# Then the remaining packages
python -m pip install lightning lightning-fabric matplotlib seaborn natsort scikit-learn scipy tensorboard tensorboardX jupyterlab easydict opencv-python hydra-core rootutils einops scipy pre-commit rich pytest colorama tqdm beartype webdataset diffusers accelerate transformers tabulate prettytable
pa
```

# Setup 

For small-mid scale datasets (MNIST/CIFAR10/CIFAR100), the datasets are converted to pytorch tensors beforehand, and stored in a pickle file. 
To create those datasets, run the notebook ds_extractor.ipynb. The datasets will be stored in a folder one directory above this one, with the following path:
../PytorchDatasets/MyDatasets/{dataset-name}/{dataset-name}.pickle.

If you change this path, just make sure to update the config files (data_path: '....').



# Fixed Point Iteration on Synthetic Data
The notebook 2d_data.ipynb shows how to run the FPI 2D on synthetic data (it is in fact cifar10 data projected to a 2D latent space using TSNE).

# Training of Conditional NF + CNN
Config files are storred inside the folder \_\_\_configs, and are built using [Hydra](https://hydra.cc/docs/configure_hydra/intro/).
There is a folder for each dataset inside of \_\_\_configs.

To run with a config for cifar10, the following command can be used (--config-name  chooses a certain config file from the cifar10 folder in this case).

```bash
python main.py --config_path=___configs/cifar10 --config-name=mean
```



It is possible to change the lambda ponderation of the KLD loss  in the fixed point iteration by adding the following arguments. Both should have the same value. Keep in mind those are unnormalized coefficients, so they are in practice further divided by the number of classes squared in the code (ie 10 * 10 = 100 for CIFAR10)
```bash
val.gmm.fpi_info.lambda_fpi_mu=100000 val.gmm.fpi_info.lambda_fpi_sigma=100000
```

the argument '_extra_' is optional: it adds an extra name to the experiment folder.
```bash
extra=new_experiment 
```


### Shorter experiments
For shorter experiments on the pickled datasets (MNIST/CIFAR10/CIFAR100), you can modify the following argument, so that it takes only 500 images per class for instance. If it is set to 0, it will take all the images.
```bash
dataset_info.items_per_class_train=500
```



# Checkpoints
Some checkpoints are available [here](https://drive.google.com/drive/folders/1vldluVI6jUnNgfzpaMNF2RozMsZTKFKT?usp=drive_link):

Download the checkpoints folder (ckpts), and drop it in the root folder (same level as main.py). The provided checkpoints focus on the NF classification accuracy (not on CNN accuracy).

To use/test a checkpoint, you need to: 
* duplicate a config file with the same dataset name in ___configs (I do not know a workaround for this step, it is limitation of Hydra config I think).
* in this new config file, put the checkpoint path in nf_load_path: '....'
* in this new config file, set train_nf: 0

Run the command by replacing 

--config-name=name_of_the_copied_config_file_that_you_just_changed

```bash
python main.py --config_path=___configs/cifar10 --config-name=name_of_the_copied_config_file_that_you_just_changed extra=ckpt_test num_test_alone=1 only_T=1 T=1.0
```

This command will:
* Classify with the NF on the testing set
* Train a classifier with artificial images sampled using the NF, with a temperature of T=1.0.


In the checkpoints folder, there are also additional checkpoints that were pretrained on Imagenet 32, and finetune on cifar10/cifar100.
They can reach slightly higher accuracy scores: ~94.60 on cifar10 and ~73.79 cifar100, and use attention.

To run them, you must specify the checkpoint path as previously, but you also need to add an extra argument
```bash
++use_attention_after=1
```


# Acknowledgment

This code is built on [MEF](https://github.com/changyi7231/MEF), [pytorch-glow]( https://github.com/rosinality/glow-pytorch) and [lucidrain EMA](https://github.com/lucidrains/ema-pytorch). We thank them for releasing their code.




