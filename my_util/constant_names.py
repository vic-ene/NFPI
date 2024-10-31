import sys

MNIST_NAME = "mnist"
FASHIONMNIST_NAME = "fashionmnist"
CIFAR10_NAME = "cifar10"
CIFAR100_NAME = "cifar100"
SVHN_NAME = "svhn"
TINYIMAGENET_NAME = "tinyimagenet"
TINYIMAGENET_32_NAME = "tinyimagenet_32"
CINIC10_NAME = "cinic10"

DOTA_V1_32="dota_v1_32"
EUROSAT="eurosat"

CIFAR10_1CH_NAME = "cifar10_1ch"
CIFAR100_1CH_NAME = "cifar100_1ch"

IMAGENET = "imagenet"

ONE_CH_DATASETS = [MNIST_NAME, FASHIONMNIST_NAME]
SMALL_FIRST_RESNET_CONVOLUTION_DATASETS = [
    MNIST_NAME, FASHIONMNIST_NAME, CIFAR10_NAME, CIFAR100_NAME, SVHN_NAME, CINIC10_NAME,
    TINYIMAGENET_NAME,TINYIMAGENET_32_NAME, 
    CIFAR10_1CH_NAME, CIFAR100_1CH_NAME,
    DOTA_V1_32, EUROSAT
]
VAE_LATENT_DS = [IMAGENET]

ONE_GAUSSIAN_METHODS  = ["one", "one_conditioned_on_images"]
NO_VAL_METHODS = [
    "one", "one_conditioned_on_images",
]

GAUSSIAN_CHECKPOINTS_METHODS = ["one_per_class_all_in_center_gaussian_checkpoint", "one_per_class_all_in_center_conditioned_on_images_gaussian_checkpoint"]
DISTANCED_GAUSSIANS_FROM_INIT_METHODS = []

ENCODING_METHODS = ["one_conditioned_on_images", "one_per_class_all_in_center_conditioned_on_images"]
ENCODING_SEVERAL_SAMPLING_TECHNIQUES = ["one_per_class_all_in_center_conditioned_on_images", "one_per_class_all_in_center_conditioned_on_images_gaussian_checkpoint"]

REPLACE_DATASET_METHODS = ["nf_datasets", "nf_datasets_resample"]
RELOAD_DATASET_EVERY_EPOCH_METHODS = ["nf_datasets_resample", "mix_datasets_resample"]
CHANGE_DS_SIZE_METHODS = ["mix_datasets", "mix_datasets_resample"]


def get_equivalent_method_after_merge_dataset(method_name):
    if method_name in ['one_per_class_all_in_center']:
        return "one_per_class_all_in_center_gaussian_checkpoint"

    else:
        sys.exit("no longer implemented")
    
SPCB ="spcb"
ALL_LEVELS_IDX = "all_levels_idx"
EXTRA_LOG_P_METHODS = [SPCB, ALL_LEVELS_IDX]