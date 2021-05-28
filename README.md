# Energy-based Conditional Generative Adversarial Network (ECGAN)

This repository is the official implementation of the paper "A Unified View of cGANs with and without Classifiers".
The code is modified from [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).

## Requirements

- Anaconda
- Python >= 3.6
- 6.0.0 <= Pillow <= 7.0.0
- scipy == 1.1.0 (Recommended for fast loading of [Inception Network](https://github.com/openai/improved-gan/blob/master/inception_score/model.py))
- sklearn
- seaborn
- h5py
- tqdm
- torch >= 1.6.0 (Recommended for mixed precision training and knn analysis)
- torchvision >= 0.7.0
- tensorboard
- 5.4.0 <= gcc <= 7.4.0 (Recommended for proper use of [adaptive discriminator augmentation module](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/tree/master/src/utils/ada_op))


You can install the recommended environment as follows:

```
conda env create -f environment.yml -n studiogan
```

With docker, you can use:
```
docker pull mgkang/studiogan:0.1
```

## Quick Start

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPU ``0``
```
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -e -c CONFIG_PATH
```

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPUs ``(0, 1, 2, 3)`` and ``DataParallel``
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -c CONFIG_PATH
```

Try ``python3 src/main.py`` to see available options.

## Dataset

* CIFAR10: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny Imagenet, Imagenet, or a custom dataset:
  1. download [Tiny Imagenet](https://tiny-imagenet.herokuapp.com) and [Imagenet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
┌── docs
├── src
└── data
    └── ILSVRC2012 or TINY_ILSVRC2012 or CUSTOM
        ├── train
        │   ├── cls0
        │   │   ├── train0.png
        │   │   ├── train1.png
        │   │   └── ...
        │   ├── cls1
        │   └── ...
        └── valid
            ├── cls0
            │   ├── valid0.png
            │   ├── valid1.png
            │   └── ...
            ├── cls1
            └── ...
```

## Reproduce Experiments
The ``src/exp_configs`` directory contains all config files used in our experiments.

To train and evaluate ECGAN-UC on CIFAR10:
```
python3 src/main.py -t -e -c src/exp_configs/CIFAR10/ecgan_v2_none_0_0p01.json --seed 0
```

To train and evaluate ECGAN-UC on Tiny ImageNet:
```
python3 src/main.py -t -e -c src/exp_configs/TINY_ILSVRC2012/ecgan_v2_none_0_0p01.json --seed 0
```
