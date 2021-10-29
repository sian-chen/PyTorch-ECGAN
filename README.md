# Energy-based Conditional Generative Adversarial Network (ECGAN)

This is the code used for the experiments described in the following paper:

> S.-A. Chen. C.-L. Li, and H.-T. Lin, "A Unified View of cGANs with and without Classifiers". In the Proceedings of the Conference on Neural Information Processing Systems (NeurIPS 2021), (Held online due to COVID-19), December 6–14, 2021

The repository is modified from [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).


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

## Examples
The ``src/configs`` directory contains config files used in our experiments.

To train and evaluate ECGAN-UC on CIFAR10:
```
python3 src/main.py -t -e -c src/configs/CIFAR10/ecgan_v2_none_0_0p01.json
```

To train and evaluate ECGAN-UC on Tiny ImageNet:
```
python3 src/main.py -t -e -c src/configs/TINY_ILSVRC2012/ecgan_v2_none_0_0p01.json --eval_type valid
```

To train and evaluate ECGAN-UCE on ImageNet (~12 days on 8 NVIDIA V100 GPUs):
```
python3 src/main.py -t -e -l -sync_bn -c src/configs/ILSVRC2012/imagenet_ecgan_v2_contra_1_0p05.json --eval_type valid
```

## Results
| Dataset | Method | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | ECGAN-UC | 10.078 | 7.936 | 0.990 | 0.988 | [Cfg](./src/configs/CIFAR10/ecgan_v2_none_1_0p01.json) | [Log](./logs/CIFAR10/ecgan_v2_none_1_0p01-train-2021_05_26_16_35_45.log) | [Link](https://drive.google.com/drive/folders/1Kig2Loo2Ds5N3Pqc85R6c46Hbx5n9heM?usp=sharing) |
| Tiny ImageNet | ECGAN-UC | 18.445 | 18.319 | 0.977 | 0.973 | [Cfg](./src/configs/TINY_ILSVRC2012/ecgan_v2_none_1_0p05.json) | [Log](./logs/TINY_ILSVRC2012/ecgan_v2_none_1_0p05-train-2021_05_26_16_47_55.log) | [Link](https://drive.google.com/drive/folders/1oVAIljTEIA3b0BHRVjcnukMf3POQQ3rw?usp=sharing) |
| ImageNet | ECGAN-UCE | 80.685 | 8.491 | 0.984 | 0.985 | [Cfg](./src/configs/ILSVRC2012/imagenet_ecgan_v2_contra_1_0p05.json) | [Log](./logs/ILSVRC2012/imagenet_ecgan_v2_contra_1_0p05-train-2021_10_03_00_11_58.log) | [Link](https://drive.google.com/drive/folders/1EkcotNsnA-KBvOCFkvpJpUVoSDRxk-EV?usp=sharing) |


## Citation
If you find this work useful, please consider citing the following paper:
```bib
@inproceedings{chen2021ECGAN,
  title   = {A Unified View of cGANs with and without Classifiers},
  author  = {Si-An Chen, Chun-Liang Li and Hsuan-Tien Lin},
  booktitle = {Advances in Neural Information Processing Systems},
  year    = {2021}
}
```
