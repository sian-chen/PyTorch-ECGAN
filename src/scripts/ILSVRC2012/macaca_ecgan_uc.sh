#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/ILSVRC2012/imagenet_ecgan_v2_none_0p5_0p01.json --seed 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/ILSVRC2012/imagenet_ecgan_v2_none_0p5_0p1.json --seed 0
