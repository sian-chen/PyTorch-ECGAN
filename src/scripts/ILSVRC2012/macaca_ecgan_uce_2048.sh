#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py -t -e -l -sync_bn --eval_type valid -c src/exp_configs/ILSVRC2012/imagenet_ecgan_v2_2048_contra_1_0p05.json
