#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -t -e --eval_type valid --seed 0 -c exp_configs/TINY_ILSVRC2012/test.json
rm -rf */test-train*
