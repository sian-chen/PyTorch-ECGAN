#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/TINY_ILSVRC2012/ContraGAN.json --seed 1 &
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/TINY_ILSVRC2012/BigGAN.json --seed 1 &
sleep 180
CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/TINY_ILSVRC2012/ContraGAN.json --seed 2 &
CUDA_VISIBLE_DEVICES=6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/TINY_ILSVRC2012/BigGAN.json --seed 2 &
wait
