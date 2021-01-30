#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_contragan0p1_ce0p05_lr2e-04_2e-04.json --seed 0 &
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_contragan0p1_ce0p05_lr4e-04_5e-05.json --seed 0 &
CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_contragan0p1_ce0p05_lr4e-04_1e-04.json --seed 0 &
CUDA_VISIBLE_DEVICES=6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_contragan0p1_ce0p05_lr4e-04_2e-04.json --seed 0 &
wait
