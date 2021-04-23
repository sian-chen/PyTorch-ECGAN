#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p01_0p99_none_ce0p099_lr1e-04_2e-04.json --seed 0 &
sleep 180
CUDA_VISIBLE_DEVICES=1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p01_0p99_none_ce0p099_lr1e-04_2e-04.json --seed 1 &
sleep 180
CUDA_VISIBLE_DEVICES=2 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p01_0p99_none_ce0p099_lr1e-04_2e-04.json --seed 2 &
sleep 180
CUDA_VISIBLE_DEVICES=3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p01_0p99_none_ce0p099_lr1e-04_2e-04.json --seed 3 &
wait
