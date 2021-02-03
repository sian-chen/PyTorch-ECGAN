#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p9_0p1_none_ce0p01.json --seed 2 &
sleep 180
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p9_0p1_none_ce0p01.json --seed 3 &
wait
