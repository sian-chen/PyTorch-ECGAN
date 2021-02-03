#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan2_none_ce0p1_lr1e-04_1e-04.json --seed 1 &
sleep 180
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan2_none_ce0p1_lr1e-04_1e-04.json --seed 2 &
sleep 180
CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan2_none_ce0p1_lr1e-04_1e-04.json --seed 3 &
wait
