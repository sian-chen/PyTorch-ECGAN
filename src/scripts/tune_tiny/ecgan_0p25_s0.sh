#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_1_0p25_none_ce0p1_lr4e-04_1e-04.json    --seed 0 &
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_1_0p25_none_ce0p1_lr4e-04_2e-04.json    --seed 0 &
CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_1_0p25_none_ce0p025_lr4e-04_1e-04.json  --seed 0 &
CUDA_VISIBLE_DEVICES=6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_1_0p25_none_ce0p025_lr4e-04_2e-04.json  --seed 0 &
wait
