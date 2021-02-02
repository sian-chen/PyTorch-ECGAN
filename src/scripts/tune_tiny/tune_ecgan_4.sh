#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p8_0p2_none_ce0p02.json        --seed 0 &
CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p8_0p2_contragan1_ce0p02.json  --seed 0 &
sleep 180
CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p8_0p2_none_ce0p02.json        --seed 1 &
CUDA_VISIBLE_DEVICES=6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p8_0p2_contragan1_ce0p02.json  --seed 1 &
wait
