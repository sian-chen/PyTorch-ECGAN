#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001_lr4e-04_2e-04.json --seed 0 &
sleep 180
CUDA_VISIBLE_DEVICES=1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001_lr4e-04_2e-04.json --seed 1 &
sleep 180
CUDA_VISIBLE_DEVICES=2 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001_lr4e-04_2e-04.json --seed 2 &
sleep 180
CUDA_VISIBLE_DEVICES=3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001_lr4e-04_2e-04.json --seed 3 &
wait
#CUDA_VISIBLE_DEVICES=0,1 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001.json --checkpoint_folder checkpoints/ecgan_0p99_0p01_none_ce0p001-train-2021_02_04_14_33_38 -current --log_output_path logs/ecgan_0p99_0p01_none_ce0p001-train-2021_02_04_14_33_38 --seed 0 &
#sleep 180
#CUDA_VISIBLE_DEVICES=2,3 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001.json --seed 1 &
#sleep 180
#CUDA_VISIBLE_DEVICES=4,5 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001.json --seed 2 &
#sleep 180
#CUDA_VISIBLE_DEVICES=6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/tune_tiny/ecgan_0p99_0p01_none_ce0p001.json --seed 3 &
#wait
