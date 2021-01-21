#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e --eval_type valid --seed 0 -c src/exp_configs/TINY_ILSVRC2012/ecgan2_none_ce0p1.json      &
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/main.py -t -e --eval_type valid --seed 0 -c src/exp_configs/TINY_ILSVRC2012/ecgan2_contragan_ce0p1.json &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e --eval_type valid --seed 1 -c src/exp_configs/TINY_ILSVRC2012/ecgan2_none_ce0p1.json      &
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/main.py -t -e --eval_type valid --seed 1 -c src/exp_configs/TINY_ILSVRC2012/ecgan2_contragan_ce0p1.json &
