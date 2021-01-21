#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e --eval_type valid --seed 2 -c src/exp_configs/TINY_ILSVRC2012/ecgan1_none.json      &
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/main.py -t -e --eval_type valid --seed 2 -c src/exp_configs/TINY_ILSVRC2012/ecgan1_contragan.json &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e --eval_type valid --seed 2 -c src/exp_configs/TINY_ILSVRC2012/ecgan1_ntxent.json    &
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/main.py -t -e --eval_type valid --seed 2 -c src/exp_configs/TINY_ILSVRC2012/ecgan2_none_ce1.json  &
