#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -t -e --eval_type valid --seed 2 -c exp_configs/TINY_ILSVRC2012/ecgan2_contragan_ce1.json   &
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py -t -e --eval_type valid --seed 2 -c exp_configs/TINY_ILSVRC2012/ecgan2_ntxent_ce1.json      &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -t -e --eval_type valid --seed 2 -c exp_configs/TINY_ILSVRC2012/ecgan2_none_ce0p1.json      &
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py -t -e --eval_type valid --seed 2 -c exp_configs/TINY_ILSVRC2012/ecgan2_contragan_ce0p1.json &
