#!/bin/bash
for i in {0..2};
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py -t -e --eval_type valid -c src/exp_configs/TINY_ILSVRC2012/ecgan_0p95.json --seed $i
done
