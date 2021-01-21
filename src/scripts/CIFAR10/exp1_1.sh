#!/bin/bash

for i in {0..3};
do
  CUDA_VISIBLE_DEVICES=$i python main.py -t -e -c exp_configs/CIFAR10/ecgan1_none.json --seed $i &
  CUDA_VISIBLE_DEVICES=$(($i+4)) python main.py -t -e -c exp_configs/CIFAR10/ecgan1_contragan.json --seed $i &
  sleep 180
done

wait
