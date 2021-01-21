#!/bin/bash
for i in {0..2};
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -t -e -c configs/TINY_ILSVRC2012/BigGAN.json --seed $i &
  CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py -t -e -c configs/TINY_ILSVRC2012/ContraGAN.json --seed $i &
  wait
done
