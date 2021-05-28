#!/bin/bash

python src/scripts/eval_cond_metric.py 0 ecgan_v2_none_1_0 &
python src/scripts/eval_cond_metric.py 0 ecgan_v2_none_0_0p05 &
python src/scripts/eval_cond_metric.py 1 ecgan_v2_none_1_0p1 &
python src/scripts/eval_cond_metric.py 1 ecgan_v2_contra_1_0p1 &
wait
