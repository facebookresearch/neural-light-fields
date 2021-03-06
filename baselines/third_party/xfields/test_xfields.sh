#!/bin/bash

source ~/miniconda3/bin/activate xfields

CUDA_VISIBLE_DEVICES=$2 \
    python test.py \
    --dataset ~/local/data/stanford_half/$1 \
    --savepath ~/local/logs/xfields_$1 \
    --epoch_end $3 \
    --type view
