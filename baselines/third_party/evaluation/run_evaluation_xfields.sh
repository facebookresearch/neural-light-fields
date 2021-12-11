#!/bin/bash

source ~/miniconda3/bin/activate eval

HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 \
    CUDA_VISIBLE_DEVICES=$2 python run_evaluation.py \
    --mode xfields \
    --gt_dir ~/local/data/stanford_half/$1 \
    --pred_dir ~/local/results/xfields_$3/rendered_images \
    --out_dir eval/eval_xfields_$3 \
    --out_file metrics/xfields_$3.txt
