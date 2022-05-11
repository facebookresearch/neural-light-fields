#!/bin/bash

source ~/miniconda3/bin/activate eval

HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 \
    CUDA_VISIBLE_DEVICES=$2 python run_evaluation.py \
    --mode ours \
    --gt_dir ~/local/results/ours_$1/ignore \
    --pred_dir ~/local/results/ours_$1/ignore \
    --out_dir eval/eval_ours_$1 \
    --out_file metrics/ours_$1.txt
