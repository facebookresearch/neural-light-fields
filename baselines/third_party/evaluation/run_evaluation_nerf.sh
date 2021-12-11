#!/bin/bash

source ~/miniconda3/bin/activate eval

HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 \
    CUDA_VISIBLE_DEVICES=$3 python run_evaluation.py \
    --mode nerf \
    --gt_dir ~/local/results/nerf_$1/testset_$2 \
    --pred_dir ~/local/results/nerf_$1/testset_$2 \
    --out_dir eval/eval_nerf_$1 \
    --out_file metrics/nerf_$1.txt
