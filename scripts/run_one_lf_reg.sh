#!/bin/bash

python main.py \
   experiment=local \
    experiment/dataset=$1 \
    experiment.dataset.collection=$2 \
    experiment/model=$3 \
    experiment.model.param.n_dims=$4 \
    experiment.model.param.fn=$5 \
    experiment.model.embedding_net.out_channels=$6 \
    +experiment/regularizers/warp_level=lf \
    experiment/training=$7
