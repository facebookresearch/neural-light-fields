#!/bin/bash

python main.py \
   experiment=local \
    experiment/dataset=$1 \
    experiment.dataset.collection=$2 \
    experiment/model=$3 \
    experiment.model.subdivision.max_hits=16 \
    experiment.model.ray.param.n_dims=$4 \
    experiment.model.ray.param.fn=$5 \
    experiment.model.ray.embedding_net.out_channels=$6 \
    experiment/training=$7

python main.py \
   experiment=local \
    experiment/dataset=$1 \
    experiment.dataset.collection=$2 \
    experiment/model=$3 \
    experiment.model.subdivision.max_hits=32 \
    experiment.model.ray.param.n_dims=$4 \
    experiment.model.ray.param.fn=$5 \
    experiment.model.ray.embedding_net.out_channels=$6 \
    experiment/training=$7
