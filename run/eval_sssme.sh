#!/bin/bash

DATA_PATH=/path/to/your/imagenet

MODEL=deit_small
ALGO=sss_me
RESULT_DIR=./results_${ALGO}/
R=20

export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p "$RESULT_DIR"

torchrun --nproc_per_node=4 -- main.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --batch-size 256 --epochs 1 \
    --output_dir ./output \
    --task_type [1,0,0] \
    --task_weight [1,0,0] \
    --algo $ALGO \
    --r $R \
    --modular \
    >> "${RESULT_DIR}/${MODEL}_R${R}.txt"
    