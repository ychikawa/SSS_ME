#!/bin/bash

DATA_PATH=/path/to/your/imagenet

MODEL=deit_small
ALGO=sss
RESULT_DIR=./results_${ALGO}/
R=20

mkdir -p "$RESULT_DIR"

python main.py \
	--eval \
	--model $MODEL \
	--data_path  $DATA_PATH \
	--task_type [1,0,0] \
	--algo $ALGO \
	--r $R \
	--benchmark \
	>> "${RESULT_DIR}/${MODEL}_R${R}.txt"
