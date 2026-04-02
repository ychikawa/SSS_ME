#!/bin/bash

DATA_PATH=/path/to/your/imagenet

MODEL=deit_small
ALGO=diffrate
RESULT_DIR=./results_${ALGO}/
TGT_FLOPS=2.3

mkdir -p "$RESULT_DIR"

python main.py --eval \
	--model $MODEL \
	--data_path  $DATA_PATH \
	--task_type [1,0,0] \
	--algo $ALGO \
	--load_schedule \
	--tgt_flops $TGT_FLOPS \
	--benchmark \
	>> "${RESULT_DIR}/${MODEL}_TGT_FLOPS${TGT_FLOPS}.txt"
