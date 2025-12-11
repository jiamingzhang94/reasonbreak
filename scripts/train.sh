#!/bin/bash

CHECKPOINT="ckpts/pre-trained.pt" # pre-trained.pt downloading url: https://huggingface.co/jiamingzz/anyattack/blob/main/checkpoints/pre-trained.pt
EPOCHS=2
JSONL_PATH="data/json/location_analysis_fixed.jsonl"
GEO6K_PATH=""
EMBEDDING_BANK_PATH="data/embedding_bank.pth"
SAVE_DIR="./checkpoints"
BATCH_SIZE=1
LR=1e-5
EPSILON=0.0627
MAX_NUM_BLOCKS=64



python train.py \
    --pretrained_decoder "$CHECKPOINT" \
    --jsonl_path "$JSONL_PATH" \
    --image_root "$GEO6K_PATH" \
    --embedding_bank_path "$EMBEDDING_BANK_PATH" \
    --save_dir "$SAVE_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --epsilon "$EPSILON" \
    --max_num_blocks "$MAX_NUM_BLOCKS" \
    --use_fp16 \
    --verbose 

