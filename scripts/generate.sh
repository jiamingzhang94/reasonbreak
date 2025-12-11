#!/bin/bash

DOX_PATH=""
ADV_CSV="outputs/adv.csv"
DECODER_PATH="checkpoints/weight.pth" # downloading url: https://huggingface.co/jiamingzz/reason_break/blob/main/weights.pth

python generate.py \
    --decoder_path "${DECODER_PATH}" \
    --image_root "${DOX_PATH}" \
    --embedding_bank_path "data/embedding_bank.pth" \
    --json_path "data/json/cot_full.json" \
    --output_dir "outputs/adv" \
    --csv_path "${ADV_CSV}" \
    --epsilon "0.0627" \


