#!/bin/bash

# available_gpus=0,1,2,3,4,5,6,7
AVAILABLE_GPUS=1,3,4,5,6,7
DATASET_TYPE=mix
PROMPT_TEMPLATE=vicuna

OUTPUT_TAG=llama2_13b_v1_3_${DATASET_TYPE}

n_epochs=1
DATA_ARGS="\
    --data_path /mnt/data/zjw/datasets/formatted-dataset \
    --lazy_preprocess True \
"


    # --use_cache False \
    # --use_custom_ds False \
    # --prompt_template $PROMPT_TEMPLATE

TRAIN_ARGS="\
    --model_name_or_path /home/zhaohang/zjw/models/vicuna-13b-v1.5 \
    --bf16 True \
    --num_train_epochs $n_epochs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
"

NUM_GPUS=$(($(echo $AVAILABLE_GPUS | grep -o "," | wc -l)+1))
TIME_TAG=$( date +"%y%m%d%H%M" )
SAVE_TAG=${OUTPUT_TAG}_${n_epochs}_epochs_$TIME_TAG
# --model_name_or_path /mnt/data/zjw/models/stable-vicuna-13b  \

PORT=$((($RANDOM % 1000) + 47000))
# PORT=47283

CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS torchrun --nproc_per_node=$NUM_GPUS  --master_port=$PORT fastchat/train/train.py \
    --output_dir checkpoints/$SAVE_TAG \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    $TRAIN_ARGS \
    $DATA_ARGS