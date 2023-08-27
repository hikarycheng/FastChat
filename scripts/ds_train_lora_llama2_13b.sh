# available_gpus=0,1,2,3,4,5,6,7
AVAILABLE_GPUS=1,3,4,5,6,7
DATASET_TYPE=mix
PROMPT_TEMPLATE=vicuna

OUTPUT_TAG=llama2_13b_lora_${DATASET_TYPE}

n_epochs=1
DATA_ARGS="\
    --data_path /mnt/data/zjw/datasets/formatted-dataset \
    --eval_data_path /mnt/data/zjw/datasets/formatted-dataset \
    --lazy_preprocess True \
    --dataset_random_seed 1311 \
"

    # --use_cache False \
    # --use_custom_ds False \
    # --prompt_template $PROMPT_TEMPLATE \

LORA_ARGS="\
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora False \
"

TRAIN_ARGS="\
    --model_name_or_path /home/zhaohang/zjw/models/vicuna-13b-v1.5 \
    --fp16 True \
    --num_train_epochs $n_epochs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
"

NUM_GPUS=$(($(echo $AVAILABLE_GPUS | grep -o "," | wc -l)+1))

echo "Using ${NUM_GPUS} GPUs, GPU ids: ${AVAILABLE_GPUS}"

TIME_TAG=$( date +"%y%m%d%H%M" )
SAVE_TAG=${OUTPUT_TAG}_${n_epochs}_epochs_$TIME_TAG

PORT=$((($RANDOM % 1000) + 47000))

deepspeed --include=localhost:$AVAILABLE_GPUS --master_port=$PORT fastchat/train/train_lora.py \
    --deepspeed playground/deepspeed_config_s3.json \
    --output_dir checkpoints/$SAVE_TAG \
    --evaluation_strategy "steps" \
    --eval_steps 1 \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    $DATA_ARGS \
    $LORA_ARGS \
    $TRAIN_ARGS \
    --gradient_checkpointing True
    # --report_to "wandb" \

