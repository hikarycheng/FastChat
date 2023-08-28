# available_gpus=0,1,2,3,4,5,6,7
AVAILABLE_GPUS=6
DATASET_TYPE=mix
MODEL_TYPE=vicuna

OUTPUT_TAG=llama2_70b_lora_${DATASET_TYPE}

n_epochs=1
DATA_ARGS="\
    --data_path /mnt/data/zjw/datasets/vicuna-dataset \
    --eval_data_path /mnt/data/zjw/datasets/formatted-dataset \
    --lazy_preprocess True \
    --dataset_random_seed 1311 \
    --model_type $MODEL_TYPE \
"

LORA_ARGS="\
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora False \
"


TRAIN_ARGS="\
    --model_name_or_path /mnt/data/lpq/glm_ppo/save/sft/llama2_70b-13k-all11-fusion6_fix2_safe \
    --tf32 True \
    --fp16 True \
    --num_train_epochs $n_epochs \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
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
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    $DATA_ARGS \
    $LORA_ARGS \
    $TRAIN_ARGS \
    --gradient_checkpointing True \
    --report_to "wandb" \