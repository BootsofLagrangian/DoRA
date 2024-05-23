#!/bin/bash

ml purge
module load cuda/12.1
echo "START TIME: $(date)"


GPU_IDS=$CUDA_VISIBLE_DEVICES
IFS=',' read -ra elements <<< "$GPU_IDS" # split and read 
DEVICE_COUNT=${#elements[@]}

adapter_name=seal
base_model=meta-llama/Meta-Llama-3-8B
output_dir=$base_model-$adapter_name-r$1-alpha$2-random
wandb_run_name=$base_model-$adapter_name-r$1-alpha$2-random

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info

MASTER_ADDR=localhost
MASTER_PORT=22555

NCCL_P2P_DISABLE=1 accelerate launch --config_file=./ddp.yaml  \
    --num_processes $DEVICE_COUNT \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --tee 3 \
    finetune.py \
    --base_model $base_model \
    --data_path 'commonsense_170k.json' \
    --output_dir $output_dir \
    --batch_size 16  --micro_batch_size 2 --num_epochs 3 \
    --learning_rate 2e-5 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name $adapter_name \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing \
    --wandb_project='seal-exp' --wandb_run_name=$wandb_run_name \
    --key_list '["./keys/random_100_32_32.npy"]'

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset boolq \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/boolq.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset piqa \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/piqa.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset social_i_qa \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/social_i_qa.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset hellaswag \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/hellaswag.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset winogrande \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/winogrande.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset ARC-Challenge \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/ARC-Challenge.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset ARC-Easy \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/ARC-Easy.txt

# python commonsense_evaluate.py \
#     --model LLaMA-7B \
#     --adapter LoRA \
#     --dataset openbookqa \
#     --base_model $base_model \
#     --batch_size 1 \
#     --lora_weights $3|tee -a $3/openbookqa.txt