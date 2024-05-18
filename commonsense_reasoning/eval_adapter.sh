#!/bin/bash

ml purge
module load cuda/12.1
echo "START TIME: $(date)"


# GPU_IDS=$CUDA_VISIBLE_DEVICES
# IFS=',' read -ra elements <<< "$GPU_IDS" # split and read 
# DEVICE_COUNT=${#elements[@]}

# adapter_name=lora
# base_model=meta-llama/Llama-2-7b-hf
adapter_path=$2
base_model=$1
output_dir=$adapter_path

eval_batch_size=2

# Define common variables
adapter="LoRA"
batch_size="$eval_batch_size"
lora_weights="$output_dir"
output_dir="$output_dir"

# List of datasets
datasets=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

# Loop through each dataset and run the evaluation script
for dataset in "${datasets[@]}"
do
    python commonsense_evaluate.py \
        --adapter $adapter \
        --dataset $dataset \
        --base_model $base_model \
        --batch_size $batch_size \
        --lora_weights $lora_weights | tee -a $output_dir/${dataset}.txt
done
