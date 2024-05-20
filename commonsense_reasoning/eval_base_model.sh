#!/bin/bash

ml purge
module load cuda/12.1
echo "START TIME: $(date)"

# adapter_name=lora
# base_model=meta-llama/Llama-2-7b-hf
base_model=$1
dataset=$2
output_dir=base-$base_model
mkdir -p $output_dir

eval_batch_size=2

# Define common variables
adapter="base"
batch_size="$eval_batch_size"
output_dir="$output_dir"

python commonsense_evaluate.py \
    --adapter $adapter \
    --dataset $dataset \
    --lora_weight "" \
    --base_model $base_model \
    --batch_size $batch_size | tee -a $output_dir/${dataset}.txt

