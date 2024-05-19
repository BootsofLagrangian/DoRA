#!/bin/bash

adapter_path=$2
base_model=$1
output_dir=$adapter_path

eval_batch_size=2

# List of datasets
datasets=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

# Loop through each dataset and run the evaluation script
for dataset in "${datasets[@]}"
do
    sbatch "$@" --gres=gpu:1 --job-name=$dataset ./eval_adapter.sh $1 $2 $dataset
done
