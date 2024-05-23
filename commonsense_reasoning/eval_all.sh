#!/bin/bash

adapter_path=$2
base_model=$1
output_dir=$adapter_path

shift 2

eval_batch_size=2

# List of datasets
datasets=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

# Loop through each dataset and run the evaluation script
for dataset in "${datasets[@]}"
do
    sbatch "$@" --gres=gpu:1 --job-name=eval-$dataset ./eval_adapter.sh $base_model $adapter_path $dataset
    sleep 1
done
