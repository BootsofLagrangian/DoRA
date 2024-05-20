#!/bin/bash

base_model=$1
eval_batch_size=2

shift 1

# List of datasets
datasets=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")
# Loop through each dataset and run the evaluation script
for dataset in "${datasets[@]}"
do
    sbatch -q big_qos -p suma_rtx4090 --gres=gpu:1 --job-name=$dataset ./eval_base_model.sh $1 $dataset
done
