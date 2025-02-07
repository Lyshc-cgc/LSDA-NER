#!/bin/bash

# 3 different seeds to run the model with
train=false
test=true
seeds=(22 32 42)
datasets=('conll2003' 'mit_movie' 'ace2005')  # 'conll2003' 'mit_movie' 'ace2005'
subset_sizes=(0.1 0.2 0.3 0.4 0.5)
k_shots=(50 20 10 5)
augmentation='lsp_all'  # baseline, lsp, lsp_all
partition_times=(3)
# run the model with each dataset and seed
for dataset in "${datasets[@]}"; do
  for subset_size in "${subset_sizes[@]}"; do
    for partition_time in "${partition_times[@]}"; do
      for k_shot in "${k_shots[@]}"; do
        for seed in "${seeds[@]}"; do
          echo "Running run.py with:"
          echo "dataset: $dataset "
          echo "k_shot: $k_shot"
          echo "seed: $seed"
          echo "augmentation: $augmentation"
          # using CLI from hydra to run the model with different datasets and seeds
          # https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
          python run.py train=$train \
          test=$test \
          dataset="$dataset" \
          k_shot="$k_shot" \
          training_args="${k_shot}_shot" \
          seed="$seed" \
          augmentation=$augmentation
        done
      done
    done
  done
done