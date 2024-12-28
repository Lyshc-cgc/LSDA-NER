#!/bin/bash

# 3 different seeds to run the model with
seeds=(22 32 42)
datasets=('conll2003' 'mit_movies' 'ace2005')

# run the model with each dataset and seed
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "Running run.py with dataset: $dataset and seed: $seed"
    # using CLI from hydra to run the model with different datasets and seeds
    # https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
    python run.py dataset="$dataset" seed="$seed"
  done
done