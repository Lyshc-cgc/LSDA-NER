#!/bin/bash

# 3 different seeds to run the model with
seeds=(22 32 42)

# run the model with each seed
for seed in "${seeds[@]}"; do
  echo "Running run.py with seed: $seed"
  # using CLI from hydra to run the model with different seeds
  # https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
  python run.py seed="$seed"
done