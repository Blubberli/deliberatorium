#!/bin/bash

args=(
  --lang english
  --eval_model_name_or_path sentence-transformers/all-mpnet-base-v2
  --eval_not_trained True
  --do_train False
  --annotated_samples_in_test True
  )
python train_triplets_kialo.py "${args[@]}" "$@"