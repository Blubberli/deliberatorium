#!/bin/bash

args=(
  --lang english
  --eval_model_name_or_path sentence-transformers/all-mpnet-base-v2
  --do_train False
  --do_eval_annotated_samples True
  --annotated_samples_in_test True
  )
python train_triplets_kialo.py "${args[@]}" "$@"