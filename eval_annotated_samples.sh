#!/bin/bash

args=(
  --do_eval_annotated_samples True
  --do_eval False
)
bash eval_baseline.sh "${args[@]}" "$@"
