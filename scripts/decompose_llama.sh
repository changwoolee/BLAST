#!/bin/bash

python blastify_llama.py --model_name_or_path "huggyllama/llama-7b" --target_layers $1  --output_dir ./decomposed_llama/llama7b-blast/ --type blast --num_blocks 4 --comp_ratio 0.5 --delta 0.1 --num_iter 300 
