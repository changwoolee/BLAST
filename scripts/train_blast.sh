#!/bin/bash

export MODEL_PATH=./decomposed_llama/llama7b-blast

export SCRIPT_ARGS=" \
--model_name_or_path huggyllama/llama-7b \
--tokenizer_name huggyllama/llama-7b \
--decomposed_weight_path $MODEL_PATH/comp0.5-nb16-ni300-delta0.1/ \
--dataset_name DKYoon/SlimPajama-6B \
--output_dir ./outputs/BLAST-flat-cr0.5-lr2e-4-nb16/output \
--num_train_epochs 1 \
--max_steps 400 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 36 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate $lr \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--do_train \
--do_eval \
--overwrite_output_dir \
--torch_dtype bfloat16 \
--bf16 \
--block_size 2048 \
--ddp_timeout 10800 \
--report_to wandb \
--num_blocks 16 \
--max_index 31 \
--precompute_matrix \
--gradient_checkpointing \
"
# change --num_processes for different number of GPUs.
accelerate launch --num_machines 1 --num_processes 4 train_blast.py $SCRIPT_ARGS

