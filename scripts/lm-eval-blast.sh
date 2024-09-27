#!/bin/bash


model_path=./outputs/BLAST-flat-cr0.5-lr2e-4-nb16/output/

wtype=blast
nb=4
targets="qkug"
cr=0.6
postfix=""

accelerate launch -m lm_eval --model $wtype \
    --model_args pretrained=$model_path \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa,openbookqa,boolq,wikitext \
    --batch_size 8 \

