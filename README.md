# BLAST: Block Level Adaptive Structured Matrix for Efficient Deep Neural Network Inference

** Changwoo Lee, Soo Min Kwon, Qing Qu, and Hun-Seok Kim **

This repo is actively updating.

## Dependencies

The packages can be installed via `conda env create --file environment.yml`.

Additionally, install `lm-evaluation-harness` with BLAST implementation via 
```bash
cd lm-evaluation-harness
pip install -e .
```

## Llama Decompsotion

Run `bash ./scripts/decompose_llama.sh 0-31`.

## Blast-Llama Retraining
Run `bash ./scripts/train_blast.sh`. The script assumes that 4 gpus are available.

## Evaluation using `lm-evaluation-harness`
Run `bash scripts/lm-eval-blast.sh`.
