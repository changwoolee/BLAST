<div align="center">
 
# BLAST: Block Level Adaptive Structured Matrix for Efficient Deep Neural Network Inference

**[Changwoo Lee](http://changwoolee.github.io), [Soo Min Kwon](https://soominkwon.github.io), [Qing Qu](https://qingqu.engin.umich.edu), and [Hun-Seok Kim](https://kim.engin.umich.edu)**
 </div>

## Notice
This repo is actively updating.
* The paper is accepted to NeurIPS 2024.

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
