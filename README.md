<div align="center">
 
# BLAST: Block Level Adaptive Structured Matrix for Efficient Deep Neural Network Inference

**[Changwoo Lee](http://changwoolee.github.io), [Soo Min Kwon](https://soominkwon.github.io), [Qing Qu](https://qingqu.engin.umich.edu), and [Hun-Seok Kim](https://kim.engin.umich.edu)**

<img src="https://github.com/changwoolee/BLAST/blob/main/imgs/blast.png?raw=true" alt="blast" width="200"/>

</div>

## Notice
This repo is being actively updated.
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


## Acklowledgment

This repo is highly inspired by [huggingface/transformers](https://github.com/huggingface/transformers/tree/main).

## Citation

Please cite our paper if you find this repo or our paper useful
```
@inproceedings{
    lee2024blast,
    title={{BLAST}: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference},
    author={Lee, Changwoo and Kwon, Soo Min and Qu, Qing and Kim, Hun-Seok},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
}
```

