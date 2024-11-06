<div align="center">
 
# BLAST: Block Level Adaptive Structured Matrix for Efficient Deep Neural Network Inference

**[Changwoo Lee](http://changwoolee.github.io), [Soo Min Kwon](https://soominkwon.github.io), [Qing Qu](https://qingqu.engin.umich.edu), and [Hun-Seok Kim](https://kim.engin.umich.edu)**

University of Michigan

<img src="https://github.com/changwoolee/BLAST/blob/main/imgs/blast.png?raw=true" alt="blast" width="200"/>

**[[Paper](https://arxiv.org/abs/2410.21262)]**

</div>

## Notice
This repo is being actively updated.
* [Blast-Llama-4B](https://huggingface.co/cwoolee/blast-llama-4B) is now available on Hugging Face! 🤗 
* [arXiv](https://arxiv.org/abs/2410.21262) version is available!
* The paper is accepted to NeurIPS 2024.

## Dependencies

The packages can be installed via `conda env create --file environment.yml`.

Additionally, install `lm-evaluation-harness` with BLAST implementation via 
```bash
cd lm-evaluation-harness
pip install -e .
```

## Blast-Llama-4B Model

Blast-Llama-4B is a Llama-7B model compressed by 50% via the procedure described below.
The model can be loaded using `transformers` library.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("cwoolee/blast-llama-4B", trust_remote_code=True)
```

## Llama Decompsotion

Run `bash ./scripts/decompose_llama.sh 0-31`.

## Blast-Llama Retraining
Run `bash ./scripts/train_blast.sh`. The script assumes that 4 gpus are available.

We re-trained the compressed Llama model for 400 steps on a subset of SlimPajama dataset available at [here](https://huggingface.co/datasets/DKYoon/SlimPajama-6B).

## Evaluation using `lm-evaluation-harness`
Run `bash scripts/lm-eval-blast.sh`.


## Acklowledgment

This repo is highly inspired by [huggingface/transformers](https://github.com/huggingface/transformers/tree/main) and [EleutherAI/lm-evaluation-harness
](https://github.com/EleutherAI/lm-evaluation-harness).

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

