import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ops.blast_ops import blast_precond_gd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Decompose the weights of Llama by Blast")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Type of the factorization",
        required=True,
    )
    parser.add_argument(
        "--target_layers",
        nargs='+',
        help="Layer indices to decompose.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="A path to directory for saving the decomposed weights.",
        required=True,
    )
    parser.add_argument(
        "--comp_ratio",
        type=float,
        default=0.8,
        help="Compression ratio."
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=300,
        help="Number of GD iterations."
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=16,
        help="Number of blocks of a Blast matrix."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Delta parameter of Blast factorization."
    )
    args = parser.parse_args()
    return args
   


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    ranks = {"q_proj": 1024, "k_proj": 1024, "v_proj": 1024, "o_proj": 1024, "gate_proj": 1488, "up_proj": 1488, "down_proj": 1488,}
    targets = ranks.keys()

    if args.type == 'blast':
        args.output_dir = os.path.join(args.output_dir, f"comp{args.comp_ratio}-nb{args.num_blocks}-ni{args.num_iter}-delta{args.delta}")
    else:
        raise NotImplementedError()
    device = torch.device("cuda")
    target_layers = []
    for tl in args.target_layers:
        if '-' in tl:
            start, end = tl.split('-')
            target_layers += [i for i in range(int(start), int(end)+1)] 
        else:
            target_layers.append(int(tl))
    target_layers = sorted(target_layers)
    print("Target Layers: ", target_layers)

    for mn, m in model.named_modules():
        for i in target_layers:
            if f".{i}." not in mn:
                continue
            for t in targets:
                if t in mn:
                    target_weight = m.weight
                    assert len(m.weight.shape)==2
                    M, N = target_weight.shape
                    if t in ("gate_proj", "up_proj", "down_proj"):
                        r = ranks[t]

                    elif t in ("q_proj", "k_proj"):
                        r = ranks[t] 
                    elif t in ("v_proj"):
                        r = ranks[t]
                    else:
                        r = ranks[t] 
                    comp_ratio = r*(M+N+args.num_blocks**2) / M / N


                    if args.type == 'blast':

                        print(f"Start decomposing {mn}--compression ratio={comp_ratio}, num_blocks={args.num_blocks}, r={r}")
                        B,C,D = blast_precond_gd(target_weight,
                                                 num_blocks=args.num_blocks,
                                                 r=r,
                                                 T=args.num_iter,
                                                 device=device,
                                                 delta=args.delta,
                                                 end_factor=0.0,
                                                 verbose=True,
                                                 )
                        output_dir = os.path.join(args.output_dir, f"{i}")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            

                        filename = f"{t}"
                        torch.save(B, os.path.join(output_dir, filename+"-B.tensor"))
                        torch.save(C, os.path.join(output_dir, filename+"-C.tensor"))
                        torch.save(D, os.path.join(output_dir, filename+"-D.tensor"))
                        break
                    else:
                        raise NotImplementedError()






if __name__=='__main__':
    main()
