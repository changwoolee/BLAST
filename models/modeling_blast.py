import os
import logging

import torch
import torch.nn as nn

from transformers import PretrainedConfig, LlamaConfig, LlamaModel, LlamaForCausalLM 
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaRMSNorm
from typing import List, Union, Tuple


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BlastLlamaConfig(LlamaConfig):
    model_type = "blast_llama"
    keys_to_ignore_at_inference = ["blast_decomposed_weight_path"]
    def __init__(
        self,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        blast_rank={'q_proj': 1024, 'k_proj': 1024, 'v_proj': 1024, 'o_proj': 1024, 'gate_proj': 1488, 'up_proj': 1488, 'down_proj': 1488},
        blast_num_blocks: Union[Union[List, Tuple], int] = 4,
        indices=[i for i in range(32)],
        precompute_matrix=False,
        **kwargs,
    ):
        self.target_modules = target_modules
        self.blast_rank = blast_rank
        self.blast_num_blocks = blast_num_blocks,
        self.indices = indices
        self.precompute_matrix = precompute_matrix
        super().__init__(**kwargs)


def get_parent(model, mn):
    parent_name = ".".join(mn.split(".")[:-1])
    for n, m in model.named_modules():
        if n == parent_name:
            return m


def replace_layers_with_blast(
    model,
    target_modules,
    blast_rank,
    blast_num_blocks,
    indices,
    precompute_matrix=False,
    #blast_decomposed_weight_path,
):
    for mn, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            for tmn in target_modules:
                if tmn in mn:
                    layer_idx = int(mn.split(".")[-3])
                    if layer_idx not in indices:
                        continue
                    if isinstance(blast_rank, dict):
                        for k in blast_rank.keys():
                            if k in mn:
                                rank = blast_rank[k]
                                break
                    elif isinstance(blast_rank, int):
                        rank = blast_rank
                    elif isinstance(blast_rank, float):
                        rank = int(blast_rank * min(m.weight.shape[0], m.weight.shape[1]))
                    else:
                        raise ValueError(f"blast_rank must have either dict, int, or float type, got: {type(blast_rank)}.")

                    if isinstance(blast_num_blocks, dict):
                        for k in blast_rank.keys():
                            if k in mn:
                                num_blocks = blast_num_blocks[k]
                                break
                    elif isinstance(blast_num_blocks, int):
                        num_blocks = blast_num_blocks
                    elif isinstance(blast_num_blocks, tuple):
                        num_blocks = blast_num_blocks
                        if len(blast_num_blocks) == 1:
                            num_blocks = num_blocks[0]
                            if isinstance(num_blocks, list):
                                num_blocks = num_blocks[0]
                    else:
                        raise ValueError(f"blast_num_blocks must have either dict, int, or tuple of ints, got: {type(blast_num_blocks)}.")

                    # Load Decomposed BLAST Weights
                    new_layer = BlastLinear(
                        in_features=m.weight.shape[1], 
                        out_features=m.weight.shape[0],
                        num_blocks=num_blocks,
                        rank=rank,
                        bias=m.bias is not None,
                        device=m.weight.device,
                        dtype=m.weight.dtype,
                        precompute_matrix=precompute_matrix,
                    )
                    
                    parent_module = get_parent(model, mn)
                    child_name = mn.split(".")[-1]
                    parent_module.add_module(child_name, new_layer)

    return model



class BlastLinear(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_blocks: Union[int, Union[List, Tuple]],
                 rank=None, 
                 bias: bool = True, 
                 device=None,
                 dtype=torch.float32,
                 precompute_matrix=False,
                ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if isinstance(num_blocks, int):
            num_blocks=(num_blocks, num_blocks)
        if isinstance(num_blocks[0], list):
            num_blocks[0] = num_blocks[0][0]
        if isinstance(num_blocks[1], list):
            num_blocks[1] = num_blocks[1][0]
        assert len(num_blocks)==2
        assert in_features % num_blocks[1] == 0 and out_features % num_blocks[0] == 0
        self.num_blocks = num_blocks
        self.precompute_matrix = precompute_matrix

        if rank is None:
            rank = min(in_features, out_features)
        if isinstance(rank, float):
            rank = int(rank * min(in_features, out_features))

        self.rank = rank


        self.B = nn.Parameter(torch.empty(num_blocks[0], out_features // num_blocks[0], rank, device=device, dtype=dtype))
        self.C = nn.Parameter(torch.empty(num_blocks[1], rank, in_features // num_blocks[1], device=device, dtype=dtype))
        self.D = nn.Parameter(torch.empty(num_blocks[0], num_blocks[1], rank, device=device, dtype=dtype))


        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.rank_score = 0.

    def get_matrix(self):
        C = self.C.unsqueeze(0) # 1,b2,r,q
        D = self.D.unsqueeze(-1) # b1,b2,r,1
        DC = C*D
        DC = DC.permute(0,1,3,2).reshape(self.num_blocks[0], self.in_features, self.rank) # b1 n r
        B = self.B # b1 p r
        A = torch.bmm(B, DC.transpose(1,2))
        A = A.view(self.out_features, self.in_features)
        return A

    #@torch.compile
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        if self.precompute_matrix:
            if self.training:
                self.A = None
                A = self.get_matrix()
            else:
                if not hasattr(self, 'A') or self.A is None:
                    self.A = self.get_matrix()
                A = self.A
            out = torch.nn.functional.linear(x, A)
            
        else:

            x_shape = x.shape
            x = x.flatten(0,-2)

            x = x.view(-1, self.num_blocks[1], x.shape[-1]//self.num_blocks[1]).transpose(0,1)
            y = torch.bmm(x, self.C.transpose(1,2)) # (nb, n, rank)

            z = y.unsqueeze(0) * self.D.unsqueeze(2)
            z = z.sum(1)

            out = torch.bmm(z, self.B.transpose(1,2))
            out = out.transpose(0,1).reshape(*(x_shape[:-1] + (self.out_features,)))


        if self.bias is not None:
            out += self.bias.to(x.dtype)
        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, num_blocks={self.num_blocks}'
 

class BlastLlamaModel(LlamaModel):
    config_class = BlastLlamaConfig

    def __init__(self, config: BlastLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        logger.info("Replacing Linear Layers to BlastLiner...")
        replace_layers_with_blast(
            self.layers, 
            config.target_modules,
            config.blast_rank,
            config.blast_num_blocks,
            config.indices,
            config.precompute_matrix,
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

class BlastModelForCausalLM(LlamaForCausalLM):
    config_class = BlastLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = BlastLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

def main():
    from transformers import AutoConfig, AutoModelForCausalLM
    logging.basicConfig(level=logging.INFO)
    config = BlastLlamaConfig.from_pretrained("huggyllama/llama-7b", blast_num_blocks=4)
    print(config)
    model = BlastModelForCausalLM.from_pretrained("huggyllama/llama-7b", config=config)
    print(model)
    model.save_pretrained("./outputs/save_test/")

    config = BlastLlamaConfig.from_pretrained("./outputs/save_test/", blast_num_blocks=4)
    print(config)
    model = BlastModelForCausalLM.from_pretrained("./outputs/save_test/", config=config)
    print(model)

if __name__=='__main__':
    main()

