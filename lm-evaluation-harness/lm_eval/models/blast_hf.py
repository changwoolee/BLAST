"""
Based on https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/models/huggingface.py
"""
import numpy as np
import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)


eval_logger = utils.eval_logger
from .huggingface import _get_accelerate_args, HFLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from modeling_blast import BlastLlamaConfig, BlastLlamaModel, BlastModelForCausalLM

AutoConfig.register("blast_llama",BlastLlamaConfig)
AutoModel.register(BlastLlamaConfig, BlastLlamaModel)
AutoModelForCausalLM.register(BlastLlamaConfig, BlastModelForCausalLM)


#class BlastLinear(torch.nn.Module):
#    def __init__(self, 
#                 in_features: int, 
#                 out_features: int, 
#                 num_blocks: Union[int, Union[List, Tuple]],
#                 rank=None, 
#                 bias: bool = True, 
#                 device=None,
#                 dtype=torch.float32,
#                ) -> None:
#
#        super().__init__()
#        self.in_features = in_features
#        self.out_features = out_features
#        if isinstance(num_blocks, int):
#            num_blocks=(num_blocks, num_blocks)
#        assert len(num_blocks)==2
#        assert in_features % num_blocks[1] == 0 and out_features % num_blocks[0] == 0
#        self.num_blocks = num_blocks
#
#
#        if rank is None:
#            rank = min(in_features, out_features)
#        if isinstance(rank, float):
#            rank = int(rank * min(in_features, out_features))
#
#        self.rank = rank
#
#
#        self.B = nn.Parameter(torch.empty(num_blocks[0], out_features // num_blocks[0], rank, device=device, dtype=dtype))
#        self.C = nn.Parameter(torch.empty(num_blocks[1], rank, in_features // num_blocks[1], device=device, dtype=dtype))
#        self.D = nn.Parameter(torch.empty(num_blocks[0], num_blocks[1], rank, device=device, dtype=dtype))
#
#
#        if bias:
#            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
#        else:
#            self.register_parameter('bias', None)
#
#
#    def forward(self, x : torch.Tensor) -> torch.Tensor:
#        x_shape = x.shape
#        batch_dim = np.prod(x_shape[:-1])
#        x = x.flatten(0,-2)
#
#        x = x.view(-1, self.num_blocks[1], x.shape[-1]//self.num_blocks[1]).transpose(0,1)
#        y = torch.bmm(x, self.C.transpose(1,2)) # (nb, n, rank)
#
#        #y = y.transpose(0,2) 
#        #D = self.D
#        #z = torch.bmm(y, D.transpose(1,2)) # (rank, n, nb)
#        #z = z.transpose(0,2)
#        z = y.unsqueeze(0) * self.D.unsqueeze(2)
#        z = z.sum(1)
#
#        out = torch.bmm(z, self.B.transpose(1,2))
#        out = out.transpose(0,1).reshape(*(x_shape[:-1] + (self.out_features,)))
#
#
#        if self.bias is not None:
#            out += self.bias.to(x.dtype)
#        return out
#        







#def get_parent(model, mn):
#    parent_name = ".".join(mn.split(".")[:-1])
#    for n, m in model.named_modules():
#        if n == parent_name:
#            return m
#





@register_model("blast-hf", "blast")
class BlastHF(HFLM, TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        weight_path: Optional[str] = None,
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        TemplateLM.__init__(self)

        # optionally: take in an already-initialized transformers.PreTrainedModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self._model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                    + ["mps", "mps:0"]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = torch.device(device)

            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                device_map_option=device_map_option,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                **kwargs,
            )

        if weight_path is not None:
            self._replace_layers(weight_path, pretrained) # Assuming pretrained is str.

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        if isinstance(pretrained, str) and (gpus >= 1 or str(self.device) == "mps"):
            # TODO: can remove this whole snippet except in the mps case, perhaps?
            if not (parallelize or autogptq or hasattr(self, "accelerator")):
                # place model onto device requested manually,
                # if not using HF Accelerate or device_map
                # or any other option that preloads model onto device
                try:
                    self.model.to(self.device)
                except ValueError:
                    eval_logger.debug(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )

        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            if getattr(self.config, "model_type", None) == "qwen":
                # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
                self.tokenizer.pad_token = "<|endoftext|>"
            elif (
                self.tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
                or self.tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
            ):
                # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
                # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
                # ---
                # Note that the world tokenizer class name, might change in the future for the final huggingface merge
                # https://github.com/huggingface/transformers/pull/26963
                assert self.tokenizer.pad_token_id == 0
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # TODO: override this for Gemma
        self.add_bos_token = add_bos_token
        if getattr(self.config, "model_type", None) == "gemma":
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if parallelize:
                    if accelerator.num_processes > 1:
                        raise RuntimeError(
                            "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                        )
                    else:
                        pass
                elif accelerator.num_processes == 1:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
                else:
                    if gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                    assert (
                        accelerator.distributed_type
                        in [
                            DistributedType.FSDP,
                            DistributedType.MULTI_GPU,
                        ]
                    ), "Unsupported distributed type provided. Only DDP and FSDP are supported."
                    if accelerator.distributed_type == DistributedType.FSDP:
                        self._model = accelerator.prepare(self.model)
                    else:
                        self._model = accelerator.prepare_model(
                            self.model, evaluation_mode=True
                        )
                    self._device = torch.device(
                        f"cuda:{accelerator.local_process_index}"
                    )
                    self.accelerator = accelerator

                    if self.accelerator.is_local_main_process:
                        eval_logger.info(f"Using {gpus} devices with data parallelism")

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )


#    def _replace_layers(self, weight_path: str, path: str):
#        for mn, m in self.model.named_modules():
#            if "llama" in path:
#                if isinstance(m, torch.nn.Linear) and 'model.layers' in mn:
#                    layer_idx, weight_type = mn.split(".")[-3], mn.split(".")[-1]
#                    #if weight_type not in ("q_proj", "k_proj", "up_proj", "gate_proj"):
#                    #    continue
#                    #if weight_type in ("up_proj", "gate_proj") and int(layer_idx) < 10:
#                    #    continue
#                    w_path = os.path.join(weight_path, layer_idx, weight_type)
#                    print(f"Loading weights from {w_path}")
#                    try:
#                        B = torch.load(w_path+"-B.tensor")
#                        C = torch.load(w_path+"-C.tensor")
#                        D = torch.load(w_path+"-D.tensor")
#                        assert B.shape[0] == D.shape[1]
#                        assert C.shape[0] == D.shape[2]
#                        assert B.shape[2] == C.shape[1] == D.shape[0]
#                        assert B.shape[0]*B.shape[1] == m.weight.shape[0]
#                        assert C.shape[0]*C.shape[2] == m.weight.shape[1]
#                        r, b1, b2 = D.shape
#                        new_layer = BlastLinear(in_features=m.weight.shape[1], 
#                                                out_features=m.weight.shape[0],
#                                                num_blocks=(b1, b2),
#                                                rank=r,
#                                                bias=m.bias is not None,
#                                                device=m.weight.device,
#                                                dtype=m.weight.dtype,
#                                                )
#                        with torch.no_grad():
#                            new_layer.B.copy_(B)
#                            new_layer.C.copy_(C)
#                            new_layer.D.copy_(D.permute(1,2,0))
#                            if m.bias is not None:
#                                new_layer.bias.copy_(m.bias)
#                        parent_module = get_parent(self.model, mn)
#                        child_name = mn.split(".")[-1]
#                        parent_module.add_module(child_name, new_layer)
#                    except FileNotFoundError:
#                        print("Decomposed weight file not found. Skipping...")
#


