from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import mamba_functional as mamba_impl
from mamba_original_generation_utils import *

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

@dataclass
class MambaLMFunctionalConfig(mamba_impl.MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> (mamba_impl.MambaConfig):
        mamba_config_fields = {field.name for field in fields(mamba_impl.MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return mamba_impl.MambaConfig(**filtered_dict)

# adapted from https://github.com/johnma2006/mamba-minimal
def from_pretrained(name: str, layers=None):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.

    Args:
        name: As of now, supports
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: a Mamba model configured with the proper parameters and initialized with the proper weights
    """   

    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))
                
    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
    
    # copy config data
    config_data = load_config_hf(name)
    strict = False
    if (layers is None):
        layers = config_data['n_layer']
        strict = True

    config = MambaLMFunctionalConfig(d_model=config_data['d_model'], n_layers=layers, vocab_size=config_data['vocab_size'])

    model = MambaLMFunctional(config)

    # copy weights
    state_dict = load_state_dict_hf(name)

    new_state_dict = {}
    for key in state_dict:
        if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
            new_key = key.replace('backbone.', '')
        else:
            new_key = key.replace('backbone', 'mamba')

        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict, strict=strict)

    return model

class MambaLMFunctional(nn.Module):
    def __init__(self, lm_config: MambaLMFunctionalConfig):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = mamba_impl.Mamba(self.config)
        self.norm_f = mamba_impl.RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input):
        # token : (B)
        # logits : (B, vocab_size)


        x = self.embedding(input)

        x, h, caches = self.mamba.forward(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, h, caches

    def step(self, input, h, caches):
        # token : (B)
        # logits : (B, vocab_size)


        x = self.embedding(input)

        x, h, caches = self.mamba.step(x, h, caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, h, caches
    
    def init_caches(self, device:str = "cpu", batch:int = 1, dtype=torch.float32):
        h = torch.zeros(len(self.mamba.layers), batch, self.config.d_inner, self.config.d_state, device=device, dtype=dtype)
        caches = torch.zeros(len(self.mamba.layers), batch, self.config.d_inner, self.config.d_conv-1, device=device, dtype=dtype)
        return h, caches


    def init_runs(self, tokenizer, prompt: str, dtype=torch.float32):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device, dtype=dtype) # (1, num_tokens)
        h, caches = self.init_caches(input_ids.device, input_ids.size(0), dtype)
        return input_ids, h, caches

    # TODO temperature
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, sample: bool = True, top_k: int = 40):
        self.eval()

        input_ids, h, caches = self.init_runs(tokenizer, prompt, num_tokens, sample, top_k)

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                (next_token_logits, h, caches) = self.step(input_ids[:, i], h, caches) # (1, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits, dim=-1) # (1, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (1, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (1)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (1)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        output = [tokenizer.decode(output.tolist()) for output in input_ids][0]

        self.train()

        return output
    
    # inspired by original mamba generate (https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py) and generate above
    # TODO temperature
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def benchmark (
            self,
            input_ids,
            max_length,
            output_scores: bool = True,
            top_k: int = 1,
            top_p=0.0,
            min_p=0.0,
            temperature=1.0,
            repetition_penalty=1.0,
            dtype = torch.float16,
            crosscheck: bool = False,
            atol: float=1e-5,
            rtol: float=1e-5,
            sequential_scan: bool = False
        ):
        assert not (crosscheck and sequential_scan)
        torch.set_grad_enabled(False)
        self.eval()

        if (not sequential_scan):
            y, h, caches = self.forward(input_ids)
            next_token_logits = y[:, -1]

        start = input_ids.shape[1]
        input_length = start
        if (crosscheck or sequential_scan):
            h_step, caches_step = self.init_caches(device=input_ids.device, dtype=dtype, batch=input_ids.size(0))
            if crosscheck:
                logits = torch.zeros(input_ids.size(0), max_length, self.embedding.num_embeddings, device=input_ids.device, dtype=dtype)
            start = 0
        if sequential_scan:
            h = h_step
            caches = caches_step
        
        for i in range(start, max_length):
            if (i >= input_length):
                next_token = sample(next_token_logits, top_k, top_p, min_p, temperature)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            # forward the new output, get new cache
            if (crosscheck):
                (next_token_logits_crosscheck, h_step, caches_step) = self.step(input_ids[:, i], h_step, caches_step) # (batch, vocab_size), caches
                logits[:, i, :] = next_token_logits_crosscheck
                if (i >= input_length): # compute using caches from parallel version too to make sure the output computations are all the same
                    (next_token_logits, h, caches) = self.step(input_ids[:, i], h, caches) # (batch, vocab_size), caches
                    y = torch.cat([y, next_token_logits.unsqueeze(1)], dim = 1)
                if (i == input_length - 1):
                    # before computing the first output token, make sure the caches and hidden layers match
                    assert torch.allclose(h, h_step, atol=atol, rtol=rtol)
                    assert torch.allclose(caches, caches_step, atol=atol, rtol=rtol)
            else:
                (next_token_logits, h, caches) = self.step(input_ids[:, i], h, caches) # (batch, vocab_size), caches


        if (crosscheck):
            assert torch.allclose(y, logits, atol=atol, rtol=rtol)
        return input_ids
