import torch

import mamba_lm
import mamba_lm_functional
import mamba_lm_state
import time
import copy
from transformers import AutoTokenizer
import sys

def nano_time(func, inps, *, iterations=100):
    start_time = time.time_ns()

    for _ in range(iterations):
        func(inps)

    end_time = time.time_ns()

    total_time_ns = end_time - start_time
    return total_time_ns


model = mamba_lm.from_pretrained('state-spaces/mamba-130m')

modelfunctional = mamba_lm_functional.from_pretrained('state-spaces/mamba-130m')
modelstate = mamba_lm_state.from_pretrained('state-spaces/mamba-130m')

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
input_text = "test code a b c random text from transformers import AutoTokenizer model.generate(tokenizer, Mamba is a type of) ob iubijkn srgt q34 dgs u56 werfawegt sdgj"
_, h, caches = modelfunctional.init_runs(tokenizer, "")
#_ = modelstate.init_runs(tokenizer, "")
input, caches_ref = model.init_runs(tokenizer, input_text)

maxTokens=input.size(-1)
if (len(sys.argv) > 1):
    maxTokens = min(min(1, int(sys.argv[1])), maxTokens)

maxError = 0.0001
if (len(sys.argv) > 2):
    maxError = float(sys.argv[2])

with torch.no_grad():
    for i in range(maxTokens):
        ref, caches_ref = model.step(input[:, i], caches_ref)
        logits, h, caches = modelfunctional.forward(input[:, i], h, caches)
        #logits = modelstate.forward(input[:, i])

# allow a .01% error
error = abs(logits-ref)
relativeError = abs(error/ref)
index = torch.argmax(error)
largestError = torch.flatten(error)[index]
largestRelativeError = torch.flatten(relativeError)[index]
print("maximum error %f (relative error %f) at index %d, output stats: min %f max %f variance %f" % (largestError, largestRelativeError, index, torch.min(ref), torch.max(ref), torch.var(ref)))
assert largestRelativeError < maxError, "mismatch"
