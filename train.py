# %% reload magic
%load_ext autoreload
%autoreload 2

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from model import PythiaLandmark, tokenize
from typing import List, Tuple, Callable, Optional
from functools import partial
import torch.nn.functional as F
import os

torch.random.manual_seed(0)

# %%
# Hyperparameters

MODEL_NAME = 'EleutherAI/pythia-14m'
device = 'mps'

assert 'pythia' in MODEL_NAME, 'only pythia tokenizer has the first 128 tokens as ascii'
token_ids = torch.arange(2, 96)
landmark_every = 4 # how many key -> val pairs for each landmark
window_len = 4 * landmark_every
n_pairs = 4*window_len # multiple of window_len required for now. easier to code


# %%

# TODO: some of this boilerplate should go in model.py
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

base_model.config.update({"max_position_embeddings": window_len})
tokenizer.add_special_tokens({'sep_token': '[LANDMARK]'})
model = PythiaLandmark(model=base_model)
model.to(device)
tokenize = partial(tokenize, window_len=window_len, tokenizer=tokenizer)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)

# %%

kv_sep_id, line_sep_id = tokenizer.encode('-->')[0], tokenizer.encode('\n\n')[0]

def generate_dataset(token_ids, bs: int = 1):
    # expand token ids to (bs, len(token_ids))
    token_ids = token_ids.expand(bs, token_ids.size(0))
    keys_idx = token_ids.float().multinomial(n_pairs, replacement=False)
    vals_idx = token_ids.float().multinomial(n_pairs, replacement=False)
    kv_perm = torch.randperm(n_pairs*bs).view(bs, n_pairs) % n_pairs

    # create context as key -> value\n\n
    N = n_pairs*4*2
    context = torch.zeros((bs, N))

    # key-->val\n\n
    context[:, 0:context.shape[1]//2:4] = token_ids.gather(-1, keys_idx)
    context[:, 1:context.shape[1]//2:4] = kv_sep_id
    context[:, 2:context.shape[1]//2:4] = token_ids.gather(-1, vals_idx)
    context[:, 3:context.shape[1]//2:4] = line_sep_id

    context[:, 0+context.shape[1]//2::4] = token_ids.gather(-1, keys_idx.gather(-1, kv_perm))
    context[:, 1+context.shape[1]//2::4] = kv_sep_id
    context[:, 2+context.shape[1]//2::4] = token_ids.gather(-1, vals_idx.gather(-1, kv_perm))
    context[:, 3+context.shape[1]//2::4] = line_sep_id

    # add landmarks, overrides some \n\n's but thats fine
    context[:, window_len-1::window_len] = tokenizer.sep_token_id

    # return the indexes for keys & values in context
    context_key_idx = torch.arange(N//2, context.size(-1), 4)[:n_pairs]
    context_val_idx = context_key_idx + 2
    assert (token_ids.gather(-1, vals_idx.gather(-1, kv_perm)) == context[:, context_val_idx]).all()
    assert (token_ids.gather(-1, keys_idx.gather(-1, kv_perm)) == context[:, context_key_idx]).all()

    return (context.long(), context_val_idx, vals_idx.gather(-1, kv_perm))

# %%

batch_size = 64
for i in tqdm(range(10000)):
    optim.zero_grad()
    
    context, ctx_val_idx, val_labels = generate_dataset(token_ids=token_ids, bs=batch_size)
    context, ctx_val_idx, val_labels = context.to('mps'), ctx_val_idx.to('mps'), val_labels.to('mps')

    # TODO: set attention mask to exclude information movement between testcases,
    # could avoid weirdness with process of elimination strategies.
    output = model(context)

    # model output shifts everything right by one bc we're predicting.
    # NOTE: loss only reflects prediction of values from keys. nothing else.
    logprobs = torch.log_softmax(output.logits, -1)
    token_loss = -logprobs[:, :-1, :].gather(dim=-1, index=context[:, 1:, None])[..., 0]
    # so token_loss[0] is the loss predicting context[1] from context[0]
    # therefor normal indexing we need to subtract one. maybe adding a zero would be cleaner?
    loss = token_loss[:, ctx_val_idx-1].mean()
    loss.backward()
    optim.step()

    # show loss & acc stuff
    val_predictions = torch.argmax(output.logits[:, ctx_val_idx-1], -1)
    accuracy = (val_predictions == token_ids.to(device)[None, val_labels]).float().mean()
    tqdm.write(f'loss: {loss} accuracy {accuracy}')


# %%

if not os.path.exists('algo-model.pt'):
    torch.save(model.state_dict(), 'algo-model.pt')
    torch.save(optim.state_dict(), 'algo-optim.pt')

# %%
# Interpret the model

model.load_state_dict(torch.load('algo-model.pt'))

# %%
# Test that we're actually using the previous context

context, ctx_val_idx, val_labels = generate_dataset()
context2, ctx_val_idx2, val_labels2 = generate_dataset()
context[:, :4*n_pairs] = context2[:, :4*n_pairs]

output = model(context)
val_predictions = torch.argmax(output.logits[:, ctx_val_idx-1], -1)
accuracy = (val_predictions == token_ids[None, val_labels]).float().mean()
print('should be low:', accuracy)

# predictions for swapped in context should be high
# TODO: To implement this requires key-matching the in-common keys and ignoring others.