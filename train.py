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
tokenize = partial(tokenize, window_len=window_len, tokenizer=tokenizer)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)

# %%

kv_sep_id, line_sep_id = tokenizer.encode('-->')[0], tokenizer.encode('\n\n')[0]

# TODO: Vectorize this, make batched.
def generate_dataset():
    # replacement would add conflicting kv pairs to training
    keys_idx = token_ids.float().multinomial(n_pairs, replacement=False)
    vals_idx = token_ids.float().multinomial(n_pairs, replacement=False)

    # create context as key -> value\n\n
    # TODO: Maybe this should be vectorized.
    context = []
    for i in range(n_pairs):
        context += [token_ids[keys_idx[i]], kv_sep_id, token_ids[vals_idx[i]]]
        context.append(tokenizer.sep_token_id if (i+1) % landmark_every == 0 else line_sep_id)

    # TODO: maybe it'd be good to add something here other than just a landmark
    # (a landmark is added because n_pairs % landmark_every = 0)
    assert context[-1] == tokenizer.sep_token_id
    testing_start = len(context)

    # add testing content for retrevial
    kv_perm = torch.randperm(n_pairs)
    for i in range(n_pairs):
        k = kv_perm[i]
        context += [token_ids[keys_idx[k]], kv_sep_id, token_ids[vals_idx[k]]]
        context.append(tokenizer.sep_token_id if (i+1) % landmark_every == 0 else line_sep_id)

    # add padding
    # edgecase: if len(context) % window_len == 0 and (n_pairs) % window_len != 0
    # then we have uglyness and need to pad extra before adding sep_token at the end.
    context += [tokenizer.pad_token_id] * (window_len - len(context) % window_len)
    context[-1] = tokenizer.sep_token_id
    context = torch.tensor([context])

    # make sure everything's right
    assert context.size(-1) % window_len == 0
    assert (context[:, 2::4][:, :vals_idx.size(0)] == token_ids[vals_idx]).all()
    assert (context[:, ::4][:, :keys_idx.size(0)] == token_ids[keys_idx]).all()
    assert (context[:, window_len-1::window_len] == tokenizer.sep_token_id).all()

    # Grab out the keys & values we're testing after the landmark
    # TODO: Fix shit variable names. causing me more mental ram usage than needed
    context_key_idx = torch.arange(testing_start, context.size(-1), 4)[:n_pairs]
    context_val_idx = context_key_idx + 2

    assert (token_ids[vals_idx[kv_perm]] == context[:, context_val_idx]).all()
    assert (token_ids[keys_idx[kv_perm]] == context[:, context_key_idx]).all()

    return (context, context_val_idx, vals_idx[kv_perm])


# %%

batch_size = 64

for i in tqdm(range(100000)):
    if (i+1) % batch_size == 0:
        optim.step()
        optim.zero_grad()
    
    context, ctx_val_idx, val_labels = generate_dataset()
    # context, ctx_val_idx, val_labels = context.to('mps'), ctx_val_idx.to('mps'), val_labels.to('mps')

    # TODO: set attention mask to exclude information movement between testcases,
    # could avoid weirdness with process of elimination strategies.
    output = model(context)

    # model output shifts everything right by one bc we're predicting.
    # NOTE: loss should *only reflect prediction of values from keys. nothing else*
    logprobs = torch.log_softmax(output.logits, -1)
    token_loss = -logprobs[:, :-1, :].gather(dim=-1, index=context[:, 1:, None])[..., 0]
    # so token_loss[0] is the loss predicting context[1] from context[0]
    # therefor normal indexing we need to subtract one. maybe adding a zero would be cleaner?
    loss = token_loss[:, ctx_val_idx-1].mean()
    loss.backward()

    if (i+1) % batch_size == 0:
        val_predictions = torch.argmax(output.logits[:, ctx_val_idx-1], -1)
        accuracy = (val_predictions == token_ids[None, val_labels]).float().mean()

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
# TODO: Fix this. requires key-matching the in-common keys and ignoring others.
# val_predictions = torch.argmax(output.logits[:, ctx_val_idx2-1], -1)
# accuracy = (val_predictions == token_ids[None, val_labels2]).float().mean()
# print('should be high:', accuracy)

# %%
# TODO: More ablation studies. Make utilities so I can easily swap e.g. one key in the past context
# and see how this changes logits for the corresponding prediction value.
