# %% reload magic
%load_ext autoreload
%autoreload 2

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from model import PythiaLandmark, tokenize

# %%

MODEL_NAME = 'EleutherAI/pythia-14m'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

window_len = 128
base_model.config.update({"max_position_embeddings": window_len})
tokenizer.add_special_tokens({'sep_token': '[LANDMARK]'})
model = PythiaLandmark(model=base_model)

# %%
# Train on algorithmic dataset

import random
random.seed(42)

N = 100
train_labels, test_labels = [random.randint(0, N*10) for i in range(N)], [random.randint(0, N*10) for i in range(N)]
dataset_texts = [f"KEY: {i} VALUE: {train_labels[i]}" for i in range(N)]
test_set = [f"KEY: {i} VALUE: {test_labels[i]}" for i in range(N)]

# task: context of [KEY: <tok> VAL: <tok>] plus few-shot of [KEY: <tok> VAL: <tok>] that repeat,
# I'd like the model to learn to pay attention to previous landmarks and predict the next value
# or... I could just train on wikipedia and interpret the logits...

tokenized_train = tokenize(dataset_texts)
tokenized_test = tokenize(test_set)

# optimizer was attached to a dead model ;/
optim = torch.optim.Adam(model.parameters(), lr=1e-5)

for i in tqdm(range(1000)):
    optim.zero_grad()
    output = model(**tokenized_train, labels=tokenized_train['input_ids'])
    # TODO: Fix dataset so next-token prediction is right
    # theoretical entropy rn is log(100) = 4.6 nats per VALUE prediction, since we average over
    loss = output.loss
    loss.backward()
    optim.step()
    if i % 10 == 0:
        test_loss = model(**tokenized_test, labels=tokenized_test['input_ids']).loss
        # TODO: accuracy
        tqdm.write(f'loss: {loss} test_loss: {test_loss}')

