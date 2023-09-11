# %%
"""
This file defines the transformer model with landmark attention. (A modified version of GPTNeoX/Pythia.)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
from functools import partial
from landmark import landmark_softmax

MODEL_NAME =  'EleutherAI/pythia-14m' # Used for testing; not imported

# %%

# exactly the same as GPTNeoXAttention._attn except we swap in landmark_softmax. ideally
# we'd just override softmax, but that doesn't seem to be possible without cursed solutions.
def landmark_attn(self, query, key, value, attention_mask=None, head_mask=None, window_len=None):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    # compute causal mask from causal mask buffer
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)

    # TODO: Why do they init to zero, seems wasted
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_scores = attn_scores + attention_mask
    
    # Landmark attention!
    attn_weights = landmark_softmax(attn_scores, window_len=window_len)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


if __name__ == '__main__':
    pass # TODO: Tests

# %%


class LandmarkRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        window_len = max_position_embeddings # NOTE: May change this later. its a little cursed
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Don't do positional stuff between landmarks
        landmarks_mask = torch.arange(self.max_seq_len_cached) % window_len == window_len-1
        freqs = freqs.masked_fill(landmarks_mask.unsqueeze(-1), 0.0) # repeat over head dim

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
    

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # In landmark attention when seq_len exceeds max_seq_len_cached (same as window_len!) we need to
        # rearrange. I'm starting off with the dumb way of moving other chunks "on top" of the current chunk,
        # in terms of positional information.
        # TODO: Implement what the paper does too
        if seq_len > self.max_seq_len_cached:
            window_len = self.max_seq_len_cached
            # modify seq_len to cycle every window_len, thus it never exceeds max_seq_len_cached
            t = torch.arange(1, seq_len+1, device=x.device) % window_len
            return self.cos_cached[:, :, t, ...].to(x.device), self.sin_cached[:, :, t, ...].to(x.device)

            
        # fixed slicing for speed. see https://github.com/huggingface/transformers/issues/25813
        return self.cos_cached[:, :, :seq_len, ...].to(x.device), self.sin_cached[:, :, :seq_len, ...].to(x.device)


if __name__ == '__main__':
    pass # TODO: Tests


# %%

# TODO: Define a modified tokenizer or modify the existing one or something. Having a separate
# function you *must use* is cursed.
def tokenize(texts: List[str], window_len: int):
    """
    huggingface tokenizers are impossible to customize. I tried modifying
    post_processor but it couldn't be done easily for FastTokenizer
    """
    inputs = tokenizer(texts, add_special_tokens=True, padding=True, pad_to_multiple_of=window_len)

    # add landmark tokens, one every window_len
    for l, a in zip(inputs['input_ids'], inputs['attention_mask']):
        assert len(l) % window_len == 0, f'len(l) % window_len'

        for i in range(window_len-1, len(l), window_len):
            l.insert(i, tokenizer.sep_token_id)
            a.insert(i, 1)

        # pad to multiple of window_len
        l[:] = l + [tokenizer.pad_token_id] * ((6 - len(a) % 6) % 6)
        a[:] = a + [0] * ((6 - len(a) % 6) % 6)

        assert len(l) % window_len == 0, 'len(l) % window_len (2)'

        # truncate l and a to multiple of window_len (bad)
        # l[:] = l[:len(l) - len(l) % window_len]
        # a[:] = a[:len(a) - len(a) % window_len]

    # print([(len(l), len(l)%window_len) for l in inputs['input_ids']])
    assert all(len(l) % window_len == 0 for l in inputs['input_ids'])

    return {k: torch.tensor(v) for k, v in inputs.items()}


if __name__ == '__main__':
    window_len = 6
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'sep_token': '[LANDMARK]'})

    tokenize(["foo bar baz qux quux corge grault garply waldo fred plugh xyzzy thud"], 6)

    assert tokenize(["foo"], window_len)['attention_mask'][0, :window_len].tolist() == [1, 0, 0, 0, 0, 1]
    # TODO: make work / remove unnecessary padding in tokenize.
    # assert tokenize(["foo"], 6)['attention_mask'] == [[1, 0, 0, 0, 0, 1]]

    texts = ["Hello, my dog is cute and my cat is also cute.", "Hello, I like apples and bannanas"]
    # really we want to pad to multiple of window len + 1 for labels. but instead we have do this hack.
    inputs = tokenize(texts, window_len)
    for k in range(len(texts)):
        end = len(inputs['input_ids'][k]) - window_len # TODO: fix extra padding (see above)
        assert all(inputs['input_ids'][k, i].item() == tokenizer.sep_token_id for i in range(window_len-1, end, window_len))
        assert all(inputs['attention_mask'][k, i].item() == 1 for i in range(window_len-1, end, window_len))


# %%
# Define main model

# Modified model to use landmark attention
class PythiaLandmark(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.config = model.config
        self.max_seq_len = model.config.max_position_embeddings

        # Replace attention with landmark attention
        for layer in model.gpt_neox.layers:
            # TODO: Figure out why this assert randomly broke when it worked before.
            # assert isinstance(layer, GPTNeoXLayer), f"layer: {layer} isn't GPTNeoXLayer"
            layer.attention._attn = partial(landmark_attn, layer.attention, window_len=self.max_seq_len)
            layer.attention.rotary_emb = LandmarkRotaryEmbedding(
                layer.attention.rotary_ndims,
                self.config.max_position_embeddings,
                base=self.config.rotary_emb_base,
            )


    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            **kwargs
    ):
        # TODO: Finish inference code
        # Iterate over window size
        # window_len = self.max_seq_len
        # for step, idx in enumerate(range(0, input_ids.shape[1], window_len)):
        #     # print('STEP', step, 'idx', idx)
        #     # print('attn mask', attention_mask.shape, 'input_ids',input_ids.shape[1])
        #     outputs = self.model(
        #         input_ids=input_ids[:, : idx + window_len],
        #         attention_mask=attention_mask[:, : idx + window_len], # donno why but they add "+ attention_mask.shape[1] - input_ids.shape[1]]"
        #         # past_key_values=past_key_values,
        #         # output_attentions=True,
        #         **kwargs,
        #     )
        #     # past_key_values = outputs.past_key_values
        #     # TODO: Concat with outputs
        # return outputs
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)



if __name__ == '__main__':
    MODEL_NAME =  'EleutherAI/pythia-14m'
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.config.update({"max_position_embeddings": window_len}) # window len
    model = PythiaLandmark(model=base_model)

    input_ids, attention_mask, labels = inputs['input_ids'], inputs['attention_mask'], inputs['input_ids']
    output = model(input_ids, attention_mask=attention_mask, labels=input_ids)

    # check loss calc
    loss = F.cross_entropy(
        output.logits[:, :-1].contiguous().view(-1, output.logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
    )
    assert output.loss == loss, f"output.loss: {output.loss}, loss: {loss}"

    # overfit to a single batch; should go to ~0 quickly
    print('---- overfitting to a single batch [test] ----')

    from tqdm import tqdm
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in tqdm(range(1000)):
        optim.zero_grad()
        output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = output.loss
        loss.backward()
        optim.step()
        if i % 100 == 0:
            tqdm.write(f'loss: {loss}')

