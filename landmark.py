"""
This file contains the code unique to landmark attention with tests.
"""

# %%

import torch
from typing import Annotated
from einops import rearrange
from torch.autograd import gradcheck

AttentionScores = Annotated[torch.Tensor, "(batch, head, seq, seq)"]

# Redefine the modified landmark_softmax function avoiding in-place operations
def landmark_softmax(
    attn_scores: AttentionScores,
    window_len: int,
):
    assert attn_scores.size(-1) == attn_scores.size(-2)
    assert attn_scores.size(-1) % window_len == 0
    seq_len = attn_scores.shape[-1]
    num_groups = seq_len // window_len

    # setup masks & constants
    same = torch.arange(seq_len)//window_len == torch.arange(seq_len).unsqueeze(1)//window_len
    landmark = (torch.arange(seq_len) % window_len == window_len-1).repeat(seq_len).reshape(seq_len, seq_len)
    normal, other = ~landmark, ~same
    min_val = torch.finfo(attn_scores.dtype).min

    # a normal token is never allowed to attend to its own landmark,
    # so we can mask the connection right away.
    attn_scores = attn_scores.masked_fill(normal.T & same & landmark, min_val)
    # likewise, landmark tokens can never attend to anything but tokens in their group (+ themselves)
    attn_scores = attn_scores.masked_fill(landmark.T & other, min_val)

    # grouped softmax. normal groups don't include landmarks
    grouped_scores = attn_scores.masked_fill(normal.T & landmark, min_val)
    grouped_scores = rearrange(grouped_scores, '... q (g w) -> ... q g w', g=num_groups, w=window_len)
    grouped_softmax = rearrange(grouped_scores.softmax(-1), '... q g w -> ... q (g w)', g=num_groups, w=window_len)
    # grouped softmax doesn't make sense on landmark queries on other tokens, so zero.
    # (previously this was done with isnan, but we don't get nan with min_val anymore.)
    grouped_softmax = torch.where(landmark.T & other, 0, grouped_softmax)
    
    # meta attention step

    meta_group_mask = normal.T & ~((same & normal) | (other & landmark))
    meta_attn_weights = attn_scores.masked_fill(meta_group_mask, min_val).softmax(-1)

    # tile meta weights. repeat_interlaced copies zeros to weights on normal in the same group as a side effect, so we add them back.
    meta_attn_weights = (
        meta_attn_weights[..., :, window_len-1::window_len].repeat_interleave(window_len, dim=-1)
        + torch.where(normal.T & same & normal, meta_attn_weights, 0.)
    )

    # we don't gate our own group
    # TODO: Figure out better names. its weird for meta weights to include the "actual weights"
    grouped_softmax = torch.where(normal.T & same & normal, 1, grouped_softmax)

    attn_weights = normal.T * (meta_attn_weights * grouped_softmax) + landmark.T * grouped_softmax
    return attn_weights


# for testing
def landmark_softmax_slow(attn_scores: AttentionScores, window_len: int):
    batched = attn_scores.ndim == 4 # (batch, head, seq, seq)
    if batched:
        assert attn_scores.size(0) == attn_scores.size(1) == 1
        attn_scores = attn_scores[0][0]

    attn_weights = torch.zeros_like(attn_scores)
    is_landmark = lambda i: i % window_len == window_len-1
    is_normal = lambda i: not is_landmark(i)
    in_same_group = lambda i, j: i // window_len == j // window_len
    landmark_idx = lambda i: i // window_len * window_len + window_len-1
    normal_group = lambda i: slice(landmark_idx(i)-window_len+1, landmark_idx(i))

    for i in range(attn_weights.shape[0]):
        if is_landmark(i):
            # softmax tokens in same group
            same_group = slice(i-window_len+1, i+1)
            attn_weights[i, same_group] = attn_scores[i, same_group].softmax(-1)
        elif is_normal(i):
            other_landmark_idx = [[j] for j in range(attn_weights.shape[1]) if is_landmark(j) and not in_same_group(i, j)]
            same_group = attn_scores[i, normal_group(i)].view(-1)
            other_landmarks = attn_scores[i, other_landmark_idx].view(-1)

            # softmax
            denom = same_group.exp().sum() + other_landmarks.exp().sum()
            same_group = same_group.exp() / denom
            other_landmarks = other_landmarks.exp() / denom
            
            for group_weight, (l,) in zip(other_landmarks, other_landmark_idx):
                attn_weights[i, normal_group(l)] = group_weight * torch.softmax(attn_scores[i, normal_group(l)], -1)

            attn_weights[i, normal_group(i)] = same_group
            

    return attn_weights if not batched else attn_weights.unsqueeze(0).unsqueeze(0)


if __name__ == '__main__':
    # test landmark softmax
    attn_scores = torch.log(torch.tensor([
        [1.,2, 3, 1, 2, 3],
        [1.,2, 3, 1, 2, 3],
        [1.,2, 3, 1, 2, 3],
        [1.,2, 3, 1, 2, 3],
        [1.,2, 3, 1, 2, 3],
        [1.,2, 3, 1, 2, 3],
    ], requires_grad=True, dtype=torch.float64))

    # landmark attention; computed manually
    want = torch.tensor([
        [1/6, 2/6, 0, 1/6, 2/6, 0],
        [1/6, 2/6, 0, 1/6, 2/6, 0],
        [1/6, 2/6, 3/6, 0, 0, 0],
        [1/6, 2/6, 0, 1/6, 2/6, 0],
        [1/6, 2/6, 0, 1/6, 2/6, 0],
        [0, 0, 0, 1/6, 2/6, 3/6],
    ], dtype=torch.float64)

    # reshape both, adding a batch & heads dimension to the beginning
    attn_scores = attn_scores.unsqueeze(0).unsqueeze(0)
    want = want.unsqueeze(0).unsqueeze(0)

    for impl in [landmark_softmax, landmark_softmax_slow]:
        attn_weights = impl(attn_scores, window_len=3)

        assert torch.allclose(attn_weights, want), f'{impl.__name__}:\nattn_weights:\n{attn_weights}\nwant:\n{want}'
        print(f'{impl.__name__} passed test!')

        gradcheck(lambda *a: impl(*a).sum(), (attn_scores, 3), eps=1e-6, atol=1e-4)
        print(f'{impl.__name__} passed gradient test!')

