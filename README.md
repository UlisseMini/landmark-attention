## Landmark Attention

Reproduction of "[Landmark Attention: Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300)"

The typical benefit of landmark attention is extrapolation, typically tested by sparse key retrieval generalizing. For example, with a long python file:

```python
# ... many lines of code ...
assert key == "afj091upgjas"
# ... many lines of code ...
```

The long-context model would output `afj091upgjas`, despite being trained on far smaller files. Landmark attention is designed to make extrapolation like this more likely.

Instead of the long-context sparse retrieval task I experimented with a dense key-value matching task that looks like this:

```
=== definitions
a --> z
b --> p
p -> w
...
=== evaluation
b --> ?
p --> ?
```

(Randomized definitions followed by evaluation in a different order)

I was able to get good performance despite small landmark groups (one group per 4 definitions), validating information movement between groups.

![](images/accuracy.png)
![](images/loss.png)

Haven't gotten extrapolation working yet. With my naive implementation of the key-value task you run out of keys, and generalizing to tokens that haven't been seen before in training doesn't work well.

TODO: Fix training loop and test extrapolation.
