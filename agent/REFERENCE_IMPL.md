# Reference implementations (citation-style)

Paper-faithful reference snippets for high-implementation-variance method cards.
POLICY (mirrors the bench constitution): snippets may contain only the canonical
algorithm as published — general principles, never campaign-measured numbers,
tuned dial values from any past run, or state-specific answers. They are shown
to the proposer ONLY for the selected card, as a fidelity aid; the proposer must
adapt them to the parent script rather than paste them verbatim.

### bpr-hybrid: reference implementation
```python
# BPR (Rendle et al. 2009): maximize sigma(s_pos - s_neg) over within-user pairs;
# hybrid keeps a pointwise BCE term for stabilization (RankTower-style).
# Pairs are formed ONLY inside one user's impressions.
pos_idx, neg_idx = [], []
for u, rows in rows_by_user.items():          # precomputed index lists per user
    pos = [i for i in rows if y[i] == 1]
    neg = [i for i in rows if y[i] == 0]
    if not pos or not neg:
        continue                              # one-class users: BCE term only
    for i in pos:
        js = rng.choice(neg, size=min(n_neg_per_pos, len(neg)), replace=False)
        pos_idx += [i] * len(js); neg_idx += list(js)

s = model(x)                                  # raw scores/logits, shape [N]
bpr = -torch.nn.functional.logsigmoid(s[pos_idx] - s[neg_idx]).mean()
bce = torch.nn.functional.binary_cross_entropy_with_logits(s, y_float)
loss = alpha * bpr + (1 - alpha) * bce        # alpha=0.5 is the canonical mix
```

### dcn-lite: reference implementation
```python
# DCNv2 cross layer (Wang et al. 2021): x_{l+1} = x0 * (W x_l + b) + x_l
# over the concatenated field embeddings; then a small MLP; sum heads.
class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias=True)
    def forward(self, x0, xl):
        return x0 * self.w(xl) + xl           # elementwise product with x0

e = torch.cat(field_embeddings, dim=-1)       # [B, n_fields*k]
x = e
for layer in cross_layers:                    # 1-2 layers; deeper overfits small data
    x = layer(e, x)
logit = cross_head(x).squeeze(-1) + mlp_head(e).squeeze(-1) + first_order(feats)
```

### context-stratified-pairs: reference implementation
```python
# Hard context stratification for BPR negatives: a fraction rho of negatives is
# drawn from the SAME context as the positive (fallback tiers when the context
# has no opposite-label row). Total negatives per positive stays unchanged.
def sample_neg(i, user_rows, rng, rho):
    ctx = context_key(i)                       # e.g. (user, date, hour) or (user, date, tab)
    if rng.random() < rho:
        for tier in (neg_by_ctx.get(ctx), neg_by_day.get(day_key(i)), user_neg):
            if tier:                           # tiered fallback: context -> day -> any
                return rng.choice(tier)
    return rng.choice(user_neg)                # uniform within-user negative
# Loss is unchanged BPR/hybrid; only the negative sampler changes.
```

