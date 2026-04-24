# Step 5: Typed Expert Steering Vectors

This step builds the first typed steering-vector artifact for `cat`, `attr`, and `rel`.

It is still a minimal baseline. It does not yet re-forward LLaVA on raw expert JSONL, and it does not yet use diff-span token activations. Instead, it reuses the already extracted train activation cache:

```text
z_pos/z_neg: [N, layer, head, head_dim]
```

For each row, the vector direction is:

```text
delta = z_pos - z_neg
```

The typed experts are mapped as:

```text
cat -> cat
cnt -> attr
col -> attr
rel -> rel
```

For each expert, the script averages deltas over its rows and selected layers:

```text
vector[expert][layer][head] = mean(delta[layer][head])
```

## Command

```bash
python scripts/build_expert_steering_vectors.py \
  --train-cache data/outputs/activations/llava_v15_7b/v0_mini_seed42/train/merged \
  --output-path data/outputs/steering/expert_vectors.pt \
  --layers 10-20 \
  --max-samples-per-type 2000 \
  --normalize false \
  --overwrite
```

## Outputs

```text
data/outputs/steering/expert_vectors.pt
data/outputs/steering/expert_vectors.stats.json
```

The torch file contains:

```text
vectors.cat:  [num_selected_layers, num_heads, head_dim]
vectors.attr: [num_selected_layers, num_heads, head_dim]
vectors.rel:  [num_selected_layers, num_heads, head_dim]
layers: original model layer indices, for example 10..20
stats: sample counts and norm diagnostics
config: source cache and construction settings
```

## Current Intervention Meaning

The prepared vector is factual-minus-counterfactual:

```text
positive activation - negative activation
```

When used for additive steering later, the intended intervention is:

```text
activation[layer][head] += alpha * vector[expert][layer][head]
```

Default planned intervention settings:

```text
layers: 10-20
head selection: top-K by vector norm
K: 64
alpha sweep: 0.25, 0.5, 1.0, 2.0, 4.0
router: no_filter first, then force/rule routers
apply_to: last generated token
steer_prefill: false
```

## Important Caveat

This first artifact is built from answer-last-token activation caches. A stricter version should rebuild vectors from diff-span token activations, where the span is the exact changed token region between factual and counterfactual answers.
