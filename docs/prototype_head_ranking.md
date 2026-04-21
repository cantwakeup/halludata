# Prototype And Head Ranking Scaffold

This document describes the current offline prototype and head-ranking pilot in `halludata`.

## Status

The current implementation is a scaffold, not the final real-model pipeline.

- `scripts/build_prototypes.py` reads balanced pairs and computes subtype-level positive and negative prototypes.
- `scripts/run_head_ranking.py` reads balanced pairs and computes simplified separation scores for each layer-head.
- Both scripts default to `MockActivationAdapter`, which generates deterministic pseudo-random activations so the pipeline can run end to end without binding to a real LVLM.

## Current Files

- `src/expert_data/model_adapter.py`
- `src/expert_data/prototypes.py`
- `src/expert_data/head_ranking.py`
- `scripts/build_prototypes.py`
- `scripts/run_head_ranking.py`

## How To Swap In A Real LVLM Adapter

The intended future path is to keep the offline pipeline unchanged and replace only the adapter.

The real adapter should preserve the same interface shape:

- inputs:
  - image path or image id
  - question
  - response
- output:
  - `layer_head_vectors`
  - keyed by layer-head ids such as `l0_h0`, `l10_h23`

An ideal future adapter should implement:

```python
class BaseActivationAdapter:
    def encode_pair(self, image_id, question, response, *, pair_id, subtype, branch):
        ...
```

with output like:

```python
{
  "layer_head_vectors": {
    "l0_h0": [...],
    "l0_h1": [...],
  }
}
```

## Current Scoring Logic

The current head-ranking pilot uses a simplified separation score:

- `sep`: distance between positive and negative head means
- `disp_pos`: within-positive dispersion
- `disp_neg`: within-negative dispersion
- `score`: `sep / (disp_pos + disp_neg + eps)`

This is meant to be lightweight and stable enough for early offline debugging, not a final research metric.

## Output Files

- `data/outputs/prototypes_v0.json`
- `data/outputs/head_ranking_v0.json`

These outputs are designed to be easy to inspect offline while keeping the model-binding layer separate.

