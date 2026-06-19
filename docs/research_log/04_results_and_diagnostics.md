# 04. Results And Diagnostics

这一页只整理当前最重要的结果和诊断。它区分三类内容：

- 可以比较稳地 claim 的。
- 只能作为局部正信号的。
- 明确不能过度 claim 的。

## 1. Category expert: most reliable

Category / object existence 是目前最稳定的部分。

### POPE random 500

在 `after_template_disjoint_v1` / typed expert top64 heads 上：

| alpha | baseline | steered | delta | wrong_to_right | right_to_wrong |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.8820 | 0.8920 | +0.0100 | 6 | 1 |
| 1.00 | 0.8820 | 0.8980 | +0.0160 | 10 | 2 |
| 1.50 | 0.8820 | 0.8960 | +0.0140 | 11 | 4 |
| 2.00 | 0.8820 | 0.8780 | -0.0040 | 10 | 12 |

解读：

- `alpha=1.0` 最稳。
- `alpha=2.0` 开始过强。
- wrong_to_right 明显多于 right_to_wrong 时，结果可解释为 factual correction。

### MSCOCO POPE full, official aligned setting

在对齐 official LLaVA / Octopus-like decoding 后，category expert 在 MSCOCO POPE full 上有稳定提升：

| Dataset | Setting | Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| MSCOCO | Random | Regular | 86.50 | 97.82 | 74.67 | 84.69 |
| MSCOCO | Random | Ours-CatExpert, alpha=1.5 | 89.77 | 93.32 | 85.67 | 89.33 |
| MSCOCO | Popular | Regular | 85.67 | 95.73 | 74.67 | 83.90 |
| MSCOCO | Popular | Ours-CatExpert, alpha=1.5 | 87.07 | 88.13 | 85.67 | 86.88 |
| MSCOCO | Adversarial | Regular | 83.70 | 91.13 | 74.67 | 82.08 |
| MSCOCO | Adversarial | Ours-CatExpert, alpha=1.0 | 84.27 | 87.46 | 80.00 | 83.57 |

这个结果最适合放进汇报或简历，因为：

- 全量 3000 setting。
- baseline 对齐过。
- 指标包括 Acc / Precision / Recall / F1。
- category expert 与 POPE object existence 任务匹配。

### GQA POPE transfer

GQA 上效果弱很多：

| Dataset | Setting | Method | Accuracy | F1 |
|---|---|---|---:|---:|
| GQA | Random | Regular | 88.50 | 87.71 |
| GQA | Random | Ours-CatExpert, alpha=0.25 | 89.40 | 88.94 |
| GQA | Popular | Regular | 84.17 | 83.83 |
| GQA | Popular | Ours-CatExpert, alpha=0.25 | 84.23 | 84.39 |
| GQA | Adversarial | Regular | 81.30 | 81.44 |
| GQA | Adversarial | Ours-CatExpert, alpha=0.25 | 81.10 | 81.86 |

解读：

- random 有小幅提升。
- popular 几乎不变。
- adversarial Acc 略降但 F1 提升。
- GQA object space 和 negative sampling 更难，不能用 MSCOCO 结果直接外推。

## 2. Attribute expert: partial but real signal

Attribute 不是完全稳定，但有若干明确正信号。

### MME color

早期 MME color 上：

```text
baseline = 0.8667
best steered = 0.9167 @ alpha=1
delta = +0.0500
wrong_to_right = 3
right_to_wrong = 0
```

这是 attribute 里比较干净的一个结果。

### MME count

后续阶段性结果记录：

```text
baseline acc = 0.7167
baseline F1 ~= 0.6531
baseline yes_rate = 0.3167
FP = 3
FN = 14

attr alpha=0.5:
acc = 0.8333
delta = +0.1167
F1 = 0.8214
delta F1 = +0.1684
FP unchanged = 3
FN reduced = 14 -> 7
right_to_wrong = 0
```

这个结果说明 attr 对 count 可能有明显正收益，主要修 false negatives。

### AMBER attribute

AMBER attribute 1000 上更弱：

| alpha | baseline | steered | delta | F1 | yes_rate steer | wrong_to_right | right_to_wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.7680 | 0.7780 | +0.0100 | 0.7902 | 0.5580 | 22 | 12 |
| 0.50 | 0.7680 | 0.7740 | +0.0060 | 0.7789 | 0.5220 | 37 | 31 |
| 0.75 | 0.7680 | 0.7580 | -0.0100 | 0.7541 | 0.4840 | 48 | 58 |
| 1.00 | 0.7680 | 0.7500 | -0.0180 | 0.7368 | 0.4500 | 61 | 79 |
| 1.50 | 0.7680 | 0.7420 | -0.0260 | 0.7108 | 0.3920 | 86 | 112 |

解读：

- 小 alpha 有正收益。
- alpha 变大后持续压低 yes_rate。
- attr vector 可能带有 answer-bias / no-shift，需要更细拆 count / color / state / action。

## 3. Relation expert: not solved, but subtype signal exists

Relation 是当前最复杂的部分。

### rel_all failure

在 AMBER relation 1664 上：

| alpha | baseline | steered | delta | F1 | yes_rate steer | wrong_to_right | right_to_wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.6767 | 0.6641 | -0.0126 | 0.6315 | 0.3257 | 27 | 48 |
| 0.25 | 0.6767 | 0.6298 | -0.0469 | 0.5668 | 0.2686 | 45 | 123 |
| 0.50 | 0.6767 | 0.5685 | -0.1082 | 0.4346 | 0.1773 | 70 | 250 |
| 0.75 | 0.6767 | 0.5078 | -0.1689 | 0.2835 | 0.1010 | 83 | 364 |

解读：

- rel_all 强烈压 Yes。
- alpha 越大越糟。
- 它不是稳定 relation truthfulness direction。

### Relation bucket improvement

拆成 contact / interaction 后，AMBER relation limit 300 出现明显正收益：

| setting | alpha | steered acc | delta | wrong_to_right | right_to_wrong | yes_rate |
|---|---:|---:|---:|---:|---:|---:|
| rel_contact_interaction | 0.25 | 0.760 | +0.060 | 25 | 7 | 0.527 |
| rel_contact | 0.25 | 0.757 | +0.057 | 25 | 8 | 0.523 |
| rel_all | -0.5 | 0.750 | +0.050 | 25 | 10 | 0.537 |
| rel_all | 0.25 | 0.697 | -0.003 | 7 | 8 | 0.377 |

诊断支持：

| pair | cosine |
|---|---:|
| rel_contact vs rel_interaction | 0.9341 |
| rel_position_2d vs rel_contact_interaction | 0.8108 |
| rel_horizontal vs rel_vertical | 0.7952 |
| rel_vertical vs rel_depth | 0.9023 |

解读：

- AMBER relation 更像 contact / interaction。
- position 和 contact/interaction 不是完全同一个方向。
- rel_all 混合后方向被污染。

### MME position remains weak

MME position 上：

```text
baseline = 0.7667
best = 0.7833
delta = +0.0167
changed = 1-3 samples
```

这个不能 claim position hallucination solved。

合理表述：

> Relation expert has subtype-level signal, especially contact/interaction on AMBER relation, but spatial position remains unresolved.

## 4. Shared component dominates raw vectors

一个重要诊断是：raw cat / attr / rel vectors 高度相似。

在 typed FAS 250-image disjoint experiment 中：

| pair | global cosine |
|---|---:|
| cos(v_cat, v_attr) | 0.881471 |
| cos(v_cat, v_rel) | 0.752529 |
| cos(v_attr, v_rel) | 0.807162 |

Per-head cosine 也高：

| pair | per-head mean | median | heads >= 0.8 |
|---|---:|---:|---:|
| cat-attr | 0.9156 | 0.9464 | 917 / 1024 |
| cat-rel | 0.8306 | 0.8800 | 725 / 1024 |
| attr-rel | 0.8565 | 0.9016 | 773 / 1024 |

Top64 norm heads overlap:

| pair | intersection | union | Jaccard |
|---|---:|---:|---:|
| cat-attr | 54 | 74 | 0.7297 |
| cat-rel | 51 | 77 | 0.6623 |
| attr-rel | 51 | 77 | 0.6623 |

三方交集 48，三方 union 只有 84 个 heads。

解读：

> Raw type vectors are dominated by a shared image-to-factual-text correction / factualization component.

这解释了为什么 off-diagonal vectors 有时也能涨。

## 5. Shared-private decomposition is mathematically clean but not sufficient

SVD / global residual 分解：

```text
global = first singular direction of [cat_raw, attr_raw, rel_raw]
residual = raw - projection_on_global(raw)
```

Raw vectors 与 global 高度相似：

| raw vector | cosine with global |
|---|---:|
| cat_raw | 0.9543 |
| attr_raw | 0.9741 |
| rel_raw | 0.9361 |

去掉 global 后，residual 与 global 基本正交：

| residual | cosine with global | residual/raw norm |
|---|---:|---:|
| cat_res | -0.0002 | 0.2987 |
| attr_res | -0.0002 | 0.2262 |
| rel_res | -0.0002 | 0.3518 |

Residual 之间变成负相关：

| pair | cosine |
|---|---:|
| cat_res vs attr_res | -0.1700 |
| cat_res vs rel_res | -0.7519 |
| attr_res vs rel_res | -0.5219 |

但 steering 结果弱，常见提升只有 +0.0033 / +0.0067 这种噪声级。

结论：

> 类型差异确实藏在 shared component 后面，但直接拿 residual 当 steering vector 不够稳。global 里也包含真正有用的 correction signal，简单剥离会损失有效成分。

## 6. Head selection issue

Norm topK 的问题越来越明显。

部分诊断：

| pair | vector cosine | Top64 overlap | Jaccard |
|---|---:|---:|---:|
| cat-attr | 0.737 | 45/64 | 0.542 |
| cat-rel | 0.799 | 50/64 | 0.641 |
| attr-rel | 0.673 | 46/64 | 0.561 |

Top layers 主要集中在 12-16，尤其 layer 14：

```text
cat:  layer 14 has 13 heads
attr: layer 14 has 13 heads
rel:  layer 14 has 12 heads
```

解读：

> Norm topK 选到的可能是一批 shared middle-layer high-response heads，而不是 type-specific causal heads。

后续需要：

- probe AUC / F1
- causal intervention score
- answer-balanced head mining
- type-specific specificity matrix
- possibly soft masks rather than hard mutually exclusive sets

## 7. Alpha and yes-rate are essential diagnostics

很多结果不能只看 accuracy。

例如：

- cat vector 在 alpha 合适时修正 object existence，但 alpha 大会过强。
- attr vector alpha 大时持续压 yes_rate。
- rel_all 正向 alpha 强烈压 Yes，导致大量 right_to_wrong。

所以每个实验都应该记录：

- accuracy
- F1
- yes_rate
- precision / recall
- wrong_to_right
- right_to_wrong
- changed prediction
- average logit margin by label

其中 yes_rate 和 wrong_to_right/right_to_wrong 对判断 steering 是不是在“真修幻觉”尤其关键。

## 当前可写结论

保守但准确的结论是：

1. Category expert 对 object hallucination 有稳定正收益，尤其 MSCOCO POPE。
2. Attribute expert 对 count/color 有局部正收益，但需要拆 subtype 和控制 yes-rate。
3. Relation all-vector 不成立；relation 必须拆成 contact/interaction/position 等 subtype。
4. Raw type vectors 存在强 shared factualization component，简单三专家没有稳定 diagonal advantage。
5. Head specialization 可能不表现为完全不同 heads，而是同一批中层 heads 里的不同 direction subspaces。
