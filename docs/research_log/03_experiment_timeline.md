# 03. Experiment Timeline

这份时间线按研究阶段整理，不严格等同于 Git commit 顺序。重点是记录每一阶段的假设、实验现象、结论和下一步转向。

## Phase 0: Idea proposal

最初目标是把 AFTER、DMAS、Spherical Steering 和 Octopus 的思想合起来：

```text
AFTER: factual text gives positive steering direction
DMAS: steering should be semantic-adaptive and head-specific
Spherical Steering: direction may matter more than norm
Octopus: token-level action selection can be trained by DPO
```

早期方法写成：

> Type-aware Dynamic Spherical Steering, TDSS.

早期 action space:

| action | meaning |
|---|---|
| 0 | no intervention |
| 1 | category intervention |
| 2 | relation intervention |
| 3 | attribute intervention |

早期 state 设计：

```text
state = [
  current hidden state,
  image embedding,
  text embedding,
  logit entropy,
  vision-attention stats,
  previous action,
  prototype similarities
]
```

当时最重要的判断是：

> 先别上 router，也别急着上 spherical rotation。先证明三类专家方向和专家 heads 有真实信号。

## Phase 1: Early prototype separability

早期做了一个 two-prototype / AUROC 诊断，记录在 `3.2 验证实验结果（被推翻）.md`：

| subtype | early result | early judgement |
|---|---:|---|
| cat | pairwise 0.9528, AUROC 0.8836 | strong |
| cnt | pairwise 0.9915, AUROC 0.9536 | very strong |
| col | axis AUROC 0.8785, AP 0.9173; pairwise 0.7627 | useful but weaker |
| rel | AUROC 0.50-0.52, pairwise 0.51-0.53 | random |

早期解读：

- Category counterfactuals are separable.
- Count is extremely separable.
- Color has signal but weaker.
- Relation is basically random.

这条结果后来被标注为“被推翻”，不是因为数字本身完全无意义，而是因为它更像一个早期 diagnostic，不能直接支持最终三专家 steering claim。

## Phase 2: First POPE category steering failed at high alpha

用 `cat_truth_vector` 在 POPE COCO random 500 上做 first-token margin 诊断：

```text
alpha = 4
baseline_logit_acc = 0.876
steered_logit_acc  = 0.864
wrong_to_right = 23
right_to_wrong = 29
```

关键现象：

```text
avg_delta_margin_label_yes = +2.491
avg_delta_margin_label_no  = +1.1085
```

由于 margin = yes_logit - no_logit，这说明 steering 同时提高了 yes 样本和 no 样本的 yes margin。

结论：

> 这个 cat vector 确实能推向 Yes，但对 no 样本有害。统一 category direction 不等于统一 factual direction。

这导致一个重要认识：

```text
present object: fact direction = No -> Yes
absent object:  fact direction = Yes -> No
```

笔记中记录 `cat_present_vector` 和 `cat_absent_vector` cosine 约为 `-0.753`，说明它们强烈对立。

这解释了为什么把 present / absent 混成一个平均向量会有偏置。

## Phase 3: AFTER-template cat vector began to work

后来改用 AFTER-template cat vector，在 POPE COCO 500 上出现稳定收益：

| setting | baseline | steered | delta |
|---|---:|---:|---:|
| random 500 | 0.882 | 0.896 | +0.014 |
| popular 500 | 0.870 | 0.890 | +0.020 |
| adversarial 500 | 0.800 | 0.838 | +0.038 |

最佳 alpha 通常在 1.5-2.0 附近。

结论：

> Direction is useful, but alpha too large causes over-steering.

这时 category expert 成为最稳的主线。

## Phase 4: MME hallucination subset

在 MME 的四个 yes/no hallucination 子类上测试：

```text
existence -> cat vector
count     -> attr vector
color     -> attr vector
position  -> rel vector
```

每类 60 条，Yes/No 各 30 条。

| MME category | expert | baseline | best steered | delta | conclusion |
|---|---|---:|---:|---:|---|
| existence | cat | 0.9667 | 0.9833 @ alpha=1 | +0.0167 | cat replicated |
| count | attr | 0.7167 | 0.7333 @ alpha=0.25 | +0.0167 | weak positive |
| color | attr | 0.8667 | 0.9167 @ alpha=1 | +0.0500 | strong attr signal |
| position | rel | 0.7667 | 0.7500 @ alpha=0.25 | -0.0167 | relation failed |

结论：

> AFTER-template steering is not only useful for POPE/category; cat and attr also show positive signals on MME. Relation/position fails and must be rebuilt.

## Phase 5: Relation v2

为了解决 relation failure，构造 relation v2：

- 使用 MME-style yes/no relation questions。
- trusted text 明确写出互逆关系。
- stricter bbox filtering。
- 减少 ambiguous / overlapping object pairs。

Relation v2 MME position sweep 中若干设置达到：

```text
baseline = 0.7667
best steered = 0.8000
delta = +0.0333
```

例如：

| setting | baseline | steered | delta | wrong_to_right | right_to_wrong |
|---|---:|---:|---:|---:|---:|
| rel_above expert_map top32 alpha=1.0 | 0.7667 | 0.8000 | +0.0333 | 3 | 1 |
| rel_all expert_map top128 alpha=0.5 | 0.7667 | 0.8000 | +0.0333 | 3 | 1 |
| rel_left expert_map top32 alpha=0.5 | 0.7667 | 0.8000 | +0.0333 | 2 | 0 |

当时总结为：

> rel 原来失败，但 relation v2 数据重构后在 MME position 上从负收益变成最高 +3.33%。

但后来这个结论被进一步收紧：MME position 只有 60 条，最多改变 1-3 个样本，不足以 claim position solved。

## Phase 6: Disjoint v1

新数据 `after_template_disjoint_v1`：

```text
COCO train2014
total images: 5000
cat: 1500 images
attr: 1500 images
rel: 2000 images
overlap: 0
direction: z_text - z_visual
head select: L2 norm top64
```

第一轮结果：

| setting | benchmark | baseline | steered | delta | conclusion |
|---|---|---:|---:|---:|---|
| attr top64 alpha1 | MME color | 0.8667 | 0.9000 | +0.0333 | attr expert heads useful |
| rel top64 alpha0.5 | MME position | 0.7667 | 0.7833 | +0.0167 | weak positive |
| cat top64 alpha2 | POPE random 500 | 0.8820 | 0.8780 | -0.0040 | alpha too high |

随后扫 alpha 后：

| type | best setting | benchmark | baseline | steered | delta | conclusion |
|---|---|---|---:|---:|---:|---|
| cat | top64 alpha1 | POPE random 500 | 0.8820 | 0.8980 | +0.0160 | clear positive |
| attr | top64 alpha0.25 | AMBER attribute 1000 | 0.7680 | 0.7780 | +0.0100 | small positive |
| rel | top64 alpha0.1 | AMBER relation 1664 | 0.6767 | 0.6641 | -0.0126 | negative |

结论：

> Cat and attr are writable. Rel still cannot be claimed.

## Phase 7: Disjoint v2 and GQA relation source

`after_template_disjoint_v2` 改进：

| type | images | pairs | source | construction |
|---|---:|---:|---|---|
| cat | 1500 | 3000 | COCO | present / absent object yes-no |
| attr | 1500 | 2894 | COCO | count / color factual text |
| rel | 2000 | 7934 | GQA scene graph | object-object relation yes-no |
| total | 5000 | 13828 | COCO + GQA | disjoint |

Cat 在 POPE random 仍然有效：

| alpha | acc | delta | wrong_to_right | right_to_wrong | yes_rate |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.884 | +0.002 | 2 | 1 | 0.468 |
| 0.50 | 0.886 | +0.004 | 4 | 2 | 0.462 |
| 0.75 | 0.886 | +0.004 | 6 | 4 | 0.454 |
| 1.00 | 0.882 | 0.000 | 6 | 6 | 0.450 |
| 1.25 | 0.886 | +0.004 | 9 | 7 | 0.442 |
| 1.50 | 0.896 | +0.014 | 9 | 2 | 0.456 |
| 2.00 | 0.832 | -0.050 | 25 | 50 | 0.616 |

Rel all-vector 仍不好：

- MME position: alpha 0.75 / 1.0 从 0.7667 降到 0.6833。
- AMBER relation full: alpha 1.0 从约 0.6767 降到 0.5445。
- yes_rate 持续下降，说明 rel_all 更像 answer-bias/control direction，而不是 relation truthfulness direction。

结论：

> 三类数据 disjoint 没有把 cat 做坏，但 relation all-vector 依然失败。

## Phase 8: Relation bucket diagnostic

为解释 rel_all 失败，按 bucket 重算 vectors：

```text
rel_horizontal
rel_vertical
rel_depth
rel_contact
rel_interaction
rel_semantic
rel_position_2d
rel_contact_interaction
```

AMBER relation limit 300:

| setting | alpha | steered acc | delta | wrong_to_right | right_to_wrong | yes_rate |
|---|---:|---:|---:|---:|---:|---:|
| rel_contact_interaction | 0.25 | 0.760 | +0.060 | 25 | 7 | 0.527 |
| rel_contact | 0.25 | 0.757 | +0.057 | 25 | 8 | 0.523 |
| rel_all | -0.5 | 0.750 | +0.050 | 25 | 10 | 0.537 |
| rel_all | 0.25 | 0.697 | -0.003 | 7 | 8 | 0.377 |

MME position:

```text
baseline = 0.7667
best = 0.7833
delta = +0.0167
changed = 1-3 samples
```

结论：

> rel_all 的失败不是 relation steering 完全无效，而是 relation 内部混合了不同方向。AMBER relation 更接近 contact / interaction，MME position 更接近 spatial position。至少应拆成 contact/interaction 和 position 两个 family。

## Phase 9: POPE full and baseline alignment

发现自己的 POPE baseline 比 DMAS / Octopus 高 3-5 个点，尤其 negative FP 太少。后来定位为 decode / prompt / parser 差异。

关键区别：

| item | previous baseline | aligned baseline |
|---|---|---|
| loader | official LLaVA loader | official LLaVA loader |
| prompt suffix | Please answer this question in one word. | Please answer this question with one word. |
| decoding | greedy | sampling |
| do_sample | false | true |
| temperature | 0.0 | 1.0 |
| max_new_tokens | 5 | 1024 |
| parser | first explicit yes/no | POPE/Octopus-style contains parser |

对齐后 full 3000 几乎复现 Octopus baseline：

| setting | Octopus reported Acc/F1 | Our Octopus-like Acc/F1 | gap |
|---|---:|---:|---:|
| random | 83.77 / 81.94 | 83.73 / 81.91 | -0.04 / -0.03 |
| popular | 82.57 / 80.86 | 82.60 / 80.88 | +0.03 / +0.02 |
| adversarial | 79.77 / 78.47 | 80.40 / 79.30 | +0.63 / +0.83 |

结论：

> Baseline gap mainly came from decode / prompt / parser, not POPE files.

这是后续正式 POPE 表格的重要前置工作。

## Phase 10: GQA source and shared-private decomposition

尝试用 COCO / GQA / mixed 构造 cat vectors，在 MSCOCO POPE 和 GQA POPE 上比较。

主要发现：

- COCO cat vector 在 MSCOCO POPE 上最稳。
- 在 GQA 上，COCO / GQA / mixed 差别不大。
- GQA popular 甚至会负收益。

这说明问题不只是数据来源，而是 shared direction 很强，type-specific expert 不够干净。

随后做 SVD / shared-private decomposition：

```text
global = first right singular vector of [cat_raw, attr_raw, rel_raw]
cat_res  = cat_raw  - proj_global(cat_raw)
attr_res = attr_raw - proj_global(attr_raw)
rel_res  = rel_raw  - proj_global(rel_raw)
```

诊断：

| raw vector | cosine with global |
|---|---:|
| cat_raw | 0.9543 |
| attr_raw | 0.9741 |
| rel_raw | 0.9361 |

Residual 与 global 基本正交：

| residual | cosine with global | residual/raw norm |
|---|---:|---:|
| cat_res | -0.0002 | 0.2987 |
| attr_res | -0.0002 | 0.2262 |
| rel_res | -0.0002 | 0.3518 |

但 residual steering 效果弱，常常只有噪声级提升：

```text
gqa_cat_val: +0.0067
gqa_rel_val: +0.0067
```

结论：

> Shared component really exists, and residuals are mathematically separable. But private residuals are too weak or unstable to serve as the main steering direction directly.

## Phase 11: 3x3 specificity matrix

项目后来明确需要做 3x3 specificity matrix：

| vector / eval | category-existence | attribute | relation |
|---|---|---|---|
| category vector | should improve | should not help/hurt | should not help/hurt |
| attribute vector | should not help/hurt | should improve | should not help/hurt |
| relation vector | should not help/hurt | should not help/hurt | should improve |

但当前 clean v2 / GQA diagnostic 中没有稳定 diagonal advantage。

这意味着：

> 当前 raw cat / attr / rel experts are not clean type-specific experts.

这不是终点，而是后续 subtype bank / clustering / router 的依据。
