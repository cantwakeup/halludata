# 06. Resume And Report Summary

这一页用于简历、组会汇报和项目介绍。它比前几篇更短，但仍保留关键技术信息。

## 中文简历版

可以写成：

> 面向多模态大模型幻觉消减，研究 type-aware activation steering 方法。基于 COCO / GQA 构造 category、attribute、relation 三类 image-disjoint factual pairs，抽取 LLaVA-v1.5 attention-head activation，并以 `z_text - z_visual` 构建事实专家干预向量；在 POPE、MME、AMBER、GQA 上完成 alpha sweep、expert head mining、yes-rate / wrong-to-right 诊断和 shared-private decomposition。实验显示 category expert 在 MSCOCO POPE 上稳定提升，attribute expert 在 MME count/color 上有正收益，并发现 relation hallucination 需进一步拆分为 contact / interaction / position 等 subtype。

更短一点：

> 研究 LVLM 幻觉消减中的 type-aware activation intervention，构建类别、属性、关系三类事实专家向量并在 LLaVA-v1.5 上完成 POPE / MME / AMBER / GQA 实验；验证 object hallucination steering 的稳定收益，分析 attribute / relation 中的 subtype mismatch、shared factualization component 和 expert head overlap 问题。

## 英文简历版

Long version:

> Investigated type-aware activation steering for hallucination mitigation in large vision-language models. Built image-disjoint category, attribute, and relation factual pairs from COCO and GQA, extracted LLaVA-v1.5 attention-head activations, and constructed factual expert vectors using `z_text - z_visual`. Evaluated expert steering on POPE, MME, AMBER, and GQA with alpha sweeps, expert-head mining, yes-rate diagnostics, wrong-to-right/right-to-wrong analysis, and shared-private vector decomposition. Found stable gains for category/object hallucination, partial gains for count/color attributes, and identified relation hallucination as requiring subtype-specific experts such as contact, interaction, and position.

Short version:

> Studied type-aware activation intervention for LVLM hallucination mitigation, constructing factual expert vectors for category, attribute, and relation errors and evaluating them on POPE, MME, AMBER, and GQA. Demonstrated stable object-hallucination gains and diagnosed shared factualization and subtype mismatch issues in attribute/relation steering.

## 一句话研究定位

```text
不是简单做一个 hallucination benchmark trick，而是在研究 LVLM 内部 factual activation signal 是否能按 hallucination type / subtype 拆成可路由的干预专家。
```

## 汇报用核心贡献

### 贡献 1: Type-aware factual expert construction

把 AFTER 的 general factual text steering 拆成：

- category expert
- attribute expert
- relation expert
- later subtype experts

并强调 image-disjoint、single-factor factual pair construction。

### 贡献 2: Systematic steering diagnostics

不仅看 accuracy，还看：

- F1
- precision / recall
- yes-rate
- wrong_to_right
- right_to_wrong
- changed prediction
- margin change
- vector cosine
- TopK head overlap

这让实验能区分 true hallucination mitigation 和 answer-bias shift。

### 贡献 3: Negative findings are part of the result

重要发现包括：

- `rel_all` 不可靠。
- raw cat / attr / rel vectors 高度共享。
- residual-only steering 太弱。
- norm top64 不能证明 typed causal heads。
- 3x3 expert matrix 当前没有稳定 diagonal advantage。

这些结果直接推动方法转向 subtype vector bank 和 token-level router。

## 当前最稳的数字

### Category on MSCOCO POPE full

| Setting | Regular Acc/F1 | Ours Acc/F1 |
|---|---:|---:|
| Random | 86.50 / 84.69 | 89.77 / 89.33 |
| Popular | 85.67 / 83.90 | 87.07 / 86.88 |
| Adversarial | 83.70 / 82.08 | 84.27 / 83.57 |

### Attribute examples

| Benchmark | Baseline | Best | Delta |
|---|---:|---:|---:|
| MME color | 0.8667 | 0.9167 | +0.0500 |
| MME count | 0.7167 | 0.8333 | +0.1167 |
| AMBER attribute 1000 | 0.7680 | 0.7780 | +0.0100 |

### Relation subtype signal

| Benchmark | Setting | Baseline | Best | Delta |
|---|---|---:|---:|---:|
| AMBER relation limit 300 | rel_contact_interaction | 0.700 | 0.760 | +0.060 |
| AMBER relation limit 300 | rel_contact | 0.700 | 0.757 | +0.057 |
| MME position | relation buckets | 0.7667 | 0.7833 | +0.0167 |

Relation 这部分需要保守表述：AMBER contact/interact 有信号，MME position 未解决。

## 不建议写进简历的说法

不要写：

```text
提出完整 TDSS 方法并全面解决三类幻觉。
```

原因：router / spherical rotation 还没有形成完整最终结果，relation 也不稳定。

不要写：

```text
证明 category / attribute / relation 分别对应不同 attention heads。
```

原因：Top64 overlap 很高。更准确是同一批中层 heads 中存在不同方向子空间。

## 推荐写法

可以写：

```text
构建并诊断 type-aware factual activation expert，用于分析和缓解 LVLM 中不同类型幻觉；系统发现 category steering 最稳定，attribute 具有 subtype-dependent gains，而 relation 需要进一步拆分为 contact / interaction / position 等更细专家。
```

如果要强调工程和实验能力，可以写：

```text
实现 typed factual pair construction、activation extraction、expert vector building、POPE/MME/AMBER/GQA steering evaluation、alpha sweep summarization 和 shared-private vector diagnostics，形成可复现实验流水线。
```
