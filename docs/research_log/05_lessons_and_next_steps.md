# 05. Lessons And Next Steps

## Lesson 1: 不要把所有涨点都当成类型专家成立

早期有些 off-diagonal 结果也会上涨，例如 attr / rel vector 在 category subset 上也可能提升。这说明：

```text
accuracy improvement != type-specific expert
```

如果一个向量在所有任务上都涨，它可能只是 shared factualization / answer calibration direction，而不是某类 hallucination expert。

因此后续必须看 3x3 specificity matrix：

| vector / eval | category | attribute | relation |
|---|---|---|---|
| category vector | high positive | neutral | neutral |
| attribute vector | neutral | high positive | neutral |
| relation vector | neutral | neutral | high positive |

目前 clean v2 没有稳定 diagonal advantage，因此不能 claim 三专家已经完全成立。

## Lesson 2: Relation 不是一个专家

Relation 至少应该拆成：

- spatial position: left / right / above / below
- depth: in front of / behind
- contact: direct contact
- interaction: holding / wearing / sitting on
- semantic relation: of / with / by 等较虚关系，可能应过滤

AMBER relation 更接近 contact。

MME position 更接近 spatial position。

GQA relation 更杂，包含大量语义关系和长尾对象。

所以统一 `rel_all` vector 很容易失败。

后续 relation route 应该是：

```text
relation router
-> position expert
-> contact expert
-> interaction expert
-> maybe semantic expert or discard
```

而不是：

```text
all relation data -> one rel vector
```

## Lesson 3: Shared component 既是问题也是资产

Raw vectors 中存在很强 shared component：

```text
cat_raw ~= global factualization direction
attr_raw ~= global factualization direction + small private component
rel_raw ~= global factualization direction + noisy private component
```

直接去掉 shared component 后，residual 数学上确实变得可分，但性能通常弱。

这说明 shared component 里包含有用信号：

- image-to-text factual correction
- yes/no calibration
- visual-textual alignment
- answer format stabilization

后续不应该简单做：

```text
use residual only
```

更合理的是：

```text
direction = alpha_global * shared + alpha_private * type_or_subtype
```

或者通过 router / retrieval 动态决定 shared 与 private 的权重。

## Lesson 4: Norm topK 不等于 causal heads

Norm topK 选到的 heads 高度重合，且集中在中层。这说明它可能选到了 shared high-energy heads。

后续 head mining 应该加入：

1. Type discriminability:
   - head activation 能否区分 factual / hallucinated。
   - probe AUC / F1。

2. Causal effect:
   - 单独干预该 head 是否改善对应 subtype。
   - 是否破坏非对应 subtype。

3. Specificity:
   - matched expert 是否优于 mismatched expert。

4. Robustness:
   - 不同 alpha / K / layer window 下是否稳定。

## Lesson 5: Router 不应太早上

早期想法是借 Octopus，用 DPO 训练 token-level router。

但现在看，router 前置条件是：

- expert vectors 本身要有稳定 matched benefit。
- wrong expert 不应显著破坏。
- alpha / gate 有可控范围。
- reward 能可靠区分 hallucination reduction 和 answer bias。

否则 router 可能只学到：

- 颜色词 -> attr
- 位置词 -> rel
- object token -> cat

而不是 hallucination risk。

## 下一步路线

### Step 1: subtype vector bank

先不急着训练 router，先把专家库做干净。

建议拆：

```text
cat_present
cat_absent
attr_count
attr_color
rel_position
rel_contact
rel_interaction
```

每个 subtype 都要有：

- data construction report
- activation statistics
- vector cosine vs others
- top head overlap
- matched benchmark result
- wrong-expert sanity

### Step 2: 3x3 or NxN specificity matrix

目标不是所有格子都涨，而是 matched diagonal clearly stronger。

对于 subtype，可以做更细矩阵：

| vector / eval | cat | count | color | position | contact |
|---|---|---|---|---|---|
| cat | high | neutral | neutral | neutral | neutral |
| count | neutral | high | maybe neutral | neutral | neutral |
| color | neutral | neutral | high | neutral | neutral |
| position | neutral | neutral | neutral | high | neutral |
| contact | neutral | neutral | neutral | neutral | high |

如果仍然没有 diagonal advantage，就不能把方法写成 type-specific expert。

### Step 3: semantic retrieval rather than hard type labels

DMAS 的聚类思路仍值得保留：

```text
question / current decoding state -> embedding
embedding -> nearest prototype cluster
cluster -> subtype vector
```

对于每个 expert 内部，不一定人为硬分所有类别，而是可以：

- 先按 hallucination family 粗分。
- family 内部再聚类。
- 推理时检索最相近 prototype。

这比一个专家只有一个平均向量更合理。

### Step 4: entropy-gated intervention

笔记里提出过一个很重要的触发逻辑：

> 幻觉往往发生在模型不确定的时候。

因此可以计算当前 token 的 logit entropy：

- entropy 低：模型很确定，默认不干预。
- entropy 高：触发 expert selection / exploration。

这比每个 token 都强行 action 更稳。

### Step 5: additive first, spherical later

不要一开始把所有创新绑在一起。

推荐顺序：

1. Additive type/subtype expert steering.
2. Head / alpha / router diagnostics.
3. Additive vs spherical rotation comparison.
4. Open-ended generation metrics: CHAIR / Hal / Cover / quality.

只有当 additive baseline 稳定后，spherical rotation 才有清楚比较对象。

## 论文或汇报里应避免的过度表述

不要写：

> We solved category, attribute, and relation hallucination with three experts.

当前证据不支持。

可以写：

> We find that factual activation steering signals differ substantially across hallucination subtypes. Category steering is relatively stable, attribute steering shows subtype-dependent gains, while relation steering requires further decomposition into contact/interaction and position families.

不要写：

> The three hallucination types correspond to disjoint attention head sets.

当前 top64 overlap 很高。

可以写：

> Type differences are not expressed as fully disjoint head sets; instead, they may lie in different directions within a shared set of middle-layer heads.

不要写：

> Residual vectors solve type specificity.

可以写：

> Shared-private decomposition reveals a dominant shared factualization component, but residual-only steering is too weak to serve as the final method.

## 当前最有价值的 research story

这个项目最有价值的地方不只是 cat vector 提升了 POPE，而是发现并系统记录了：

1. Factual activation steering 对 LVLM hallucination 有信号。
2. 这个信号被 shared factualization component 强烈主导。
3. 简单三类平均向量不够干净。
4. Relation hallucination 尤其需要 subtype 化。
5. Head specialization 可能是 shared heads + different subspaces，而不是 disjoint heads。
6. Router / DPO 只有在 expert bank 更干净后才值得上。

这条研究路线比“我做了一个 vector 提升 1%”更像科研问题。
