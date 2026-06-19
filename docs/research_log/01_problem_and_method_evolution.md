# 01. Problem And Method Evolution

## 最初的问题

最初的问题可以概括成四个问号：

```text
打什么方向？
打到哪里？
什么时候打？
用什么几何方式打？
```

对应到 LVLM hallucination mitigation，就是：

- direction：用什么 activation direction 去纠正模型。
- location：在哪些 layers / attention heads / subspaces 上干预。
- timing：prefill、decode、某些 token，还是整个生成过程都干预。
- geometry：直接 additive steering，还是 norm-preserving spherical rotation。

早期想法不是单纯提高某个 benchmark 的 accuracy，而是希望把多模态幻觉消减重新表述为一个内部控制问题：

> 幻觉不是单一静态偏差，而是不同生成阶段、不同 token、不同视觉概念之间的错配。对象类别、属性、关系幻觉可能需要不同的专家方向、不同的干预位置和不同的触发条件。

## 对幻觉本质的理解

笔记里对 hallucination 的定义逐渐收敛为：

> 幻觉是语言概率建模与事实真实性之间存在目标错位时，在知识不足、表示混淆或推理漂移条件下产生的高置信度但低真实性生成结果。

更短地说：

> 似然最优不等于真实最优。

对于多模态大模型，错误可能来自多个环节：

- 前端视觉信息不充分，更像“看错了”。
- 中段视觉-文本对齐不稳，更像“绑定错了”。
- 后端语言先验过强，更像“说过头了”。

这直接影响了 activation intervention 的设计：如果只用一个全局向量，很可能只能修一部分情况，同时破坏另一部分情况。

## 相关工作给出的启发

### ITI

ITI 的核心启发是：

- LLM 可能知道真相，但不会自然说出来。
- Truthfulness 可以在少数 attention heads 中被 probe 读出来。
- 推理时沿 truthful direction 做稀疏 activation 平移，可以提升 TruthfulQA。
- 但 truthfulness 与 helpfulness / fluency 之间存在 trade-off。

这给了项目第一个基本信念：内部 activation 里存在可以被利用的 truthfulness / factuality signal。

### AFTER

AFTER 的关键不是简单 activation editing，而是把 factual textual semantics 引入 steering：

```text
trusted:   factual text t+ + question q
untrusted: image x + same question q
direction: z_text - z_visual
```

AFTER 证明了 factual-augmented text 比普通 caption 更适合做 steering guidance。但它的 FAS 模块最终还是学习一个 general factual-guided vector，再由 QAO 学 query-specific offset。

笔记里对 AFTER 的主要疑问是：

> AFTER 学到了如何偏移激活，但没有显式学习生成过程中何时偏移、偏移哪一类。

也就是说，AFTER 的 query adaptation 是 query-level 的，而开放式生成中的幻觉风险可能是 token-level / decoding-state-level 动态变化的。

### DMAS

DMAS 的启发更接近这个项目：

- Truthfulness heads 和 visual perception heads 可能是不同子集。
- Truthfulness vector 会随语义上下文变化。
- 应该建 vector database，通过输入语义检索最匹配的 steering vector。
- 只干预 top-K heads 更稳、更省。

这说明固定全局 vector 不够，steering direction 和 intervention heads 都应该更自适应。

但 DMAS 的聚类仍偏粗，例如离散的语义簇未必覆盖连续语义变化。因此项目继续往 type-aware / subtype-aware 方向推进。

### Spherical Steering

Spherical Steering 带来的启发是：方向和长度是两类不同信息。

在它的 TruthfulQA 分析中，truthful 与 hallucinated activation 的 norm 曲线高度重合，说明 factuality 可能更多编码在 direction 上，而不是 magnitude 上。

这启发了两个问题：

1. LVLM 的 factual / hallucinated states 是否也存在 norm-overlap？
2. 如果存在，head selection 和 steering 是否应该从 L2 difference 转向 angle / cosine / rotation？

但笔记里也记录了一个重要警惕：

> Spherical Steering 主要在 LLM / TruthfulQA 场景里证明了方向更重要，不能直接假设 LVLM 的 object / attribute / relation hallucination 也满足同样几何前提。

所以 spherical rotation 被放在后续升级，而不是一开始就作为主 claim。

### Octopus

Octopus 的价值在于 token-level action selection。

它引入可学习 eye token，根据生成状态选择不同 contrastive decoding strategy，并用 DPO 训练动作选择器。对本项目的启发是：

- 幻觉缓解可以被建模为动作序列选择。
- 不是每一步都该用同一个策略。
- 正负样本可以来自 action workflow 的生成结果优劣，而不是人工标注每一步动作。

项目早期因此设计过：

```text
action = {
  0: no intervention,
  1: category expert,
  2: attribute expert,
  3: relation expert,
  4: mixed expert
}
```

但笔记也指出，steering 比 contrastive decoding 更危险，因为错误 action 可能直接破坏 hidden states。因此 DPO / router 不能太早上，必须先验证 expert vector 本身有效。

### ICT

ICT 使用 trusted / untrusted image pairs，例如原图 vs 加噪图，构造 shift vector：

```text
S = A_trusted - A_untrusted
```

它还用 SVM 判断哪些 heads 能区分 trusted / untrusted，从而选 intervention heads。

这对项目的启发是：head selection 不一定只能按 vector norm，也可以按可分性、probe AUC、因果效应来找。

## 初始方法设想：TDSS

早期方法名是 Type-aware Dynamic Spherical Steering, 简写 TDSS。

设想如下：

1. 把对象幻觉拆成三类：
   - category hallucination
   - attribute hallucination
   - relation hallucination

2. 为每一类构造事实引导 activation prototype bank：
   - cat prototypes
   - attr prototypes
   - rel prototypes

3. 做 type-specific causal head discovery：
   - Ho: category-related heads
   - Ha: attribute-related heads
   - Hr: relation-related heads

4. 推理时由 token-level router 动态选择：
   - no intervention
   - category expert
   - attribute expert
   - relation expert
   - mixed expert

5. 在对应 heads 上做 local activation intervention。

6. 后续再从 additive steering 升级到 spherical rotation。

这个设想后来被多轮实验修正：三类 raw vector 没有预期中干净，relation 尤其不稳定，head sets 也高度重合。所以当前更保守的定位是：

> type-aware factual expert steering 的诊断与原型验证，而不是完整 TDSS 已经成立。

## 方法论硬伤和自我修正

笔记里反复记录了几个风险：

### 1. 三类动作可能过粗

一个 token 可能同时有类别和属性风险，例如 red apple 同时涉及 object 和 color。强制 softmax 单选 `cat / attr / rel` 可能成为信息瓶颈。

更合理的路线可能是：

- mixture-of-experts soft routing
- subtype-level experts
- prototype retrieval
- entropy-gated intervention

### 2. Router 可能学到 token pattern 而不是 hallucination risk

如果 router 看到颜色词就选 attribute expert，看到 left/right 就选 relation expert，它学到的是词类，而不是真正 hallucination risk。

因此需要控制：

- answer-token balancing
- same word positive/negative balancing
- image-level split
- wrong-expert sanity
- 3x3 specificity matrix

### 3. Relation 幻觉不是单一类

AMBER relation 主要是 direct contact。

MME position 是 left/right/above/below 这类空间位置。

GQA relation 可能包含 holding、wearing、near、on、behind、of、with 等语义关系。

把这些全部平均成一个 `rel_all` vector，很容易互相污染。

### 4. Spherical rotation 需要先验证几何前提

如果 LVLM 的 factual / hallucinated direction 不主要体现在角度上，直接套 spherical steering 会显得像迁移 trick。

因此当前更稳的策略是：

1. 先用 additive steering 验证 type-specific vector 是否有效。
2. 再做 additive vs spherical rotation。
3. 证明 rotation 在生成质量、norm disturbance 或 representation collapse 上更稳。

## 当前最合理的研究叙事

当前可以这样讲：

> AFTER 证明 factual text 可作为正向 activation guidance；DMAS 证明 steering direction 和 heads 需要语义自适应；Octopus 证明 token-level 动作选择有价值。本项目进一步研究：这些 factual directions 是否可以按 hallucination type 或 subtype 拆成专家，并在推理时进行 type-aware activation intervention。实验发现 category expert 最稳定，attribute 有局部信号，relation 必须进一步拆 subtype；同时 raw type vectors 中存在强 shared factualization component，简单三专家假设不成立，需要更细粒度的 vector bank 与 router。
