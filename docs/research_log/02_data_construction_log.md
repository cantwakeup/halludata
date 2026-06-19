# 02. Data Construction Log

## 核心原则

数据构造的目标不是让样本“像自然用户问题”，而是让变量干净。

早期笔记里反复强调三点：

1. 不要让 relation 数据退化成 object existence。
   - 例如问 `Is the cat under the table?`，如果图里没有 cat，那不是 relation hallucination，而是 category hallucination。

2. 不要让 attribute 数据退化成 category。
   - 例如问 `Is there a red bus?`，模型可能只是在判断 bus 是否存在，而不是 red 是否正确。

3. 不要直接拿原始 benchmark 问题学 expert。
   - 原始问题混有问法长度、yes/no 形式、counting 形式、relation template 等偏置，activation difference 里会混进 question-style bias，而不是 hallucination-type signal。

所以构造阶段更强调 single-factor factual construction。

## AFTER-template extraction

这一项目大多数向量构造都沿用了 AFTER 的 FAS 思路：

```text
trusted   : factual textual description t+ + question q
untrusted : raw image x + same question q
direction : z_text - z_visual
```

更具体地：

```text
z_text_i[l,h]   = trusted textual input 的 last-token head activation
z_visual_i[l,h] = raw visual input 的 last-token head activation
delta_i[l,h]    = z_text_i[l,h] - z_visual_i[l,h]
v_type[l,h]     = mean_i(delta_i[l,h])
```

每个 vector 的常见形状是：

```text
[L, H, D_head] = [32, 32, 128]
```

这点很重要：它不是一开始就 flatten 成一个大向量；flatten cosine 只是后处理诊断。

## Category data

Category 主要处理 object existence：

```text
Is there a dog in the image?
```

构造来自 COCO object category labels。

样本有两类：

### Present object

```text
image really contains dog
question: Is there a dog in the image?
trusted text: There is a dog in the image.
```

### Absent object

```text
image does not contain bus
question: Is there a bus in the image?
trusted text: There is no bus in the image.
```

这个方向后来被证明最稳定。POPE 的 object existence 正好与 category expert 匹配，因此 category 结果最容易解释。

## Attribute data

Attribute 初期只保留比较可靠的属性：

- count
- color
- 后续可选 shape / state / action

### Count

COCO annotation 里有每个 object instance，因此可以统计数量。

例子：

```text
question: How many dogs are there in the image?
trusted text: There are 2 dogs in the image.
```

Count 的优势是构造相对干净，后来在 MME count 上出现过明显正收益。

### Color

Color 使用 bbox / segmentation 区域里的 dominant color 粗分类。

颜色词表大致是：

```text
red, blue, green, yellow, black, white, gray,
brown, orange, purple, pink
```

如果颜色不稳定就跳过，不强行构造。

Color 的困难在于：颜色事实必须绑定 object category，例如 `black chair`、`gray bowl`。因此 attr vector 天然携带 category component，导致 cat-attr cosine 很高。

## Relation data

Relation 是整个项目里最不稳定也最有信息量的一部分。

### 早期 COCO bbox relation

最早尝试从 COCO bbox 中计算空间关系：

```text
left of
right of
above
below
on
under
next to
```

例子：

```text
question: Is the dog to the left of the person in the image?
trusted text: The dog is located on the left side of the person in the image.
```

但这有明显问题：

- bbox 几何关系不一定等于语义关系。
- 多实例对象容易歧义。
- overlap / too small / too large 等情况会引入噪声。
- left/right 与 above/below 混在一起，方向可能不一致。

### Relation v2

后来做了 relation v2，强调成对构造和互逆关系。

如果真实关系是：

```text
A left_of B
```

构造 yes：

```text
Question: Is the A to the left of the B in the image?
Label: yes
Trusted text: The A is located on the left side of the B in the image.
This means the B is on the right side of the A.
```

构造 no：

```text
Question: Is the A to the right of the B in the image?
Label: no
Trusted text: The A is located on the left side of the B in the image,
not on the right side. This means the B is on the right side of the A.
```

Relation v2 的数据统计记录为：

```text
total pairs: 2798
train / val / test: 1674 / 558 / 566
label yes / no: 1399 / 1399
left_of: 1030
right_of: 954
above: 418
below: 396
```

它比早期 relation 更稳，在 MME position 小样本上曾从负收益转为弱正收益。

### GQA relation

后续又改用 GQA scene graph，因为 GQA 包含 object / attribute / relation 结构化信息：

```text
image
-> objects
-> object attributes
-> object-object relations
-> questions
-> answers
-> functional program
```

GQA 可以覆盖：

- object / category
- attribute
- spatial relation
- comparison
- logical / compositional reasoning

但 GQA 也更难：

- object space 更长尾。
- scene graph 可能漏标或粒度不一致。
- relation negatives 不一定视觉上绝对不可能。
- GQA-POPE 更容易触发语言先验和共现偏置。

## Disjoint image split

一个重要设计是三类图像严格不重合。

例如 disjoint v1 / v2 中：

```text
cat: 1500 images
attr: 1500 images
rel: 2000 images
total: 5000 images
cat_attr overlap: 0
cat_rel overlap: 0
attr_rel overlap: 0
```

这样做是为了避免：

```text
same image contributes to cat / attr / rel vectors
=> shared image-specific activation
=> artificial overlap between experts
```

但后续实验说明：即使 image-disjoint，三类 raw vectors 仍然高度相似。这意味着 shared component 不只是 image leakage，而是更深层的 factualization / text alignment component。

## AFTER-style typed factual text

在 `8.2` 笔记里，数据构造被整理为：

```text
COCO annotations
-> structured facts
-> typed verbalizer prompt
-> gpt-4o-mini typed t+
-> AFTER trusted/untrusted pairs
-> activation extraction
```

三类 typed `t+` 的区别：

| typed t+ | verbalized facts | deliberately excluded |
|---|---|---|
| `t_cat+` | object categories | count / color / shape / relation |
| `t_attr+` | count / color / shape | spatial relation |
| `t_rel+` | object-object spatial relation | count / color / shape |

这比把所有 facts 混成一个综合 caption 更符合 type-aware expert 的目标。

## Relation bucket split

由于 `rel_all` 一直不稳，后续把 relation 拆成 bucket：

```text
rel_horizontal
rel_vertical
rel_depth
rel_contact
rel_interaction
rel_semantic
rel_position_2d = horizontal + vertical
rel_position = horizontal + vertical + depth
rel_contact_interaction = contact + interaction
```

记录的 bucket count：

| vector | count |
|---|---:|
| rel | 4379 |
| rel_horizontal | 2326 |
| rel_vertical | 627 |
| rel_depth | 259 |
| rel_contact | 886 |
| rel_interaction | 195 |
| rel_semantic | 86 |
| rel_position_2d | 2953 |
| rel_position | 3212 |
| rel_contact_interaction | 1081 |

这个拆分后来带来一个关键发现：

> AMBER relation 本质上更接近 contact / interaction 判断，而不是 MME position 那种 left/right/above/below 判断。把它们平均成 rel_all 会互相污染。

## Benchmark 角色

这些数据集在项目中承担不同角色：

### COCO

主要作为事实构造源：

- object category
- bbox
- mask / segmentation
- color / shape heuristic
- captions

### POPE

主要验证 object existence hallucination：

```text
image -> real objects
yes question: Is there a real_object?
no question: Is there a nonexistent_object?
negative sampling: random / popular / adversarial
```

### MME

用于 yes/no perception subset：

- existence
- count
- position
- color

MME 有 Accuracy 和 Accuracy+。Accuracy+ 更严格，因为一张图的 yes/no 配对题都答对才算对。

### AMBER

用于更广泛的 hallucination：

- generation: `Describe this image.`
- existence
- attribute
- relation

注意：AMBER relation 基本是 direct contact，而不是广义空间关系。

### GQA

用于受控 type-aware diagnostic：

- category
- attribute
- relation
- comparison
- logical / compositional

GQA 不是专门 hallucination benchmark，但适合验证专家向量是否真的具有 type specificity。

### SEED

作为参考 benchmark / semantic clustering 来源。它覆盖图像和视频理解，包括 scene、instance identity、attribute、location、counting、spatial relation、interaction、OCR 等维度。

## 数据构造阶段的当前判断

1. Category 数据最干净，和 POPE 评测最匹配。
2. Attribute 数据可行，但 count 和 color 应该分开看。
3. Relation 必须 subtype 化，不能用一个 `rel_all` 覆盖 contact、interaction、position。
4. Image-disjoint 是必要控制，但不能解决 shared factualization component。
5. 原始 benchmark 问题不适合直接拿来训练 expert；构造阶段应优先 single-factor minpair。
