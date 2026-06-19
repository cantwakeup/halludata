# Research Log: Type-Aware Activation Intervention for LVLM Hallucination

整理日期：2026-06-19

这组日志来自 Obsidian `论文2` 目录里的研究笔记。它不是论文正文，也不是只保留成功结果的实验报告；它更像一份科研过程账本，用来记录这个项目从 idea、数据构造、实验失败、诊断转向到当前可写结论的演化过程。

## 为什么单独写这个日志

仓库里已经有很多正式实验文档，例如：

- `docs/experiment_step*.md`：代码流程和实验步骤。
- `docs/gqa_typeaware_diagnostic.md`：GQA type-aware diagnostic 的可复现流程。
- `data/*/SUMMARY.md`：具体实验输出和 summary。

但这些文件大多回答“怎么跑”和“跑出了什么”。Obsidian 笔记里保留了另一类信息：

- 当时为什么这么构造数据。
- 哪个假设后来被证伪。
- 为什么某个看起来涨点的结果不能 claim。
- 为什么从三类专家转向 subtype / shared-private / routing。
- 哪些指标比 accuracy 更能解释 steering 是否真的在消除幻觉。

这些内容对后续写论文、做答辩、写简历、向导师汇报都很重要，所以单独整理成科研日志。

## 阅读顺序

1. `01_problem_and_method_evolution.md`
   - 研究问题、核心动机、相关工作如何影响方法设计。

2. `02_data_construction_log.md`
   - 从 COCO / GQA / AMBER / MME 到 typed factual pairs 的数据构造演化。

3. `03_experiment_timeline.md`
   - 按阶段记录主要实验、正结果、失败和转向。

4. `04_results_and_diagnostics.md`
   - 当前最值得保留的结果，以及不能过度 claim 的地方。

5. `05_lessons_and_next_steps.md`
   - 从失败里提炼出来的下一步路线。

6. `06_resume_summary.md`
   - 面向简历和汇报的浓缩版本。

## 当前一句话结论

这个项目最稳的结论不是“三类专家已经完全成立”，而是：

> LVLM hallucination steering 中确实存在可利用的 factual activation signal；category expert 在 object hallucination 上最稳定，attribute expert 在 count/color 上有局部正收益，relation 需要进一步拆成 contact / interaction / position 等 subtype。简单的 cat / attr / rel 三专家不够干净，raw vectors 被 shared factualization component 主导，后续应转向 subtype-specific vector bank、causal head discovery 和 token-level routing。

## 当前方法定位

项目的定位可以写成：

> Type-aware token-level activation intervention: 构建 category / attribute / relation 或更细 subtype-family 的事实专家干预向量，并在生成过程中由 router 或 gate 动态选择是否干预、干预哪一类、在什么 heads 上干预。

## 重要提醒

这份日志保留了 negative findings。比如：

- relation all-vector 在 AMBER relation / MME position 上曾经明显失败。
- GQA 上 3x3 expert matrix 没有稳定 diagonal advantage。
- shared-private residual 数学上可分，但作为 steering vector 并不稳定。
- norm top64 很可能选到 shared middle-layer heads，而不是真正 typed causal heads。

这些不是项目失败，而是后续方法设计的依据。
