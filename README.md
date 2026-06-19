# halludata

Research code and experiment records for type-aware activation intervention on LVLM hallucination.

## Research Log

Start here: [Type-Aware Activation Intervention Research Log](docs/research_log/README.md)

这份日志整理了项目从 idea、数据构造、实验失败、诊断转向到当前可写结论的完整过程。它不是只保留成功结果的短摘要，而是保留了实验中的判断、negative findings 和后续路线。

Main entries:

- [Problem and method evolution](docs/research_log/01_problem_and_method_evolution.md)
- [Data construction log](docs/research_log/02_data_construction_log.md)
- [Experiment timeline](docs/research_log/03_experiment_timeline.md)
- [Results and diagnostics](docs/research_log/04_results_and_diagnostics.md)
- [Lessons and next steps](docs/research_log/05_lessons_and_next_steps.md)
- [Resume and report summary](docs/research_log/06_resume_summary.md)

## Existing Experiment Docs

- [GQA type-aware diagnostic](docs/gqa_typeaware_diagnostic.md)
- [External benchmark steering runner](docs/experiment_step6_external_steering_benchmark.md)
- [Typed expert vectors](docs/experiment_step5_typed_expert_vectors.md)
- [Cloud run checklist](docs/cloud_run_checklist.md)

## Current Project Position

The most stable finding is not that three clean experts are already solved. The current evidence is more nuanced:

- Category/object-existence steering is the most reliable.
- Attribute steering shows subtype-dependent gains, especially count/color.
- Relation steering requires subtype-specific treatment such as contact, interaction, and position.
- Raw type vectors are strongly affected by a shared factualization component.
- Future work should move toward subtype-specific vector banks, causal head discovery, and token-level routing.
