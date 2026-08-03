# Versioned Contents

Files versioned from this pass:

| path | purpose |
| --- | --- |
| `experiments/typed_fas_next/run_vector_only_diagnostics.py` | Cached tensor diagnostic runner for contrast/PCA/head-mask experiments. |
| `experiments/typed_fas_next/vector_only/VECTOR_ONLY_DIAGNOSTICS.md` | Human-readable representation diagnostics. |
| `experiments/typed_fas_next/vector_only/*_summary.csv` and `condition_*.csv` | Compact numerical summaries for review. |
| `experiments/typed_fas_next/00_repo_audit.md` | Required repo audit. |
| `experiments/typed_fas_next/01_existing_results_summary.md` | Existing result reconstruction and selectivity margins. |
| `experiments/typed_fas_next/02_vector_only_experiments.md` | Executed vector-only experiment report. |
| `experiments/typed_fas_next/03_anchor_cancellation.md` | Anchor cancellation status and next-step plan. |
| `experiments/typed_fas_next/04_correct_wrong_minpair.md` | Correct-vs-wrong minimal-pair status and prior evidence. |
| `experiments/typed_fas_next/05_subtype_and_routing.md` | Subtype/routing status and prerequisites. |
| `experiments/typed_fas_next/results_summary.csv` | Compact CSV summary. |
| `experiments/typed_fas_next/results_summary.json` | Compact JSON summary. |
| `experiments/typed_fas_next/commands.sh` | Key commands run plus non-run dev template. |
| `experiments/typed_fas_next/changed_files.md` | This manifest. |

No existing vectors, raw data, or benchmark result directories were modified.

The generated vector bundle, exhaustive head maps, head-map file index, and full diagnostic JSON remain local and are ignored by Git. They are reproducible by running `python experiments/typed_fas_next/run_vector_only_diagnostics.py --overwrite` against the documented cached inputs.
