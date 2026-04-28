"""Run baseline vs additive expert-steered LLaVA on a lightweight yes/no benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import write_json, write_jsonl
from expert_data.image_resolver import CocoImageResolver
from expert_data.steering import (
    ExpertSteeringController,
    _stable_float,
    build_llava_prefix_prompt,
    normalize_bool,
    parse_csv_items,
)

YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS = {"no", "n", "false", "0"}
YES_CANDIDATES = ("Yes", " yes", "YES", " yes")
NO_CANDIDATES = ("No", " no", "NO", " no")


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", required=True, help="JSONL/JSON benchmark file.")
    parser.add_argument("--benchmark-name", default="yesno", help="Benchmark label stored in metrics/config.")
    parser.add_argument("--out-dir", required=True, help="Output run directory.")
    parser.add_argument("--adapter", choices=["mock", "llava"], default="mock", help="Generation backend.")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="HF model ID or local model path.")
    parser.add_argument("--image-root", default="", help="Image root for LLaVA benchmark images.")
    parser.add_argument("--instances-json", default="", help="COCO instances JSON for image_id resolution.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for LLaVA.")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Generation length cap.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N samples.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty run dir.")
    parser.add_argument("--dry-run", action="store_true", help="Load resources and run one sample without full eval.")

    parser.add_argument("--steer-enable", action="store_true", help="Run an additional steered pass.")
    parser.add_argument("--steer-vector-path", default="", help="expert_vectors.pt path.")
    parser.add_argument("--steer-layers", default="10-20", help="Layers to hook, e.g. 10-20.")
    parser.add_argument("--steer-alpha", type=float, default=1.0, help="Additive steering strength.")
    parser.add_argument("--steer-k-heads", type=int, default=64, help="Global top-K layer-head pairs.")
    parser.add_argument("--steer-head-select", choices=["norm", "random", "all", "expert_map"], default="norm")
    parser.add_argument("--steer-head-map", default="", help="Head-map JSON for --steer-head-select expert_map.")
    parser.add_argument("--steer-expert-key", default="", help="Vector/head-map expert key for expert_map steering.")
    parser.add_argument("--steer-router", choices=["no_filter", "force_cat", "force_attr", "force_rel", "rule"], default="no_filter")
    parser.add_argument("--steer-enabled-experts", default="cat,attr,rel", help="Comma-separated experts.")
    parser.add_argument("--steer-prefill", default="false", help="Whether to edit prompt/prefill tokens.")
    parser.add_argument("--steer-decode", default="true", help="Whether to edit decoding-token forwards.")
    parser.add_argument("--steer-apply-to", choices=["last_token", "all_tokens"], default="last_token")
    parser.add_argument("--prefill-apply-to", choices=["last_token", "all_tokens"], default="last_token")
    parser.add_argument("--decode-apply-to", choices=["last_token"], default="last_token")
    parser.add_argument("--debug-log-hook-delta", default="false", help="Record first-hit edit magnitudes per hooked layer.")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve optional project-relative paths and treat empty values as absent."""

    text = "" if raw_path is None else str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create or validate one benchmark output directory."""

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)


def read_benchmark_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or JSON-list benchmark samples."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} of {input_path}")
                rows.append(payload)
        return rows
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        # POPE uses `.json` filenames for line-delimited JSON question files.
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} of {input_path}")
                rows.append(payload)
        return rows
    if isinstance(payload, dict):
        for key in ("data", "samples", "questions", "annotations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list or JSONL benchmark file: {input_path}")
    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at index {index} of {input_path}")
        rows.append(item)
    return rows


def first_present(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    """Return the first non-empty value under any candidate key."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_label(value: Any) -> str | None:
    """Normalize a yes/no label value."""

    text = str(value).strip().lower()
    if text in YES_WORDS:
        return "yes"
    if text in NO_WORDS:
        return "no"
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def extract_yes_no(text: str) -> str | None:
    """Extract the first yes/no decision from a generated output."""

    cleaned = "".join(char.lower() if char.isalnum() else " " for char in str(text))
    for token in cleaned.split():
        if token in YES_WORDS:
            return "yes"
        if token in NO_WORDS:
            return "no"
    return None


def single_token_ids(tokenizer: Any, candidates: Iterable[str]) -> list[int]:
    """Return unique token IDs for candidates that encode as one token."""

    token_ids: list[int] = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in token_ids:
            token_ids.append(int(ids[0]))
    if not token_ids:
        raise RuntimeError(f"No single-token IDs found for candidates: {list(candidates)}")
    return token_ids


def yes_no_margin(logits: Any, yes_ids: list[int], no_ids: list[int]) -> dict[str, Any]:
    """Compute max Yes/No logits and their first-token margin."""

    yes_logit = float(logits[yes_ids].max().item())
    no_logit = float(logits[no_ids].max().item())
    margin = yes_logit - no_logit
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "margin": margin,
        "prediction": "yes" if margin >= 0.0 else "no",
    }


def normalize_sample(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Normalize common POPE/MME-style field names into one internal schema."""

    question = first_present(row, ("question", "query", "prompt", "text"))
    label = normalize_label(first_present(row, ("answer", "label", "gt_answer", "ground_truth", "target")))
    image_id = first_present(row, ("image_id", "id", "coco_id"), "")
    image_path = first_present(row, ("image_path", "image", "img", "file_name"), "")
    if not question:
        raise ValueError(f"benchmark row {index} is missing a question/prompt field")
    return {
        "sample_id": str(first_present(row, ("sample_id", "question_id", "id"), index)),
        "image_id": str(image_id) if image_id != "" else "",
        "image_path": str(image_path) if image_path != "" else "",
        "question": str(question),
        "label": label,
        "raw": dict(row),
    }


def compute_yesno_metrics(prediction_rows: list[dict[str, Any]], benchmark_name: str) -> dict[str, Any]:
    """Compute yes/no accuracy, F1, yes-rate, and output-length metrics."""

    rows_with_labels = [row for row in prediction_rows if row.get("label") in {"yes", "no"}]
    correct = sum(1 for row in rows_with_labels if row.get("prediction") == row.get("label"))
    yes_predictions = sum(1 for row in prediction_rows if row.get("prediction") == "yes")
    avg_output_length = mean([len(str(row.get("output", "")).split()) for row in prediction_rows] or [0.0])
    tp = sum(1 for row in rows_with_labels if row.get("label") == "yes" and row.get("prediction") == "yes")
    fp = sum(1 for row in rows_with_labels if row.get("label") == "no" and row.get("prediction") == "yes")
    fn = sum(1 for row in rows_with_labels if row.get("label") == "yes" and row.get("prediction") != "yes")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = correct / len(rows_with_labels) if rows_with_labels else None
    metrics = {
        "benchmark_name": benchmark_name,
        "num_samples": len(prediction_rows),
        "num_labeled_samples": len(rows_with_labels),
        "accuracy": accuracy,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "yes_rate": yes_predictions / len(prediction_rows) if prediction_rows else 0.0,
        "average_output_length": avg_output_length,
    }
    if "mme" in benchmark_name.lower():
        metrics["mme_score_proxy"] = None if accuracy is None else accuracy * 100.0
    return metrics


class MockBenchmarkGenerator:
    """Small deterministic generator for unit tests and dry plumbing checks."""

    def generate(self, sample: Mapping[str, Any], mode: str) -> str:
        """Generate a deterministic yes/no output."""

        label = sample.get("label")
        if mode == "steered" and label in {"yes", "no"}:
            return str(label)
        return "yes" if _stable_float(str(sample["sample_id"]) + str(sample["question"])) >= 0.5 else "no"


class LlavaBenchmarkGenerator:
    """Hugging Face LLaVA generator with optional expert steering controller."""

    def __init__(
        self,
        *,
        model_id: str,
        image_root: Path | None,
        instances_json: Path | None,
        device: str,
        compute_dtype: str,
        max_new_tokens: int,
        controller: ExpertSteeringController | None = None,
    ) -> None:
        """Load LLaVA for greedy benchmark generation."""

        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForVision2Seq, AutoProcessor

            try:
                from transformers import LlavaForConditionalGeneration
            except ImportError:
                LlavaForConditionalGeneration = None
        except Exception as exc:
            raise RuntimeError(
                "LLaVA benchmark generation requires working torch, transformers, and Pillow. "
                f"Import failed with {type(exc).__name__}: {exc}"
            ) from exc

        self._torch = torch
        self._Image = Image
        self.model_id = str(model_id)
        self.device = str(device)
        self.compute_dtype = self._resolve_torch_dtype(compute_dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json) if image_root else None
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        model_classes = [candidate for candidate in (LlavaForConditionalGeneration, AutoModelForVision2Seq) if candidate]
        model_kwargs = {"torch_dtype": self.compute_dtype}
        last_error: Exception | None = None
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:  # pragma: no cover - depends on model availability.
                last_error = exc
        else:
            raise RuntimeError(f"Failed to load LLaVA model: {self.model_id}") from last_error
        self.model.to(self.device)
        self.model.eval()
        self.controller = controller
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.yes_token_ids = single_token_ids(tokenizer, YES_CANDIDATES)
        self.no_token_ids = single_token_ids(tokenizer, NO_CANDIDATES)

    def _resolve_torch_dtype(self, dtype_name: str) -> Any:
        """Resolve a torch dtype string."""

        mapping = {
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "float32": self._torch.float32,
        }
        normalized = str(dtype_name).lower()
        if normalized not in mapping:
            raise ValueError(f"Unsupported compute dtype: {dtype_name}")
        return mapping[normalized]

    def _inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Move processor outputs to the configured device."""

        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            if hasattr(value, "to"):
                if key == "pixel_values" and getattr(value, "is_floating_point", lambda: False)():
                    moved[key] = value.to(device=self.device, dtype=self.compute_dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def resolve_image_path(self, sample: Mapping[str, Any]) -> str:
        """Resolve one sample's image path."""

        image_path = str(sample.get("image_path") or "").strip()
        if image_path:
            path = Path(image_path)
            if not path.is_absolute() and self.resolver is not None:
                path = Path(self.resolver.image_root) / path
            if path.exists():
                return str(path)
        if self.resolver is None:
            raise FileNotFoundError("No image path and no --image-root resolver were provided")
        return self.resolver.resolve(sample["image_id"])

    def _prepare_controller(self, question: str, mode: str, sign: float) -> None:
        """Configure and enable/disable the steering controller for one forward."""

        if self.controller is None:
            return
        self.controller.set_context(question)
        self.controller.set_sign(sign)
        if mode == "steered" and float(sign) != 0.0:
            self.controller.enable()
        else:
            self.controller.disable()

    def first_token_margin(self, sample: Mapping[str, Any], *, mode: str, sign: float = 1.0) -> dict[str, Any]:
        """Return first-token Yes/No logits and margin for one sample."""

        image = self._Image.open(self.resolve_image_path(sample)).convert("RGB")
        question = str(sample["question"])
        prompt = build_llava_prefix_prompt(question)
        self._prepare_controller(question, mode, sign)
        inputs = self._inputs_to_device(self.processor(text=prompt, images=image, return_tensors="pt"))
        with self._torch.inference_mode():
            outputs = self.model(**inputs, use_cache=False)
        if self.controller is not None:
            self.controller.disable()
        prompt_len = int(inputs["input_ids"].shape[1])
        logits = outputs.logits[0, prompt_len - 1, :].detach().float().cpu()
        return yes_no_margin(logits, self.yes_token_ids, self.no_token_ids)

    def generate(self, sample: Mapping[str, Any], mode: str, sign: float = 1.0) -> str:
        """Generate one benchmark answer."""

        image = self._Image.open(self.resolve_image_path(sample)).convert("RGB")
        question = str(sample["question"])
        prompt = build_llava_prefix_prompt(question)
        self._prepare_controller(question, mode, sign)
        inputs = self._inputs_to_device(self.processor(text=prompt, images=image, return_tensors="pt"))
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        if self.controller is not None:
            self.controller.disable()
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[0][prompt_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_mode(
    generator: Any,
    samples: list[dict[str, Any]],
    *,
    mode: str,
    benchmark_name: str,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate predictions for one mode and compute metrics."""

    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        output = generator.generate(sample, mode=mode)
        prediction = extract_yes_no(output)
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "label": sample["label"],
                "mode": mode,
                "output": output,
                "prediction": prediction,
            }
        )
        if progress_every > 0 and index % int(progress_every) == 0:
            print(f"[{mode}] processed {index}/{len(samples)} samples")
    return predictions, compute_yesno_metrics(predictions, benchmark_name)


def _safe_accuracy(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    """Return yes/no accuracy for combined steering rows."""

    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    if not labeled:
        return None
    return sum(1 for row in labeled if row.get(pred_key) == row.get("label")) / len(labeled)


def _safe_f1_yes(rows: list[dict[str, Any]], pred_key: str) -> float:
    """Return positive-class F1 for combined steering rows."""

    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    tp = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) == "yes")
    fp = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) == "yes")
    fn = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) != "yes")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _yes_rate(rows: list[dict[str, Any]], pred_key: str) -> float:
    """Return the fraction of rows predicted as yes."""

    return sum(1 for row in rows if row.get(pred_key) == "yes") / len(rows) if rows else 0.0


def yesno_bundle(rows: list[dict[str, Any]], *, pred_key: str, output_key: str, benchmark_name: str) -> dict[str, Any]:
    """Compute yes/no metrics for fixed-positive steering rows."""

    labeled = [row for row in rows if row.get("label") in {"yes", "no"}]
    correct = sum(1 for row in labeled if row.get(pred_key) == row.get("label"))
    tp = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) == "yes")
    fp = sum(1 for row in labeled if row.get("label") == "no" and row.get(pred_key) == "yes")
    fn = sum(1 for row in labeled if row.get("label") == "yes" and row.get(pred_key) != "yes")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "benchmark_name": benchmark_name,
        "num_samples": len(rows),
        "num_labeled_samples": len(labeled),
        "accuracy": correct / len(labeled) if labeled else None,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "yes_rate": _yes_rate(rows, pred_key),
        "average_output_length": mean([len(str(row.get(output_key, "")).split()) for row in rows] or [0.0]),
    }


def summarize_fixed_steering_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark_name: str,
    alpha: float,
) -> dict[str, Any]:
    """Summarize fixed-positive steering rows with first-token diagnostics."""

    baseline_acc = _safe_accuracy(rows, "baseline_pred")
    steered_acc = _safe_accuracy(rows, "steered_pred")
    baseline_f1 = _safe_f1_yes(rows, "baseline_pred")
    steered_f1 = _safe_f1_yes(rows, "steered_pred")
    label_yes = [row["delta_margin"] for row in rows if row.get("label") == "yes"]
    label_no = [row["delta_margin"] for row in rows if row.get("label") == "no"]
    wrong_to_right = [
        row for row in rows
        if row.get("label") in {"yes", "no"}
        and row.get("baseline_pred") != row.get("label")
        and row.get("steered_pred") == row.get("label")
    ]
    right_to_wrong = [
        row for row in rows
        if row.get("label") in {"yes", "no"}
        and row.get("baseline_pred") == row.get("label")
        and row.get("steered_pred") != row.get("label")
    ]
    sign_counts = {
        "num_pos_sign": sum(1 for row in rows if int(row.get("steer_sign", 0)) > 0),
        "num_neg_sign": sum(1 for row in rows if int(row.get("steer_sign", 0)) < 0),
        "num_zero_sign": sum(1 for row in rows if int(row.get("steer_sign", 0)) == 0),
    }
    changed_text = sum(1 for row in rows if bool(row.get("changed_text")))
    changed_pred = sum(1 for row in rows if bool(row.get("changed_pred")))
    return {
        "benchmark_name": benchmark_name,
        "num_samples": len(rows),
        "steering_mode": "fixed_positive",
        "alpha": float(alpha),
        **sign_counts,
        "avg_delta_margin_all": mean([row["delta_margin"] for row in rows] or [0.0]),
        "avg_delta_margin_label_yes": mean(label_yes or [0.0]),
        "avg_delta_margin_label_no": mean(label_no or [0.0]),
        "wrong_to_right": len(wrong_to_right),
        "right_to_wrong": len(right_to_wrong),
        "yes_rate_baseline": _yes_rate(rows, "baseline_pred"),
        "yes_rate_steered": _yes_rate(rows, "steered_pred"),
        "accuracy_baseline": baseline_acc,
        "accuracy_steered": steered_acc,
        "delta_accuracy": (
            steered_acc - baseline_acc
            if baseline_acc is not None and steered_acc is not None else None
        ),
        "f1_baseline": baseline_f1,
        "f1_steered": steered_f1,
        "delta_f1": steered_f1 - baseline_f1,
        "changed_text": changed_text,
        "changed_pred": changed_pred,
        "avg_output_length": mean([len(str(row.get("steered_output", "")).split()) for row in rows] or [0.0]),
    }


def evaluate_fixed_steering_mode(
    generator: Any,
    samples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run baseline and fixed-positive steered generation with first-token margin diagnostics."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        label = sample.get("label")
        baseline_margin = generator.first_token_margin(sample, mode="baseline", sign=0.0)
        baseline_output = generator.generate(sample, mode="baseline", sign=0.0)
        baseline_pred = extract_yes_no(baseline_output)
        steer_sign = 1
        steered_margin = generator.first_token_margin(sample, mode="steered", sign=steer_sign)
        steered_output = generator.generate(sample, mode="steered", sign=steer_sign)
        steered_pred = extract_yes_no(steered_output)
        changed_text = baseline_output != steered_output
        changed_pred = baseline_pred != steered_pred
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "image": str(sample.get("raw", {}).get("image", sample.get("image_path", sample["image_id"]))),
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "label": label,
                "baseline_output": baseline_output,
                "steered_output": steered_output,
                "baseline_pred": baseline_pred,
                "steered_pred": steered_pred,
                "baseline_logit_pred": baseline_margin["prediction"],
                "steered_logit_pred": steered_margin["prediction"],
                "baseline_yes_logit": baseline_margin["yes_logit"],
                "baseline_no_logit": baseline_margin["no_logit"],
                "baseline_margin": baseline_margin["margin"],
                "steered_yes_logit": steered_margin["yes_logit"],
                "steered_no_logit": steered_margin["no_logit"],
                "steered_margin": steered_margin["margin"],
                "delta_margin": steered_margin["margin"] - baseline_margin["margin"],
                "steer_sign": int(steer_sign),
                "steer_alpha": float(args.steer_alpha),
                "steering_mode": "fixed_positive",
                "was_steered": True,
                "changed_text": changed_text,
                "changed_pred": changed_pred,
            }
        )
        if int(args.progress_every) > 0 and index % int(args.progress_every) == 0:
            print(f"[fixed-steering] processed {index}/{len(samples)} samples")
    return rows, summarize_fixed_steering_rows(
        rows,
        benchmark_name=args.benchmark_name,
        alpha=float(args.steer_alpha),
    )


def build_config(args: argparse.Namespace, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a serializable run config."""

    return {
        "benchmark_name": args.benchmark_name,
        "benchmark_data": str(resolve_project_path(args.benchmark_data)),
        "adapter": args.adapter,
        "model_id": args.model_id,
        "num_samples": len(samples),
        "limit": int(args.limit),
        "steering": {
            "enabled": bool(args.steer_enable),
            "vector_path": str(resolve_optional_project_path(args.steer_vector_path) or ""),
            "layers": str(args.steer_layers),
            "alpha": float(args.steer_alpha),
            "k_heads": int(args.steer_k_heads),
            "head_select": str(args.steer_head_select),
            "head_map": str(resolve_optional_project_path(args.steer_head_map) or ""),
            "expert_key": str(args.steer_expert_key),
            "router": str(args.steer_router),
            "enabled_experts": parse_csv_items(args.steer_enabled_experts),
            "apply_to": str(args.steer_apply_to),
            "steer_prefill": normalize_bool(args.steer_prefill),
            "steer_decode": normalize_bool(args.steer_decode),
            "prefill_apply_to": str(args.prefill_apply_to),
            "decode_apply_to": str(args.decode_apply_to),
            "debug_log_hook_delta": normalize_bool(args.debug_log_hook_delta),
            "steering_mode": "fixed_positive",
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run baseline and optional steered benchmark passes."""

    out_dir = resolve_project_path(args.out_dir)
    ensure_output_dir(out_dir, args.overwrite)
    rows = read_benchmark_rows(resolve_project_path(args.benchmark_data))
    samples = [normalize_sample(row, index) for index, row in enumerate(rows)]
    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]
    if args.dry_run:
        samples = samples[:1]
    if not samples:
        raise ValueError("No benchmark samples to evaluate")

    controller = None
    if args.adapter == "mock":
        generator = MockBenchmarkGenerator()
    else:
        image_root = resolve_optional_project_path(args.image_root)
        if image_root is None:
            raise ValueError("--image-root is required when --adapter llava")
        model_probe = None
        if args.steer_enable:
            if not args.steer_vector_path:
                raise ValueError("--steer-vector-path is required when --steer-enable and --adapter llava")
        generator = LlavaBenchmarkGenerator(
            model_id=args.model_id,
            image_root=image_root,
            instances_json=resolve_optional_project_path(args.instances_json),
            device=args.device,
            compute_dtype=args.compute_dtype,
            max_new_tokens=args.max_new_tokens,
            controller=None,
        )
        if args.steer_enable:
            controller = ExpertSteeringController(
                generator.model,
                resolve_project_path(args.steer_vector_path),
                layers=args.steer_layers,
                alpha=float(args.steer_alpha),
                k_heads=int(args.steer_k_heads),
                head_select=str(args.steer_head_select),
                head_map_path=resolve_optional_project_path(args.steer_head_map),
                expert_key=str(args.steer_expert_key).strip() or None,
                router=str(args.steer_router),
                enabled_experts=tuple(parse_csv_items(args.steer_enabled_experts)),
                apply_to=str(args.steer_apply_to),
                steer_prefill=normalize_bool(args.steer_prefill),
                steer_decode=normalize_bool(args.steer_decode),
                prefill_apply_to=str(args.prefill_apply_to),
                decode_apply_to=str(args.decode_apply_to),
                debug_log_hook_delta=normalize_bool(args.debug_log_hook_delta),
            )
            generator.controller = controller
        del model_probe

    if args.steer_enable and args.adapter == "llava":
        fixed_rows, fixed_metrics = evaluate_fixed_steering_mode(generator, samples, args=args)
        baseline_metrics = yesno_bundle(
            fixed_rows,
            pred_key="baseline_pred",
            output_key="baseline_output",
            benchmark_name=args.benchmark_name,
        )
        steered_metrics = yesno_bundle(
            fixed_rows,
            pred_key="steered_pred",
            output_key="steered_output",
            benchmark_name=args.benchmark_name,
        )
        metrics: dict[str, Any] = {
            "baseline": baseline_metrics,
            "steered": steered_metrics,
            "delta_accuracy": fixed_metrics["delta_accuracy"],
            "fixed_steering": fixed_metrics,
            "accuracy_baseline": fixed_metrics["accuracy_baseline"],
            "accuracy_steered": fixed_metrics["accuracy_steered"],
            "precision_yes": steered_metrics.get("precision_yes"),
            "recall_yes": steered_metrics.get("recall_yes"),
            "f1_yes": steered_metrics.get("f1_yes"),
            "yes_rate_baseline": fixed_metrics["yes_rate_baseline"],
            "yes_rate_steered": fixed_metrics["yes_rate_steered"],
            "wrong_to_right": fixed_metrics["wrong_to_right"],
            "right_to_wrong": fixed_metrics["right_to_wrong"],
            "avg_delta_margin_all": fixed_metrics["avg_delta_margin_all"],
            "avg_delta_margin_label_yes": fixed_metrics["avg_delta_margin_label_yes"],
            "avg_delta_margin_label_no": fixed_metrics["avg_delta_margin_label_no"],
            "changed_pred": fixed_metrics["changed_pred"],
            "changed_text": fixed_metrics["changed_text"],
            "avg_output_length": fixed_metrics["avg_output_length"],
        }
        if controller is not None:
            metrics["steering_diagnostics"] = controller.summary()
        config = build_config(args, samples)
        if not args.dry_run:
            write_jsonl(out_dir / "predictions.jsonl", fixed_rows)
            write_json(out_dir / "metrics.json", metrics)
            write_json(out_dir / "config.json", config)
        elif controller is not None:
            print(f"Dry-run steering diagnostics: {controller.summary()}")
        return {"out_dir": out_dir, "metrics": metrics, "config": config}

    all_predictions: list[dict[str, Any]] = []
    baseline_predictions, baseline_metrics = evaluate_mode(
        generator,
        samples,
        mode="baseline",
        benchmark_name=args.benchmark_name,
        progress_every=int(args.progress_every),
    )
    all_predictions.extend(baseline_predictions)
    metrics: dict[str, Any] = {"baseline": baseline_metrics}
    if args.steer_enable:
        steered_predictions, steered_metrics = evaluate_mode(
            generator,
            samples,
            mode="steered",
            benchmark_name=args.benchmark_name,
            progress_every=int(args.progress_every),
        )
        all_predictions.extend(steered_predictions)
        metrics["steered"] = steered_metrics
        if baseline_metrics["accuracy"] is not None and steered_metrics["accuracy"] is not None:
            metrics["delta_accuracy"] = steered_metrics["accuracy"] - baseline_metrics["accuracy"]
        if controller is not None:
            metrics["steering_diagnostics"] = controller.summary()

    config = build_config(args, samples)
    if not args.dry_run:
        write_jsonl(out_dir / "predictions.jsonl", all_predictions)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "config.json", config)
    elif controller is not None:
        print(f"Dry-run steering diagnostics: {controller.summary()}")
    return {"out_dir": out_dir, "metrics": metrics, "config": config}


def main() -> int:
    """Run the benchmark CLI."""

    args = parse_args()
    try:
        result = run_benchmark(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("Dry run completed without writing full benchmark files.")
    else:
        print(f"Wrote benchmark run to {result['out_dir']}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
