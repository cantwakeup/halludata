"""Run POPE Regular vs CatExpert with the official LLaVA loader.

This script is the official-loader counterpart of `run_pope_cat_expert_eval.py`.
It intentionally keeps the old HF POPE runner untouched while using:

- `llava.model.builder.load_pretrained_model`
- `llava.conversation.conv_templates`
- `llava.mm_utils.tokenizer_image_token`
- the repository's existing `ExpertSteeringController`

The goal is to compare Regular and CatExpert inside one official LLaVA
coordinate system. It does not run DMAS/Octopus/VCD/M3ID/AVISC methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import write_json, write_jsonl
from expert_data.steering import ExpertSteeringController, normalize_bool


PROMPT_SUFFIX = "Please answer this question in one word."
PARSER_MODES = ("first_yes_no", "contains_yes_no_octopus_like")
YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS = {"no", "n", "false", "0"}
DATASET_ALIASES = {
    "mscoco": ("MSCOCO", "coco", "mscoco"),
    "coco": ("MSCOCO", "coco", "mscoco"),
    "ms_coco": ("MSCOCO", "coco", "mscoco"),
    "gqa": ("GQA", "gqa"),
}
SETTINGS = ("random", "popular", "adversarial")


@dataclass(frozen=True)
class OfficialLlavaImports:
    torch: Any
    image_cls: Any
    image_token_index: int
    default_image_token: str
    default_im_start_token: str
    default_im_end_token: str
    separator_style: Any
    conv_templates: Mapping[str, Any]
    get_model_name_from_path: Any
    tokenizer_image_token: Any
    load_pretrained_model: Any
    disable_torch_init: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--llava-repo-path", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--pope-root", required=True)
    parser.add_argument("--coco-image-root", default="")
    parser.add_argument("--gqa-image-root", default="")
    parser.add_argument("--cat-vector-path", default="")
    parser.add_argument("--cat-vector-source", default="unknown")
    parser.add_argument("--datasets", nargs="+", default=["MSCOCO", "GQA"])
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    parser.add_argument("--methods", nargs="+", default=["regular", "cat"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alphas", default="")
    parser.add_argument("--layers", default="5-25")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--head-select", default="norm", choices=["norm", "random", "all", "expert_map"])
    parser.add_argument("--prefill", default="true")
    parser.add_argument("--decode", default="true")
    parser.add_argument("--apply-to", default="last_token")
    parser.add_argument("--prefill-apply-to", default="last_token")
    parser.add_argument("--decode-apply-to", default="last_token")
    parser.add_argument("--limit", type=int, default=0, help="0 means full file.")
    parser.add_argument("--output-dir", default="data/pope_cat_expert_eval/official_llava_cat_expert_alpha_sweep_full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--prompt-suffix", default=PROMPT_SUFFIX)
    parser.add_argument("--parser-mode", default="first_yes_no", choices=PARSER_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true", help="Skip a raw output file if it already exists.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compat-new-transformers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop cache_position / skip generation kwarg validation for old LLaVA under newer transformers.",
    )
    return parser.parse_args()


def parse_alpha_values(args: argparse.Namespace) -> list[float]:
    if str(args.alphas).strip():
        return [float(item) for item in str(args.alphas).replace(",", " ").split() if item.strip()]
    return [float(args.alpha)]


def canonical_dataset(name: str) -> tuple[str, tuple[str, ...]]:
    key = str(name).strip().lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset '{name}'. Expected MSCOCO or GQA.")
    display = DATASET_ALIASES[key][0]
    return display, tuple(DATASET_ALIASES[key][1:])


def canonical_setting(name: str) -> str:
    setting = str(name).strip().lower()
    if setting not in SETTINGS:
        raise ValueError(f"Unsupported POPE setting '{name}'. Expected one of {SETTINGS}.")
    return setting


def normalize_method(name: str) -> str:
    text = str(name).strip().lower()
    if text in {"regular", "baseline", "base"}:
        return "regular"
    if text in {"cat", "catexpert", "cat_expert", "ours", "ours-cat"}:
        return "cat"
    raise ValueError(f"Unsupported method '{name}'. Expected regular or cat.")


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list or JSONL file: {path}")
        return [dict(item) for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows


def find_pope_file(pope_root: Path, dataset_tokens: tuple[str, ...], setting: str) -> Path | None:
    exact: list[Path] = []
    for token in dataset_tokens:
        exact.extend(
            [
                pope_root / "output" / token / f"{token}_pope_{setting}.json",
                pope_root / "output" / token / f"{token}_pope_{setting}.jsonl",
                pope_root / "output" / "seem" / token / f"{token}_pope_seem_{setting}.json",
                pope_root / "output" / "seem" / token / f"{token}_pope_seem_{setting}.jsonl",
                pope_root / "questions" / f"{token}_pope_{setting}.json",
                pope_root / "questions" / f"{token}_pope_{setting}.jsonl",
                pope_root / f"{token}_pope_{setting}.json",
                pope_root / f"{token}_pope_{setting}.jsonl",
                pope_root / token / f"{setting}.json",
                pope_root / token / f"{setting}.jsonl",
                pope_root / token.upper() / f"{setting}.json",
                pope_root / token.upper() / f"{setting}.jsonl",
            ]
        )
    for path in exact:
        if path.exists():
            return path
    candidates = []
    if pope_root.exists():
        for path in pope_root.rglob("*"):
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            full = str(path).lower()
            if setting in path.name.lower() and any(token in full for token in dataset_tokens) and "pope" in full:
                candidates.append(path)
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0] if candidates else None


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in YES_WORDS or text.startswith("yes"):
        return "yes"
    if text in NO_WORDS or text.startswith("no"):
        return "no"
    raise ValueError(f"Could not normalize POPE label: {value!r}")


def first_present(row: Mapping[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def ensure_question_prompt(question: str, suffix: str = PROMPT_SUFFIX) -> str:
    question = str(question).strip()
    suffix = str(suffix).strip()
    if not suffix:
        return question
    if question.lower().endswith(suffix.lower()):
        return question
    return f"{question} {suffix}"


def image_roots_for_dataset(dataset: str, args: argparse.Namespace) -> list[Path]:
    roots = []
    if dataset == "MSCOCO" and str(args.coco_image_root).strip():
        roots.append(Path(args.coco_image_root))
    if dataset == "GQA" and str(args.gqa_image_root).strip():
        roots.append(Path(args.gqa_image_root))
    return roots


def resolve_image_path(dataset: str, row: Mapping[str, Any], image_roots: list[Path]) -> str:
    candidates: list[str] = []
    for key in ("image_path", "image", "filename", "file_name", "img"):
        value = row.get(key)
        if value not in (None, ""):
            candidates.append(str(value))
    image_id = first_present(row, ("image_id", "id", "coco_id"), "")
    if image_id not in (None, ""):
        image_id_text = str(image_id)
        candidates.append(image_id_text)
        if dataset == "MSCOCO":
            try:
                candidates.append(f"COCO_val2014_{int(image_id_text):012d}.jpg")
            except Exception:
                pass
        if dataset == "GQA":
            candidates.extend([f"{image_id_text}.jpg", f"{image_id_text}.png"])
    for candidate in candidates:
        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)
        for root in image_roots:
            joined = root / candidate
            if joined.exists():
                return str(joined.resolve())
            joined_name = root / path.name
            if joined_name.exists():
                return str(joined_name.resolve())
    return ""


def load_pope_samples(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {"pope_root": str(args.pope_root), "groups": {}}
    limit = int(args.limit)
    for dataset_arg in args.datasets:
        dataset, tokens = canonical_dataset(dataset_arg)
        roots = image_roots_for_dataset(dataset, args)
        for setting_arg in args.settings:
            setting = canonical_setting(setting_arg)
            source_file = find_pope_file(Path(args.pope_root), tokens, setting)
            if source_file is None:
                raise FileNotFoundError(f"Could not find POPE file for dataset={dataset}, setting={setting}, root={args.pope_root}")
            raw_rows = read_json_or_jsonl(source_file)
            selected_rows = raw_rows if limit <= 0 else raw_rows[:limit]
            samples: list[dict[str, Any]] = []
            missing_images: list[dict[str, Any]] = []
            for index, row in enumerate(selected_rows):
                question = str(first_present(row, ("question", "text", "query", "prompt"))).strip()
                if not question:
                    raise ValueError(f"POPE row {index} in {source_file} has no question/text/query field")
                label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target")))
                image_path = resolve_image_path(dataset, row, roots)
                image_id = str(
                    first_present(
                        row,
                        ("image_id", "id", "coco_id"),
                        Path(str(first_present(row, ("image", "image_path", "filename", "file_name"), index))).stem,
                    )
                )
                sample = {
                    "dataset": dataset,
                    "setting": setting,
                    "sample_index": index,
                    "image_id": image_id,
                    "image_path": image_path,
                    "question": question,
                    "prompt": ensure_question_prompt(question, str(args.prompt_suffix)),
                    "label": label,
                    "source_file": str(source_file),
                    "raw": dict(row),
                }
                if not image_path:
                    missing_images.append(sample)
                samples.append(sample)
            if missing_images:
                preview = [{"index": item["sample_index"], "image_id": item["image_id"], "raw": item["raw"]} for item in missing_images[:5]]
                raise FileNotFoundError(
                    f"Missing images for dataset={dataset}, setting={setting}: {len(missing_images)}/{len(samples)}. First missing: {preview}"
                )
            counts = Counter(sample["label"] for sample in samples)
            groups[(dataset, setting)] = samples
            manifest["groups"][f"{dataset}_{setting}"] = {
                "source_file": str(source_file),
                "image_roots": [str(path) for path in roots],
                "num_samples": len(samples),
                "label_counts": dict(counts),
            }
    return groups, manifest


def parse_first_yes_no(text: str) -> str:
    matches = [(match.start(), match.group(1)) for match in re.finditer(r"\b(yes|no)\b", str(text).lower())]
    if not matches:
        return "invalid"
    return sorted(matches)[0][1]


def parse_prediction(text: str, label: str, parser_mode: str) -> str:
    if parser_mode == "first_yes_no":
        return parse_first_yes_no(text)
    if parser_mode == "contains_yes_no_octopus_like":
        generated = str(text).strip().lower()
        label = str(label).strip().lower()
        # This mirrors Octopus eval_pope.py's label-conditioned substring
        # scoring: for yes-labeled rows, absence of "yes" counts as no; for
        # no-labeled rows, absence of "no" counts as yes.
        if label == "yes":
            return "yes" if "yes" in generated else "no"
        if label == "no":
            return "no" if "no" in generated else "yes"
        return "invalid"
    raise ValueError(f"Unsupported parser mode: {parser_mode}")


def import_official_llava(llava_repo_path: str) -> OfficialLlavaImports:
    repo_path = Path(llava_repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        import torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except Exception as exc:
        raise ImportError(f"Could not import official LLaVA from {repo_path}: {exc!r}") from exc
    return OfficialLlavaImports(
        torch=torch,
        image_cls=Image,
        image_token_index=IMAGE_TOKEN_INDEX,
        default_image_token=DEFAULT_IMAGE_TOKEN,
        default_im_start_token=DEFAULT_IM_START_TOKEN,
        default_im_end_token=DEFAULT_IM_END_TOKEN,
        separator_style=SeparatorStyle,
        conv_templates=conv_templates,
        get_model_name_from_path=get_model_name_from_path,
        tokenizer_image_token=tokenizer_image_token,
        load_pretrained_model=load_pretrained_model,
        disable_torch_init=disable_torch_init,
    )


def maybe_apply_new_transformers_compat(model: Any) -> None:
    if getattr(model, "_official_pope_compat_applied", False):
        return
    orig_forward = model.forward

    def forward_without_cache_position(*forward_args: Any, **forward_kwargs: Any) -> Any:
        forward_kwargs.pop("cache_position", None)
        return orig_forward(*forward_args, **forward_kwargs)

    model.forward = forward_without_cache_position  # type: ignore[method-assign]
    model._validate_model_kwargs = lambda model_kwargs: None
    try:
        type(model)._validate_model_kwargs = lambda self, model_kwargs: None
    except Exception:
        pass
    setattr(model, "_official_pope_compat_applied", True)
    print("Applied compatibility shim: drop cache_position and skip model_kwargs validation.")


def load_official_model(args: argparse.Namespace, llava: OfficialLlavaImports) -> tuple[Any, Any, Any, int, str]:
    llava.disable_torch_init()
    model_name = llava.get_model_name_from_path(str(args.model_path))
    print(f"Official LLaVA model path: {args.model_path}")
    print(f"Official LLaVA model name: {model_name}")
    print(f"Official LLaVA repo path: {args.llava_repo_path}")
    print(f"Conversation mode: {args.conv_mode}")
    try:
        tokenizer, model, image_processor, context_len = llava.load_pretrained_model(
            str(args.model_path),
            args.model_base,
            model_name,
            device=str(args.device),
        )
    except TypeError:
        tokenizer, model, image_processor, context_len = llava.load_pretrained_model(
            str(args.model_path),
            args.model_base,
            model_name,
        )
    if bool(args.compat_new_transformers):
        maybe_apply_new_transformers_compat(model)
    model.eval()
    return tokenizer, model, image_processor, int(context_len), model_name


def build_official_prompt(prompt_text: str, model: Any, conv_mode: str, llava: OfficialLlavaImports) -> tuple[str, dict[str, Any]]:
    image_token = llava.default_image_token
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = llava.default_im_start_token + image_token + llava.default_im_end_token
    question_with_image = image_token + "\n" + prompt_text
    conv = llava.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question_with_image)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != llava.separator_style.TWO else conv.sep2
    return prompt, {
        "question_with_image": question_with_image,
        "system": getattr(conv, "system", ""),
        "roles": list(conv.roles),
        "sep": getattr(conv, "sep", None),
        "sep2": getattr(conv, "sep2", None),
        "sep_style": str(getattr(conv, "sep_style", "")),
        "stop_str": stop_str,
        "mm_use_im_start_end": bool(getattr(model.config, "mm_use_im_start_end", False)),
    }


def decode_completion(tokenizer: Any, output_ids: Any, prompt_len: int) -> tuple[str, str, int]:
    generated_ids = output_ids[0][prompt_len:]
    if generated_ids.numel() == 0:
        generated_ids = output_ids[0]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    raw_full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return raw_output, raw_full_output, int(output_ids.shape[-1])


class OfficialLlavaPopeGenerator:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        llava: OfficialLlavaImports,
        tokenizer: Any,
        model: Any,
        image_processor: Any,
        controller: ExpertSteeringController | None = None,
    ) -> None:
        self.args = args
        self.llava = llava
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.controller = controller
        self.do_sample = normalize_bool(args.do_sample)

    def _prepare_controller(self, question: str, mode: str, sign: float) -> None:
        if self.controller is None:
            return
        self.controller.disable()
        self.controller.set_context(question)
        self.controller.set_sign(sign)
        if mode == "steered":
            self.controller.enable()

    def generate(self, sample: Mapping[str, Any], *, mode: str, sign: float) -> tuple[str, dict[str, Any]]:
        torch = self.llava.torch
        image = self.llava.image_cls.open(sample["image_path"]).convert("RGB")
        prompt_text = str(sample["prompt"])
        full_prompt, prompt_info = build_official_prompt(prompt_text, self.model, str(self.args.conv_mode), self.llava)
        input_ids = self.llava.tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            self.llava.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).to(self.model.device, dtype=torch.float16)
        self._prepare_controller(prompt_text, mode, sign)
        generate_kwargs = {
            "images": image_tensor,
            "do_sample": self.do_sample,
            "temperature": float(self.args.temperature),
            "top_p": float(self.args.top_p),
            "num_beams": int(self.args.num_beams),
            "max_new_tokens": int(self.args.max_new_tokens),
            "use_cache": True,
        }
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(input_ids, **generate_kwargs)
            except Exception as exc:
                if not bool(self.args.compat_new_transformers):
                    raise
                message = repr(exc)
                if not any(token in message for token in ("cache_position", "attention_mask", "model_kwargs", "new_ones")):
                    raise
                maybe_apply_new_transformers_compat(self.model)
                generate_kwargs["attention_mask"] = torch.ones_like(input_ids)
                output_ids = self.model.generate(input_ids, **generate_kwargs)
        if self.controller is not None:
            self.controller.disable()
        prompt_len = int(input_ids.shape[1])
        raw_output, raw_full_output, output_token_len = decode_completion(self.tokenizer, output_ids, prompt_len)
        prompt_info.update(
            {
                "full_prompt": full_prompt,
                "prompt_token_len": prompt_len,
                "output_token_len": output_token_len,
                "raw_full_output": raw_full_output,
                "parser_mode": str(self.args.parser_mode),
            }
        )
        return raw_output, prompt_info


def prediction_row(
    *,
    sample: Mapping[str, Any],
    method: str,
    alpha: float | None,
    raw_output: str,
    prompt_info: Mapping[str, Any],
) -> dict[str, Any]:
    label = str(sample["label"])
    pred = parse_prediction(raw_output, label, str(prompt_info.get("parser_mode", "first_yes_no")))
    return {
        "dataset": sample["dataset"],
        "setting": sample["setting"],
        "method": method,
        "alpha": alpha,
        "image_id": sample["image_id"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "prompt": sample["prompt"],
        "full_prompt": prompt_info.get("full_prompt", ""),
        "label": label,
        "raw_output": raw_output,
        "raw_full_output": prompt_info.get("raw_full_output", ""),
        "pred": pred,
        "is_correct": bool(pred == label),
        "parser_mode": prompt_info.get("parser_mode", ""),
        "prompt_token_len": prompt_info.get("prompt_token_len", ""),
        "output_token_len": prompt_info.get("output_token_len", ""),
    }


def run_group(
    *,
    generator: OfficialLlavaPopeGenerator,
    samples: list[dict[str, Any]],
    method: str,
    alpha: float | None,
    output_path: Path,
    progress_every: int,
) -> None:
    rows: list[dict[str, Any]] = []
    mode = "steered" if method == "CatExpert" else "baseline"
    sign = 1.0 if method == "CatExpert" else 0.0
    for index, sample in enumerate(samples, start=1):
        raw_output, prompt_info = generator.generate(sample, mode=mode, sign=sign)
        row = prediction_row(sample=sample, method=method, alpha=alpha, raw_output=raw_output, prompt_info=prompt_info)
        if index > 3:
            row["full_prompt"] = ""
        rows.append(row)
        if int(progress_every) > 0 and index % int(progress_every) == 0:
            print(f"[{sample['dataset']} {sample['setting']} {method} alpha={alpha}] processed {index}/{len(samples)}")
    write_jsonl(output_path, rows)
    print(f"Wrote raw predictions to {output_path}")


def inspect_cat_vector(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cat vector path: {path}")
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        import torch
        payload = torch.load(path, map_location="cpu")
    vectors = payload.get("vectors", {})
    if "cat" not in vectors:
        raise ValueError(f"Cat vector file does not contain vectors['cat']: {path}")
    return {
        "path": str(path),
        "cat_shape": list(vectors["cat"].shape),
        "layers": list(payload.get("layers", [])),
        "num_heads": int(payload.get("num_heads", 0)),
        "head_dim": int(payload.get("head_dim", 0)),
        "hidden_size": int(payload.get("hidden_size", 0)),
    }


def output_name(dataset: str, setting: str, method: str, alpha: float | None) -> str:
    dataset_key = "mscoco" if dataset == "MSCOCO" else dataset.lower()
    if method == "Regular":
        return f"{dataset_key}_{setting}_regular.jsonl"
    alpha_text = str(alpha).replace("-", "neg").replace(".", "p")
    return f"{dataset_key}_{setting}_cat_alpha{alpha_text}.jsonl"


def write_summary_csv(raw_dir: Path, output_path: Path) -> None:
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        rows.extend(read_json_or_jsonl(path))
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        alpha = "" if row.get("alpha") in (None, "") else str(row.get("alpha"))
        grouped.setdefault((str(row.get("dataset")), str(row.get("setting")), str(row.get("method")), alpha), []).append(row)
    fields = ["Dataset", "Setting", "Method", "Alpha", "N", "Accuracy", "Precision", "Recall", "F1", "Yes Rate", "TP", "TN", "FP", "FN", "Invalid"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group_rows in sorted(grouped.items()):
            dataset, setting, method, alpha = key
            metrics = compute_metrics(group_rows)
            writer.writerow({"Dataset": dataset, "Setting": setting, "Method": method, "Alpha": alpha, **metrics})


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = tn = fp = fn = invalid = pred_yes = 0
    for row in rows:
        label = str(row.get("label", "")).lower()
        pred = str(row.get("pred", "invalid")).lower()
        if pred == "yes":
            pred_yes += 1
        if pred not in {"yes", "no"}:
            invalid += 1
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        elif label == "yes":
            fn += 1
        elif label == "no":
            fp += 1
    total = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "N": total,
        "Accuracy": ((tp + tn) / total * 100.0) if total else 0.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
        "F1": f1 * 100.0,
        "Yes Rate": (pred_yes / total * 100.0) if total else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Invalid": invalid,
    }


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite.")
        raw_dir.mkdir(parents=True, exist_ok=True)
        methods = [normalize_method(method) for method in args.methods]
        methods = [method for method in ("regular", "cat") if method in set(methods)]
        alpha_values = parse_alpha_values(args)
        samples_by_group, pope_manifest = load_pope_samples(args)
        llava = import_official_llava(str(args.llava_repo_path))
        llava.torch.manual_seed(int(args.seed))
        if llava.torch.cuda.is_available():
            llava.torch.cuda.manual_seed_all(int(args.seed))
        tokenizer, model, image_processor, context_len, model_name = load_official_model(args, llava)

        print(f"Prompt template: '{{question}} {args.prompt_suffix}'")
        print(f"Prompt suffix: {args.prompt_suffix!r}")
        print(f"Parser mode: {args.parser_mode}")
        print(
            "Decoding: "
            f"do_sample={normalize_bool(args.do_sample)}, temperature={args.temperature}, top_p={args.top_p}, "
            f"num_beams={args.num_beams}, max_new_tokens={args.max_new_tokens}"
        )
        for key, info in pope_manifest["groups"].items():
            print(f"POPE group {key}: {json.dumps(info, ensure_ascii=False)}")

        vector_info: dict[str, Any] | None = None
        if "cat" in methods:
            if not str(args.cat_vector_path).strip():
                raise ValueError("--cat-vector-path is required when methods include cat")
            vector_info = inspect_cat_vector(Path(args.cat_vector_path))
            print(f"Cat vector: {json.dumps(vector_info, ensure_ascii=False)}")
            print(
                f"Steering config: alpha(s)={alpha_values}, layers={args.layers}, topk={args.topk}, "
                f"head_select={args.head_select}, prefill={args.prefill}, decode={args.decode}, apply_to={args.apply_to}"
            )

        generator = OfficialLlavaPopeGenerator(
            args=args,
            llava=llava,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            controller=None,
        )
        raw_files: list[str] = []
        if "regular" in methods:
            for (dataset, setting), samples in samples_by_group.items():
                output_path = raw_dir / output_name(dataset, setting, "Regular", None)
                if args.skip_existing and output_path.exists():
                    print(f"Skip existing raw predictions: {output_path}")
                else:
                    run_group(generator=generator, samples=samples, method="Regular", alpha=None, output_path=output_path, progress_every=int(args.progress_every))
                raw_files.append(str(output_path))

        if "cat" in methods:
            controller = ExpertSteeringController(
                model=model,
                vector_path=Path(args.cat_vector_path),
                layers=str(args.layers),
                alpha=float(alpha_values[0]),
                k_heads=int(args.topk),
                head_select=str(args.head_select),
                router="no_filter",
                enabled_experts=("cat",),
                apply_to=str(args.apply_to),
                steer_prefill=normalize_bool(args.prefill),
                steer_decode=normalize_bool(args.decode),
                prefill_apply_to=str(args.prefill_apply_to),
                decode_apply_to=str(args.decode_apply_to),
            )
            generator.controller = controller
            for alpha in alpha_values:
                controller.alpha = float(alpha)
                for (dataset, setting), samples in samples_by_group.items():
                    output_path = raw_dir / output_name(dataset, setting, "CatExpert", alpha)
                    if args.skip_existing and output_path.exists():
                        print(f"Skip existing raw predictions: {output_path}")
                    else:
                        run_group(generator=generator, samples=samples, method="CatExpert", alpha=float(alpha), output_path=output_path, progress_every=int(args.progress_every))
                    raw_files.append(str(output_path))

        config = {
            "runner": "official_llava",
            "model_path": str(args.model_path),
            "model_base": args.model_base,
            "model_name": model_name,
            "context_len": context_len,
            "llava_repo_path": str(args.llava_repo_path),
            "conv_mode": str(args.conv_mode),
            "pope_root": str(args.pope_root),
            "coco_image_root": str(args.coco_image_root),
            "gqa_image_root": str(args.gqa_image_root),
            "cat_vector": vector_info,
            "cat_vector_source": str(args.cat_vector_source),
            "datasets": [canonical_dataset(item)[0] for item in args.datasets],
            "settings": [canonical_setting(item) for item in args.settings],
            "methods": methods,
            "alphas": alpha_values,
            "prompt_template": f"{{question}} {args.prompt_suffix}",
            "parser_mode": str(args.parser_mode),
            "decode": {
                "do_sample": normalize_bool(args.do_sample),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
            },
            "seed": int(args.seed),
            "steering": {
                "layers": str(args.layers),
                "topk": int(args.topk),
                "head_select": str(args.head_select),
                "prefill": normalize_bool(args.prefill),
                "decode": normalize_bool(args.decode),
                "apply_to": str(args.apply_to),
                "prefill_apply_to": str(args.prefill_apply_to),
                "decode_apply_to": str(args.decode_apply_to),
                "enabled_experts": ["cat"],
            },
            "pope_manifest": pope_manifest,
            "raw_files": raw_files,
        }
        write_json(output_dir / "config.json", config)
        write_summary_csv(raw_dir, output_dir / "summary_local.csv")
    except Exception as exc:
        if os.environ.get("POPE_OFFICIAL_TRACEBACK", "1").strip().lower() not in {"0", "false", "no"}:
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote official POPE cat expert eval config to {output_dir / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
