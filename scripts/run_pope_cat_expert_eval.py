"""Run Regular vs CatExpert POPE evaluation.

This script is a thin POPE-specific wrapper around the repository's existing
LLaVA benchmark generator and ExpertSteeringController. It does not implement a
new steering hook.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from expert_data.activation_cache import write_json, write_jsonl
from expert_data.steering import ExpertSteeringController
from run_steered_benchmark import LlavaBenchmarkGenerator, build_llava_prefix_prompt


PROMPT_SUFFIX = "Please answer this question in one word."
YES_WORDS = {"yes", "y", "true", "1"}
NO_WORDS = {"no", "n", "false", "0"}
DATASET_ALIASES = {
    "mscoco": ("MSCOCO", "coco", "mscoco"),
    "coco": ("MSCOCO", "coco", "mscoco"),
    "ms_coco": ("MSCOCO", "coco", "mscoco"),
    "gqa": ("GQA", "gqa"),
}
SETTINGS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model-id", dest="model_path", required=True)
    parser.add_argument("--pope-root", required=True)
    parser.add_argument("--coco-image-root", default="")
    parser.add_argument("--gqa-image-root", default="")
    parser.add_argument("--cat-vector-path", required=True)
    parser.add_argument("--datasets", nargs="+", default=["MSCOCO", "GQA"])
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    parser.add_argument("--methods", nargs="+", default=["regular", "cat"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alphas", default="", help="Optional whitespace/comma separated alpha sweep.")
    parser.add_argument("--layers", default="5-25")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--head-select", default="norm", choices=["norm", "random", "all", "expert_map"])
    parser.add_argument("--prefill", default="true")
    parser.add_argument("--decode", default="true")
    parser.add_argument("--apply-to", default="last_token")
    parser.add_argument("--prefill-apply-to", default="last_token")
    parser.add_argument("--decode-apply-to", default="last_token")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="data/pope_cat_expert_eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--construct-if-missing",
        action="store_true",
        help="Construct POPE-style fallback samples only when official POPE files are missing.",
    )
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_alpha_values(args: argparse.Namespace) -> list[float]:
    if str(args.alphas).strip():
        text = str(args.alphas).replace(",", " ")
        return [float(item) for item in text.split() if item.strip()]
    return [float(args.alpha)]


def canonical_dataset(name: str) -> tuple[str, tuple[str, ...]]:
    key = str(name).strip().lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset '{name}'. Expected MSCOCO or GQA.")
    display = DATASET_ALIASES[key][0]
    tokens = tuple(DATASET_ALIASES[key][1:])
    return display, tokens


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
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "samples", "questions", "annotations"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list or JSONL file: {path}")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at index {index} in {path}")
            rows.append(item)
        return rows
    except json.JSONDecodeError:
        pass
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}")
            rows.append(item)
    return rows


def find_pope_file(pope_root: Path, dataset_tokens: tuple[str, ...], setting: str) -> Path | None:
    exact: list[Path] = []
    for token in dataset_tokens:
        exact.extend(
            [
                pope_root / "output" / token / f"{token}_pope_{setting}.json",
                pope_root / "output" / token / f"{token}_pope_{setting}.jsonl",
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
            lower = path.name.lower()
            full = str(path).lower()
            if setting in lower and any(token in full for token in dataset_tokens) and "pope" in full:
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


def ensure_question_prompt(question: str) -> str:
    question = str(question).strip()
    suffix = PROMPT_SUFFIX
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
            if path.name != candidate:
                joined_name = root / path.name
                if joined_name.exists():
                    return str(joined_name.resolve())
    return ""


def normalize_pope_rows(
    *,
    dataset: str,
    setting: str,
    source_file: Path,
    image_roots: list[Path],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = read_json_or_jsonl(source_file)
    samples: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        question = str(first_present(row, ("question", "text", "query", "prompt"))).strip()
        if not question:
            raise ValueError(f"POPE row {index} in {source_file} has no question/text/query field")
        label = normalize_label(first_present(row, ("label", "answer", "gt_answer", "ground_truth", "target")))
        image_path = resolve_image_path(dataset, row, image_roots)
        image_id = first_present(row, ("image_id", "id", "coco_id"), "")
        if not image_id:
            image_id = Path(str(first_present(row, ("image", "image_path", "filename", "file_name"), index))).stem
        sample = {
            "sample_id": str(first_present(row, ("sample_id", "question_id", "id"), index)),
            "dataset": dataset,
            "setting": setting,
            "image_id": str(image_id),
            "image_path": image_path,
            "question": question,
            "prompt": ensure_question_prompt(question),
            "label": label,
            "raw": dict(row),
        }
        if not image_path:
            missing_images.append(sample)
        samples.append(sample)
    if int(limit) > 0:
        samples = samples[: int(limit)]
        sample_ids = {sample["sample_id"] for sample in samples}
        missing_images = [sample for sample in missing_images if sample["sample_id"] in sample_ids]
    return samples, missing_images


def load_pope_samples(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    pope_root = Path(args.pope_root)
    sample_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {"pope_root": str(pope_root), "groups": {}}
    for dataset_arg in args.datasets:
        dataset, tokens = canonical_dataset(dataset_arg)
        image_roots = image_roots_for_dataset(dataset, args)
        if not image_roots:
            raise ValueError(f"Missing image root for {dataset}. Pass --coco-image-root or --gqa-image-root.")
        for setting_arg in args.settings:
            setting = canonical_setting(setting_arg)
            source_file = find_pope_file(pope_root, tokens, setting)
            if source_file is None:
                if args.construct_if_missing:
                    raise NotImplementedError(
                        "POPE fallback construction is not implemented in this runner yet. "
                        "Please provide official POPE annotation files."
                    )
                discovered = sorted(str(path) for path in pope_root.rglob("*.json"))[:80] if pope_root.exists() else []
                raise FileNotFoundError(
                    f"Could not find official POPE file for dataset={dataset}, setting={setting} under {pope_root}. "
                    f"Discovered JSON files (first 80): {discovered}"
                )
            samples, missing = normalize_pope_rows(
                dataset=dataset,
                setting=setting,
                source_file=source_file,
                image_roots=image_roots,
                limit=int(args.limit),
            )
            if missing:
                preview = [
                    {"sample_id": row["sample_id"], "image_id": row["image_id"], "raw_image": row["raw"].get("image")}
                    for row in missing[:10]
                ]
                raise FileNotFoundError(
                    f"Missing images for dataset={dataset}, setting={setting}: {len(missing)}/{len(samples)}. "
                    f"First missing: {preview}"
                )
            counts = Counter(sample["label"] for sample in samples)
            sample_groups[(dataset, setting)] = samples
            manifest["groups"][f"{dataset}_{setting}"] = {
                "source_file": str(source_file),
                "image_roots": [str(path) for path in image_roots],
                "num_samples": len(samples),
                "label_counts": dict(counts),
            }
    return sample_groups, manifest


def parse_first_yes_no(text: str) -> str:
    matches = []
    for match in re.finditer(r"\b(yes|no)\b", str(text).lower()):
        matches.append((match.start(), match.group(1)))
    if not matches:
        return "invalid"
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def generate_answer(generator: LlavaBenchmarkGenerator, sample: Mapping[str, Any], *, mode: str, sign: float) -> str:
    image = generator._Image.open(generator.resolve_image_path(sample)).convert("RGB")
    prompt_text = str(sample["prompt"])
    llava_prompt = build_llava_prefix_prompt(prompt_text)
    generator._prepare_controller(prompt_text, mode, sign)
    inputs = generator._inputs_to_device(generator.processor(text=llava_prompt, images=image, return_tensors="pt"))
    with generator._torch.inference_mode():
        output_ids = generator.model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            max_new_tokens=generator.max_new_tokens,
            use_cache=True,
        )
    if generator.controller is not None:
        generator.controller.disable()
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[0][prompt_len:]
    return generator.processor.decode(generated_ids, skip_special_tokens=True).strip()


def prediction_row(
    *,
    sample: Mapping[str, Any],
    method: str,
    alpha: float | None,
    raw_output: str,
) -> dict[str, Any]:
    pred = parse_first_yes_no(raw_output)
    label = str(sample["label"])
    is_correct = pred == label if pred in {"yes", "no"} else False
    return {
        "dataset": sample["dataset"],
        "setting": sample["setting"],
        "method": method,
        "alpha": alpha,
        "image_id": sample["image_id"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "prompt": sample["prompt"],
        "label": label,
        "raw_output": raw_output,
        "pred": pred,
        "is_correct": bool(is_correct),
    }


def run_group(
    *,
    generator: LlavaBenchmarkGenerator,
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
        raw_output = generate_answer(generator, sample, mode=mode, sign=sign)
        rows.append(prediction_row(sample=sample, method=method, alpha=alpha, raw_output=raw_output))
        if int(progress_every) > 0 and index % int(progress_every) == 0:
            print(f"[{sample['dataset']} {sample['setting']} {method}] processed {index}/{len(samples)}")
    write_jsonl(output_path, rows)
    print(f"Wrote raw predictions to {output_path}")


def inspect_cat_vector(path: Path) -> dict[str, Any]:
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
    }


def output_name(dataset: str, setting: str, method: str, alpha: float | None) -> str:
    dataset_key = "mscoco" if dataset == "MSCOCO" else dataset.lower()
    if method == "Regular":
        return f"{dataset_key}_{setting}_regular.jsonl"
    alpha_text = str(alpha).replace("-", "neg").replace(".", "p")
    return f"{dataset_key}_{setting}_cat_alpha{alpha_text}.jsonl"


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

        print(f"LLaVA model path: {args.model_path}")
        print(f"Prompt template: '{{question}} {PROMPT_SUFFIX}'")
        print("Decoding: greedy, temperature=0, do_sample=False, num_beams=1, max_new_tokens=" + str(args.max_new_tokens))
        samples_by_group, pope_manifest = load_pope_samples(args)
        for key, info in pope_manifest["groups"].items():
            print(
                f"POPE group {key}: file={info['source_file']} image_roots={info['image_roots']} "
                f"samples={info['num_samples']} labels={info['label_counts']}"
            )

        vector_info = inspect_cat_vector(Path(args.cat_vector_path))
        print(f"Cat vector: {json.dumps(vector_info, ensure_ascii=False)}")
        print(
            f"Steering config: alpha(s)={alpha_values}, layers={args.layers}, topk={args.topk}, "
            f"head_select={args.head_select}, prefill={args.prefill}, decode={args.decode}, apply_to={args.apply_to}"
        )

        generator = LlavaBenchmarkGenerator(
            model_id=str(args.model_path),
            image_root=Path("/"),
            instances_json=None,
            device=str(args.device),
            compute_dtype=str(args.compute_dtype),
            max_new_tokens=int(args.max_new_tokens),
            controller=None,
        )

        raw_files: list[str] = []
        if "regular" in methods:
            for (dataset, setting), samples in samples_by_group.items():
                output_path = raw_dir / output_name(dataset, setting, "Regular", None)
                run_group(
                    generator=generator,
                    samples=samples,
                    method="Regular",
                    alpha=None,
                    output_path=output_path,
                    progress_every=int(args.progress_every),
                )
                raw_files.append(str(output_path))

        if "cat" in methods:
            controller = ExpertSteeringController(
                model=generator.model,
                vector_path=Path(args.cat_vector_path),
                layers=str(args.layers),
                alpha=float(alpha_values[0]),
                k_heads=int(args.topk),
                head_select=str(args.head_select),
                router="no_filter",
                enabled_experts=("cat",),
                apply_to=str(args.apply_to),
                steer_prefill=truthy(args.prefill),
                steer_decode=truthy(args.decode),
                prefill_apply_to=str(args.prefill_apply_to),
                decode_apply_to=str(args.decode_apply_to),
            )
            generator.controller = controller
            for alpha in alpha_values:
                controller.alpha = float(alpha)
                for (dataset, setting), samples in samples_by_group.items():
                    output_path = raw_dir / output_name(dataset, setting, "CatExpert", alpha)
                    run_group(
                        generator=generator,
                        samples=samples,
                        method="CatExpert",
                        alpha=float(alpha),
                        output_path=output_path,
                        progress_every=int(args.progress_every),
                    )
                    raw_files.append(str(output_path))

        config = {
            "model_path": str(args.model_path),
            "pope_root": str(args.pope_root),
            "coco_image_root": str(args.coco_image_root),
            "gqa_image_root": str(args.gqa_image_root),
            "cat_vector": vector_info,
            "datasets": [canonical_dataset(item)[0] for item in args.datasets],
            "settings": [canonical_setting(item) for item in args.settings],
            "methods": methods,
            "alphas": alpha_values,
            "prompt_template": f"{{question}} {PROMPT_SUFFIX}",
            "decode": {
                "do_sample": False,
                "temperature": 0.0,
                "num_beams": 1,
                "max_new_tokens": int(args.max_new_tokens),
            },
            "steering": {
                "layers": str(args.layers),
                "topk": int(args.topk),
                "head_select": str(args.head_select),
                "prefill": truthy(args.prefill),
                "decode": truthy(args.decode),
                "apply_to": str(args.apply_to),
                "enabled_experts": ["cat"],
            },
            "pope_manifest": pope_manifest,
            "raw_files": raw_files,
        }
        write_json(output_dir / "config.json", config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote POPE cat expert eval config to {output_dir / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
