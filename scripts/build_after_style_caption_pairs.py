"""Build AFTER-style caption-aligned trusted prompts from typed pair banks.

This script does not call a remote model by default. It prepares the same two
caption stages used by AFTER:

1. Image-level factual description t+ from image facts (Ifst prompt).
2. Query-focused factual description t* from t+ and query objects (Iqst prompt).

The output pair files keep the existing visual side unchanged and replace the
trusted side with caption-style factual text. Use --backend cache with generated
captions for the closest AFTER alignment, or --backend template for a local
smoke test before spending model/API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib import error, request


EXPERT_TYPES = ("cat", "attr", "rel")
DEFAULT_SPLITS = ("train", "val", "test")

IFST_PROMPT_TEMPLATE = """Your task is to generate comprehensive and factual description of the given image based on the factual information in a single paragraph. Specifically, the first step is to interpret the content of the given image. The second step is to generate a complete and accurate image description by integrating the factual information of all provided objects. Each object's factual information includes its category, position, color, shape, count, and relationships with other objects. In the generated description, ensure that all provided factual information is included as comprehensively as possible.

Factual Information:
{factual_information}

Output Description:"""

IQST_PROMPT_TEMPLATE = """Your task is to extract and return the related description of the given category. For each category, retain only the parts of the original description that are relevant to the category. Finally, output the object-related textual descriptions in the same order as the input objects, separated by semicolons.

Original description: {paragraph}
Objects: {objects}
Object-related description:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/after_template_disjoint_v2/pairs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated split names to process.")
    parser.add_argument(
        "--backend",
        choices=["template", "cache", "openai", "prompt_only"],
        default="template",
        help=(
            "template: deterministic local caption fallback; "
            "cache: read generated best/query captions from JSONL caches; "
            "openai: generate captions from prompts via the OpenAI API; "
            "prompt_only: only write prompts and metadata, leave trusted prompts unchanged."
        ),
    )
    parser.add_argument("--caption-cache", default="", help="JSONL with image_key plus best_cap/factual_caption/text.")
    parser.add_argument("--query-cache", default="", help="JSONL with pair_id/id plus query_cap/text.")
    parser.add_argument(
        "--trusted-caption-mode",
        choices=["query", "best"],
        default="query",
        help="Which caption becomes trusted_factual_text in output pairs.",
    )
    parser.add_argument("--limit-images", type=int, default=0, help="Optional smoke limit by unique image keys.")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-tokens", type=int, default=300)
    parser.add_argument("--openai-query-max-tokens", type=int, default=160)
    parser.add_argument("--openai-timeout", type=float, default=120.0)
    parser.add_argument("--openai-retries", type=int, default=3)
    parser.add_argument(
        "--openai-generate-query-captions",
        action="store_true",
        help="Also call OpenAI for each query-focused caption. Without this, query captions use cache or rule fallback.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root() / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_names(raw: str) -> list[str]:
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not names:
        raise ValueError("--splits must provide at least one split name")
    return names


def render_trusted_prompt(trusted_factual_text: str, question: str) -> str:
    return (
        f"The given image depicts the following scene: {trusted_factual_text}\n"
        "Please directly answer the following question from the image description, "
        f"without guessing or reasoning. Question: {question}"
    )


def image_key(row: Mapping[str, Any]) -> str:
    value = row.get("image_id")
    if value not in (None, ""):
        return str(value)
    value = row.get("image")
    if value not in (None, ""):
        return str(value)
    return str(row.get("pair_id") or row.get("id") or "")


def pair_id(row: Mapping[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("id") or "")


def normalize_objects(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, Iterable):
        items = value
    else:
        return [str(value)]
    objects: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            objects.append(text)
    return objects


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def fact_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("hallucination_type") or ""),
        str(row.get("subtype") or ""),
        str(row.get("label") or ""),
        str(row.get("factual_fact") or row.get("trusted_factual_text") or ""),
    )


def factual_information_for_image(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    deduped: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = fact_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    facts: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERT_TYPES}
    for row in deduped:
        expert = str(row.get("hallucination_type") or "").strip()
        if expert not in facts:
            continue
        text = str(row.get("factual_fact") or "").strip()
        if not text:
            text = str(row.get("trusted_factual_text") or "").strip()
        facts[expert].append(
            {
                "subtype": str(row.get("subtype") or ""),
                "objects": normalize_objects(row.get("objects")),
                "label": str(row.get("label") or ""),
                "fact": text,
            }
        )
    return {
        "image_key": image_key(rows[0]) if rows else "",
        "image": str(rows[0].get("image") or "") if rows else "",
        "facts": facts,
    }


def factual_information_text(info: Mapping[str, Any]) -> str:
    facts = info.get("facts", {})
    if not isinstance(facts, Mapping):
        facts = {}
    labels = {"cat": "Category Facts", "attr": "Attribute Facts", "rel": "Relation Facts"}
    payload: dict[str, list[dict[str, Any]]] = {}
    for expert in EXPERT_TYPES:
        entries = facts.get(expert, []) if isinstance(facts, Mapping) else []
        cleaned: list[dict[str, Any]] = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                cleaned.append(
                    {
                        "subtype": entry.get("subtype", ""),
                        "objects": entry.get("objects", []),
                        "label": entry.get("label", ""),
                        "fact": entry.get("fact", ""),
                    }
                )
        payload[labels[expert]] = cleaned
    return json.dumps(payload, indent=2, ensure_ascii=False)


def template_best_caption(info: Mapping[str, Any], max_facts: int = 24) -> str:
    facts = info.get("facts", {})
    pieces: list[str] = []
    if isinstance(facts, Mapping):
        for expert in EXPERT_TYPES:
            entries = facts.get(expert, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                fact = str(entry.get("fact") or "").strip()
                if fact:
                    pieces.append(fact.rstrip(".") + ".")
    pieces = unique_preserve_order(pieces)
    if not pieces:
        return "The image contains factual visual information from the annotations."
    selected = pieces[:max_facts]
    if len(pieces) > max_facts:
        selected.append("Additional annotated visual facts are present.")
    return " ".join(selected)


def load_caption_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Caption cache not found: {path}")
    cache: dict[str, str] = {}
    for row in read_jsonl(path):
        key = str(row.get("image_key") or row.get("image_id") or row.get("image") or "").strip()
        text = str(row.get("best_cap") or row.get("factual_caption") or row.get("caption") or row.get("text") or "").strip()
        if key and text:
            cache[key] = text
    return cache


def load_query_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Query cache not found: {path}")
    cache: dict[str, str] = {}
    for row in read_jsonl(path):
        key = str(row.get("pair_id") or row.get("id") or "").strip()
        text = str(row.get("query_cap") or row.get("query_caption") or row.get("text") or "").strip()
        if key and text:
            cache[key] = text
    return cache


def openai_chat_completion(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    """Call OpenAI chat completions using stdlib only.

    Keeping this dependency-free avoids changing the fragile LLaVA environments.
    """

    endpoint = str(base_url).rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate concise factual image descriptions from provided "
                    "annotation facts. Return only the requested description text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=float(timeout)) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            text = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            text = str(text).strip()
            if not text:
                raise RuntimeError(f"OpenAI returned empty content: {body[:500]}")
            return text
        except (error.HTTPError, error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {detail[:1000]}")
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"OpenAI caption generation failed: {last_error}")


def no_object_sentence(objects: list[str]) -> str:
    if not objects:
        return "The queried object is not present in the image."
    if len(objects) == 1:
        return f"There is no {objects[0]} in the image."
    joined = ", ".join(objects[:-1]) + f", or {objects[-1]}"
    return f"There is no {joined} in the image."


def rule_query_caption(row: Mapping[str, Any], best_cap: str) -> str:
    objects = normalize_objects(row.get("objects"))
    label = str(row.get("label") or "").lower()
    fact = str(row.get("factual_fact") or "").strip()
    expert = str(row.get("hallucination_type") or "").strip()
    if label == "no" and expert in {"cat", "attr"}:
        return no_object_sentence(objects)
    if fact:
        return fact.rstrip(".") + "."
    return best_cap


def query_prompt(row: Mapping[str, Any], best_cap: str) -> str:
    objects = normalize_objects(row.get("objects"))
    if not objects:
        objects = [str(row.get("question") or row.get("visual_prompt") or "the query").strip()]
    return IQST_PROMPT_TEMPLATE.format(paragraph=best_cap, objects=json.dumps(objects, ensure_ascii=False))


def output_exists(output_dir: Path, splits: list[str]) -> bool:
    expected = [output_dir / f"{split}.jsonl" for split in splits]
    expected.extend(
        [
            output_dir / "image_caption_prompts.jsonl",
            output_dir / "query_caption_prompts.jsonl",
            output_dir / "stats.json",
            output_dir / "REPORT.md",
        ]
    )
    return any(path.exists() for path in expected)


def markdown_table(headers: list[str], rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(
    *,
    input_dir: Path,
    output_dir: Path,
    stats: Mapping[str, Any],
    examples: list[Mapping[str, Any]],
) -> str:
    split_rows = [
        {"split": split, "rows": count}
        for split, count in sorted(dict(stats.get("rows_by_split", {})).items())
    ]
    type_rows = [
        {"type": expert, "rows": count}
        for expert, count in sorted(dict(stats.get("rows_by_type", {})).items())
    ]
    lines = [
        "# AFTER-Style Caption Pair Build",
        "",
        f"- Input dir: `{input_dir}`",
        f"- Output dir: `{output_dir}`",
        f"- Backend: `{stats.get('backend', '')}`",
        f"- Trusted caption mode: `{stats.get('trusted_caption_mode', '')}`",
        f"- OpenAI model: `{stats.get('openai_model', '')}`",
        f"- Unique images: `{stats.get('unique_images', 0)}`",
        f"- Total rows: `{stats.get('total_rows', 0)}`",
        f"- Caption cache hits: `{stats.get('caption_cache_hits', 0)}`",
        f"- Query cache hits: `{stats.get('query_cache_hits', 0)}`",
        "",
        "## Rows By Split",
        "",
        markdown_table(["split", "rows"], split_rows) if split_rows else "No split rows.",
        "",
        "## Rows By Type",
        "",
        markdown_table(["type", "rows"], type_rows) if type_rows else "No type rows.",
        "",
        "## Outputs",
        "",
        "- `image_caption_prompts.jsonl`: one AFTER Ifst prompt per image.",
        "- `query_caption_prompts.jsonl`: one AFTER Iqst prompt per pair.",
        "- `generated_best_caps.jsonl`: OpenAI-generated image-level captions when `--backend openai` is used.",
        "- `generated_query_caps.jsonl`: OpenAI-generated query captions when `--openai-generate-query-captions` is used.",
        "- `<split>.jsonl`: pair files with updated trusted_factual_text/trusted_prompt.",
        "- `caption_cache_template.jsonl`: template fallback captions for smoke tests.",
        "",
        "## Example Rows",
        "",
        "```json",
        json.dumps(examples[:3], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Notes",
        "",
        "- API keys are read from an environment variable, default `OPENAI_API_KEY`; do not write keys into committed files.",
        "- For closest AFTER alignment, use `--backend openai --openai-generate-query-captions`, or generate caches externally and rerun with `--backend cache --caption-cache ...`.",
        "- If query captions are not generated, the rule fallback uses the pair's current factual fact or no-object sentence.",
        "- This script only rewrites pair JSONL. Re-run activation extraction and vector building after creating the aligned pairs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    splits = split_names(args.splits)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input pair dir not found: {input_dir}")
    if output_exists(output_dir, splits) and not args.overwrite:
        raise FileExistsError(f"Output exists under {output_dir}. Pass --overwrite to replace.")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    rows_by_split: dict[str, int] = {}
    for split in splits:
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"[caption-pairs] skip missing split: {path}", file=sys.stderr)
            continue
        rows = read_jsonl(path)
        split_rows[split] = rows
        all_rows.extend(rows)
        rows_by_split[split] = len(rows)
    if not all_rows:
        raise RuntimeError(f"No pair rows found in {input_dir} for splits {splits}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = image_key(row)
        if not key:
            raise ValueError(f"Row lacks image/image_id/pair_id: {row}")
        grouped[key].append(row)

    if args.limit_images and int(args.limit_images) > 0:
        allowed = set(sorted(grouped)[: int(args.limit_images)])
        grouped = {key: rows for key, rows in grouped.items() if key in allowed}
        split_rows = {
            split: [row for row in rows if image_key(row) in allowed]
            for split, rows in split_rows.items()
        }
        all_rows = [row for rows in split_rows.values() for row in rows]

    caption_cache: dict[str, str] = {}
    query_cache: dict[str, str] = {}
    if args.caption_cache:
        caption_cache = load_caption_cache(resolve_project_path(args.caption_cache))
    if args.query_cache:
        query_cache = load_query_cache(resolve_project_path(args.query_cache))
    if args.backend == "cache" and not caption_cache:
        raise ValueError("--backend cache requires --caption-cache with generated captions")
    openai_api_key = ""
    if args.backend == "openai":
        openai_api_key = os.environ.get(str(args.openai_api_key_env), "").strip()
        if not openai_api_key:
            raise ValueError(
                f"--backend openai requires environment variable {args.openai_api_key_env}. "
                f"Example: export {args.openai_api_key_env}='sk-...'"
            )

    image_infos = {key: factual_information_for_image(rows) for key, rows in grouped.items()}
    template_caption_cache: dict[str, str] = {}
    image_prompt_rows: list[dict[str, Any]] = []
    for key, info in sorted(image_infos.items()):
        factual_info_text = factual_information_text(info)
        prompt = IFST_PROMPT_TEMPLATE.format(factual_information=factual_info_text)
        template_cap = template_best_caption(info)
        template_caption_cache[key] = template_cap
        image_prompt_rows.append(
            {
                "image_key": key,
                "image": info.get("image", ""),
                "factual_information": info.get("facts", {}),
                "ifst_prompt": prompt,
                "template_best_cap": template_cap,
            }
        )

    caption_cache_hits = 0
    query_cache_hits = 0
    generated_best_rows: list[dict[str, Any]] = []
    generated_query_rows: list[dict[str, Any]] = []

    if args.backend == "openai":
        print(
            f"[caption-pairs] generating {len(image_prompt_rows)} image-level captions "
            f"with {args.openai_model}",
            file=sys.stderr,
            flush=True,
        )
        for index, row in enumerate(image_prompt_rows, start=1):
            key = str(row["image_key"])
            if key in caption_cache:
                continue
            text = openai_chat_completion(
                prompt=str(row["ifst_prompt"]),
                api_key=openai_api_key,
                base_url=str(args.openai_base_url),
                model=str(args.openai_model),
                temperature=float(args.openai_temperature),
                max_tokens=int(args.openai_max_tokens),
                timeout=float(args.openai_timeout),
                retries=int(args.openai_retries),
            )
            caption_cache[key] = text
            generated_best_rows.append(
                {
                    "image_key": key,
                    "image": row.get("image", ""),
                    "best_cap": text,
                    "caption_source": "openai",
                    "model": str(args.openai_model),
                }
            )
            if index % 20 == 0 or index == len(image_prompt_rows):
                print(
                    f"[caption-pairs] generated image captions {index}/{len(image_prompt_rows)}",
                    file=sys.stderr,
                    flush=True,
                )

    def best_caption_for(row: Mapping[str, Any]) -> str:
        nonlocal caption_cache_hits
        key = image_key(row)
        cached = caption_cache.get(key)
        if cached:
            caption_cache_hits += 1
            return cached
        if args.backend == "prompt_only":
            return str(row.get("trusted_factual_text") or "")
        return template_caption_cache.get(key, str(row.get("trusted_factual_text") or ""))

    query_prompt_rows: list[dict[str, Any]] = []
    output_split_rows: dict[str, list[dict[str, Any]]] = {}
    examples: list[dict[str, Any]] = []

    for split, rows in split_rows.items():
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            row_id = pair_id(row)
            best_cap = best_caption_for(row)
            cached_query = query_cache.get(row_id)
            if cached_query:
                query_cache_hits += 1
                query_cap = cached_query
            elif args.backend == "prompt_only":
                query_cap = str(row.get("trusted_factual_text") or "")
            else:
                query_cap = rule_query_caption(row, best_cap)

            prompt = query_prompt(row, best_cap)
            if args.backend == "openai" and args.openai_generate_query_captions and row_id not in query_cache:
                cached_query = openai_chat_completion(
                    prompt=prompt,
                    api_key=openai_api_key,
                    base_url=str(args.openai_base_url),
                    model=str(args.openai_model),
                    temperature=float(args.openai_temperature),
                    max_tokens=int(args.openai_query_max_tokens),
                    timeout=float(args.openai_timeout),
                    retries=int(args.openai_retries),
                )
                query_cache[row_id] = cached_query
                query_cap = cached_query
                generated_query_rows.append(
                    {
                        "pair_id": row_id,
                        "image_key": image_key(row),
                        "query_cap": cached_query,
                        "caption_source": "openai",
                        "model": str(args.openai_model),
                    }
                )
                if len(generated_query_rows) % 50 == 0:
                    print(
                        f"[caption-pairs] generated query captions {len(generated_query_rows)}",
                        file=sys.stderr,
                        flush=True,
                    )
            query_prompt_rows.append(
                {
                    "pair_id": row_id,
                    "image_key": image_key(row),
                    "image": row.get("image", ""),
                    "question": row.get("question", ""),
                    "objects": normalize_objects(row.get("objects")),
                    "label": row.get("label", ""),
                    "hallucination_type": row.get("hallucination_type", ""),
                    "subtype": row.get("subtype", ""),
                    "iqst_prompt": prompt,
                    "query_cap": query_cap,
                    "rule_query_cap": query_cap,
                }
            )

            trusted_text = best_cap if args.trusted_caption_mode == "best" else query_cap
            updated = dict(row)
            updated["original_trusted_factual_text"] = row.get("trusted_factual_text", "")
            updated["original_trusted_prompt"] = row.get("trusted_prompt", "")
            updated["after_best_cap"] = best_cap
            updated["after_query_cap"] = query_cap
            updated["after_caption_backend"] = args.backend
            updated["after_trusted_caption_mode"] = args.trusted_caption_mode
            updated["trusted_factual_text"] = trusted_text
            updated["trusted_prompt"] = render_trusted_prompt(trusted_text, str(row.get("question") or ""))
            updated["prompt_style"] = "after_fas_caption_aligned"
            output_rows.append(updated)
            if len(examples) < 3:
                examples.append(
                    {
                        "split": split,
                        "pair_id": row_id,
                        "type": row.get("hallucination_type", ""),
                        "subtype": row.get("subtype", ""),
                        "question": row.get("question", ""),
                        "old_trusted_factual_text": row.get("trusted_factual_text", ""),
                        "after_best_cap": best_cap,
                        "after_query_cap": query_cap,
                        "new_trusted_prompt": updated["trusted_prompt"],
                    }
                )
        output_split_rows[split] = output_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output_split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "image_caption_prompts.jsonl", image_prompt_rows)
    write_jsonl(output_dir / "query_caption_prompts.jsonl", query_prompt_rows)
    write_jsonl(
        output_dir / "caption_cache_template.jsonl",
        [
            {"image_key": key, "best_cap": text, "caption_source": "template_fallback"}
            for key, text in sorted(template_caption_cache.items())
        ],
    )
    if generated_best_rows:
        write_jsonl(output_dir / "generated_best_caps.jsonl", generated_best_rows)
    if generated_query_rows:
        write_jsonl(output_dir / "generated_query_caps.jsonl", generated_query_rows)

    rows_by_type = Counter(str(row.get("hallucination_type") or "unknown") for row in all_rows)
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": splits,
        "backend": args.backend,
        "trusted_caption_mode": args.trusted_caption_mode,
        "total_rows": len(all_rows),
        "unique_images": len(grouped),
        "rows_by_split": rows_by_split,
        "rows_by_type": dict(rows_by_type),
        "caption_cache": str(resolve_project_path(args.caption_cache)) if args.caption_cache else "",
        "query_cache": str(resolve_project_path(args.query_cache)) if args.query_cache else "",
        "caption_cache_hits": caption_cache_hits,
        "query_cache_hits": query_cache_hits,
        "openai_model": str(args.openai_model) if args.backend == "openai" else "",
        "openai_api_key_env": str(args.openai_api_key_env) if args.backend == "openai" else "",
        "openai_generate_query_captions": bool(args.openai_generate_query_captions),
        "generated_best_caps": len(generated_best_rows),
        "generated_query_caps": len(generated_query_rows),
    }
    write_json(output_dir / "stats.json", stats)
    (output_dir / "REPORT.md").write_text(
        render_report(input_dir=input_dir, output_dir=output_dir, stats=stats, examples=examples),
        encoding="utf-8",
    )

    print(f"Wrote AFTER-style caption-aligned pairs to {output_dir}")
    print(f"Wrote image prompts to {output_dir / 'image_caption_prompts.jsonl'}")
    print(f"Wrote query prompts to {output_dir / 'query_caption_prompts.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
