"""Extract AFTER-template activations with the official LLaVA loader.

This is the official-LLaVA counterpart of
``scripts/extract_after_template_activations.py``. It intentionally keeps the
same output cache schema so downstream vector builders can be reused:

- ``z_visual``: image + visual_prompt activation
- ``z_text``: trusted_prompt activation, text-only by default

The downstream steering direction remains ``z_text - z_visual``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.activation_cache import sha256_file, tensor_shape, utc_now_iso, write_json, write_jsonl
from expert_data.image_resolver import CocoImageResolver
from expert_data.io_utils import read_jsonl
from expert_data.model_adapter import find_decoder_layers


REQUIRED_FIELDS = (
    "id",
    "image_id",
    "question",
    "visual_prompt",
    "trusted_factual_text",
    "trusted_prompt",
    "hallucination_type",
    "subtype",
)


@dataclass
class OfficialLlavaImports:
    """Small bundle of official LLaVA symbols loaded from a repo path."""

    image_token_index: int
    default_image_token: str
    default_im_start_token: str
    default_im_end_token: str
    conv_templates: Any
    tokenizer_image_token: Any
    get_model_name_from_path: Any
    load_pretrained_model: Any
    disable_torch_init: Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for official-LLaVA activation extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Official LLaVA checkpoint path.")
    parser.add_argument("--model-base", default=None, help="Optional base model for LoRA checkpoints.")
    parser.add_argument("--llava-repo-path", required=True, help="Path to the official LLaVA repository.")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--pair-file", required=True, help="AFTER-template pair JSONL.")
    parser.add_argument("--image-root", required=True, help="Image directory for relative image filenames.")
    parser.add_argument("--instances-json", default="", help="Optional COCO instances json for image id resolution.")
    parser.add_argument("--output", required=True, help="Output .pt cache path.")
    parser.add_argument("--metadata-output", default="", help="Output metadata JSONL path.")
    parser.add_argument("--types", nargs="*", default=[], help="Optional hallucination types to keep, e.g. --types cat.")
    parser.add_argument("--layers", default="all", help="Records all layers; kept for provenance.")
    parser.add_argument("--position-mode", choices=["last_token"], default="last_token")
    parser.add_argument("--trusted-input-mode", choices=["text_only", "image_with_fact"], default="text_only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--split", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compat-new-transformers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_optional_project_path(raw_path: str | Path | None) -> Path | None:
    """Resolve optional project-relative paths, treating empty values as absent."""

    text = "" if raw_path is None else str(raw_path).strip()
    if not text:
        return None
    return resolve_project_path(text)


def require_torch() -> Any:
    """Import torch lazily so CLI help remains lightweight."""

    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("extract_after_template_activations_official_llava requires torch.") from exc


def require_pil_image() -> Any:
    """Import PIL lazily."""

    try:
        from PIL import Image

        return Image
    except Exception as exc:
        raise RuntimeError("extract_after_template_activations_official_llava requires Pillow.") from exc


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve a user-facing storage dtype."""

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(dtype_name)]


def import_official_llava(llava_repo_path: str | Path) -> OfficialLlavaImports:
    """Import official LLaVA modules from a given local repository."""

    repo_path = Path(llava_repo_path).expanduser().resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
    except Exception as exc:
        raise ImportError(f"Could not import official LLaVA from {repo_path}: {exc!r}") from exc
    return OfficialLlavaImports(
        image_token_index=IMAGE_TOKEN_INDEX,
        default_image_token=DEFAULT_IMAGE_TOKEN,
        default_im_start_token=DEFAULT_IM_START_TOKEN,
        default_im_end_token=DEFAULT_IM_END_TOKEN,
        conv_templates=conv_templates,
        tokenizer_image_token=tokenizer_image_token,
        get_model_name_from_path=get_model_name_from_path,
        load_pretrained_model=load_pretrained_model,
        disable_torch_init=disable_torch_init,
    )


def maybe_apply_new_transformers_compat(model: Any) -> None:
    """Patch older official LLaVA models for newer transformers when needed."""

    if getattr(model, "_official_activation_compat_applied", False):
        return
    original_forward = model.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("cache_position", None)
        return original_forward(*args, **kwargs)

    model.forward = patched_forward
    setattr(model, "_official_activation_compat_applied", True)


def model_device(model: Any) -> Any:
    """Return the first parameter device for a loaded model."""

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def model_float_dtype(model: Any, torch: Any) -> Any:
    """Return a safe floating dtype for image tensors."""

    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float16
    return dtype if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float16


def load_official_model(args: argparse.Namespace, llava: OfficialLlavaImports) -> tuple[Any, Any, Any, int, str]:
    """Load tokenizer/model/image_processor through official LLaVA."""

    torch = require_torch()
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
            device_map={"": str(args.device)},
        )
    except TypeError:
        tokenizer, model, image_processor, context_len = llava.load_pretrained_model(
            str(args.model_path),
            args.model_base,
            model_name,
        )
    if str(args.device):
        wanted = torch.device(str(args.device))
        if model_device(model) != wanted:
            model.to(wanted)
    if args.compat_new_transformers:
        maybe_apply_new_transformers_compat(model)
    model.eval()
    return tokenizer, model, image_processor, int(context_len), str(model_name)


def validate_row(row: Mapping[str, Any], row_index: int) -> None:
    """Validate one AFTER-template pair row."""

    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
    if missing:
        raise ValueError(f"row_index={row_index} missing field(s): {', '.join(missing)}")
    if str(row["hallucination_type"]) not in {"cat", "attr", "rel"}:
        raise ValueError(f"row_index={row_index} has unsupported hallucination_type={row['hallucination_type']}")


def resolve_image_path(row: Mapping[str, Any], image_root: Path, resolver: CocoImageResolver | None) -> str:
    """Resolve the image path for one pair row."""

    image = str(row.get("image") or row.get("image_path") or "").strip()
    if image:
        path = Path(image)
        if not path.is_absolute():
            path = image_root / path
        if path.exists():
            return str(path)
    if resolver is None:
        raise FileNotFoundError(f"Could not resolve image for id={row.get('id')}")
    return resolver.resolve(row["image_id"])


def image_token_for_model(model: Any, llava: OfficialLlavaImports) -> str:
    """Return the right image-token string for this LLaVA config."""

    image_token = llava.default_image_token
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = llava.default_im_start_token + image_token + llava.default_im_end_token
    return image_token


def build_conv_prompt(
    user_text: str,
    *,
    include_image: bool,
    model: Any,
    conv_mode: str,
    llava: OfficialLlavaImports,
) -> str:
    """Build a full official-LLaVA conversation prompt."""

    if include_image:
        user_text = image_token_for_model(model, llava) + "\n" + user_text
    conv = llava.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class OfficialLlavaActivationExtractor:
    """Capture official-LLaVA layer/head activations for prompt pairs."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        image_processor: Any,
        llava: OfficialLlavaImports,
        conv_mode: str,
        storage_dtype: str,
    ) -> None:
        self.torch = require_torch()
        self.Image = require_pil_image()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.llava = llava
        self.conv_mode = str(conv_mode)
        self.storage_dtype_name = str(storage_dtype)
        self.storage_dtype = torch_dtype(self.torch, storage_dtype)
        self.decoder_layers = list(find_decoder_layers(self.model))
        self.num_heads, self.hidden_size = self._resolve_head_config()
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError(f"hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}")
        self.head_dim = self.hidden_size // self.num_heads
        self._current_target_idx: int | None = None
        self._actual_target_idx: int | None = None
        self._actual_sequence_len: int | None = None
        self._layer_outputs: dict[int, Any] = {}
        self._hooks: list[Any] = []
        self._register_hooks()

    def close(self) -> None:
        """Remove registered hooks."""

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _resolve_head_config(self) -> tuple[int, int]:
        """Resolve number of attention heads and hidden size."""

        config = getattr(self.model, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        if num_heads is None and self.decoder_layers:
            num_heads = getattr(getattr(self.decoder_layers[0], "self_attn", None), "num_heads", None)
        if hidden_size is None and self.decoder_layers:
            hidden_size = getattr(getattr(self.decoder_layers[0], "self_attn", None), "hidden_size", None)
        if num_heads is None or hidden_size is None:
            raise RuntimeError("Could not resolve num_attention_heads and hidden_size from the official LLaVA model.")
        return int(num_heads), int(hidden_size)

    def _register_hooks(self) -> None:
        """Register o_proj pre-hooks and capture the selected token activation."""

        def make_hook(layer_index: int) -> Any:
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                if self._current_target_idx is None or not inputs:
                    return
                hidden_states = inputs[0]
                if int(self._current_target_idx) < 0:
                    actual_idx = int(hidden_states.shape[1] - 1)
                else:
                    actual_idx = int(self._current_target_idx)
                if actual_idx >= int(hidden_states.shape[1]):
                    raise RuntimeError(
                        f"Target index {actual_idx} out of range for hidden sequence length {hidden_states.shape[1]}"
                    )
                if self._actual_target_idx is None:
                    self._actual_target_idx = actual_idx
                    self._actual_sequence_len = int(hidden_states.shape[1])
                target_hidden = hidden_states[:, actual_idx, :].detach()
                target_hidden = target_hidden.reshape(target_hidden.shape[0], self.num_heads, self.head_dim)
                self._layer_outputs[layer_index] = target_hidden[0].to("cpu").to(self.storage_dtype)

            return hook

        for layer_index, layer in enumerate(self.decoder_layers):
            self_attn = getattr(layer, "self_attn", None)
            o_proj = getattr(self_attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"Layer {layer_index} does not expose self_attn.o_proj")
            self._hooks.append(o_proj.register_forward_pre_hook(make_hook(layer_index)))

    def _visual_inputs(self, prompt: str, image_path: str) -> tuple[dict[str, Any], int, tuple[int, int]]:
        """Tokenize an image prompt with official tokenizer_image_token."""

        image = self.Image.open(image_path).convert("RGB")
        input_ids = self.llava.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.llava.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0)
        device = model_device(self.model)
        image_dtype = model_float_dtype(self.model, self.torch)
        # Match the already-validated official POPE runner: LLaVA-1.5 expects
        # `images` here, and older forks can mis-handle explicit image_sizes.
        inputs = {
            "input_ids": input_ids.to(device),
            "images": image_tensor.to(device=device, dtype=image_dtype),
        }
        return inputs, int(input_ids.shape[1] - 1), image.size

    def _text_inputs(self, prompt: str) -> tuple[dict[str, Any], int]:
        """Tokenize a text-only official LLaVA prompt."""

        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = model_device(self.model)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(encoded).items()}
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Tokenizer output did not include input_ids.")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            positions = attention_mask[0].nonzero(as_tuple=False).flatten()
            if len(positions) == 0:
                raise RuntimeError("Text prompt tokenized to an empty sequence.")
            tokenized_target = int(positions[-1].item())
        else:
            tokenized_target = int(input_ids.shape[1] - 1)
        return inputs, tokenized_target

    def _forward_with_optional_image_sizes(self, inputs: dict[str, Any]) -> None:
        """Forward once, retrying without image_sizes for older official LLaVA signatures."""

        with self.torch.inference_mode():
            try:
                self.model(**inputs, use_cache=False)
            except TypeError as exc:
                if "image_sizes" not in inputs:
                    raise
                retry_inputs = dict(inputs)
                retry_inputs.pop("image_sizes", None)
                try:
                    self.model(**retry_inputs, use_cache=False)
                except TypeError:
                    raise exc

    def _run_prompt(self, prompt: str, *, image_path: str | None, include_image: bool) -> tuple[Any, int, int, int]:
        """Run one prompt and return activations plus target diagnostics."""

        if include_image:
            if not image_path:
                raise ValueError("image_path is required when include_image=True")
            inputs, tokenized_target, _image_size = self._visual_inputs(prompt, image_path)
        else:
            inputs, tokenized_target = self._text_inputs(prompt)
        self._current_target_idx = -1
        self._actual_target_idx = None
        self._actual_sequence_len = None
        self._layer_outputs = {}
        self._forward_with_optional_image_sizes(inputs)
        self._current_target_idx = None
        missing_layers = [index for index in range(len(self.decoder_layers)) if index not in self._layer_outputs]
        if missing_layers:
            raise RuntimeError(f"Missing hooked activations for decoder layers: {missing_layers[:5]}")
        stacked = self.torch.stack([self._layer_outputs[index] for index in range(len(self.decoder_layers))], dim=0)
        actual_target = int(self._actual_target_idx if self._actual_target_idx is not None else tokenized_target)
        sequence_len = int(self._actual_sequence_len if self._actual_sequence_len is not None else actual_target + 1)
        return stacked.to("cpu").to(self.storage_dtype), tokenized_target, actual_target, sequence_len

    def encode_prompt_pair(
        self,
        *,
        image_path: str,
        visual_prompt: str,
        trusted_prompt: str,
        trusted_input_mode: str,
    ) -> dict[str, Any]:
        """Encode visual-query and trusted-text branches."""

        mode = str(trusted_input_mode).strip().lower()
        if mode not in {"text_only", "image_with_fact"}:
            raise ValueError("trusted_input_mode must be one of: text_only, image_with_fact")
        visual_full_prompt = build_conv_prompt(
            str(visual_prompt),
            include_image=True,
            model=self.model,
            conv_mode=self.conv_mode,
            llava=self.llava,
        )
        trusted_full_prompt = build_conv_prompt(
            str(trusted_prompt),
            include_image=(mode == "image_with_fact"),
            model=self.model,
            conv_mode=self.conv_mode,
            llava=self.llava,
        )
        z_visual, token_visual, actual_visual, visual_seq_len = self._run_prompt(
            visual_full_prompt,
            image_path=image_path,
            include_image=True,
        )
        z_text, token_text, actual_text, text_seq_len = self._run_prompt(
            trusted_full_prompt,
            image_path=image_path if mode == "image_with_fact" else None,
            include_image=(mode == "image_with_fact"),
        )
        return {
            "z_visual": z_visual,
            "z_text": z_text,
            "meta": {
                "num_layers": int(z_visual.shape[0]),
                "num_heads": int(z_visual.shape[1]),
                "head_dim": int(z_visual.shape[2]),
                "target_token_index_visual": int(actual_visual),
                "target_token_index_text": int(actual_text),
                "tokenized_target_index_visual": int(token_visual),
                "tokenized_target_index_text": int(token_text),
                "hidden_sequence_len_visual_approx": int(visual_seq_len),
                "hidden_sequence_len_text_approx": int(text_seq_len),
                "visual_full_prompt": visual_full_prompt,
                "trusted_full_prompt": trusted_full_prompt,
            },
        }


def metadata_row(
    row_index: int,
    row: Mapping[str, Any],
    image_path: str,
    split: str,
    branch_meta: Mapping[str, Any],
    model_path: str,
    model_name: str,
    llava_repo_path: str,
    conv_mode: str,
    trusted_input_mode: str,
) -> dict[str, Any]:
    """Build one metadata row for an official-LLaVA activation."""

    identifier = str(row.get("id") or row.get("pair_id"))
    return {
        "row_index": int(row_index),
        "id": identifier,
        "pair_id": identifier,
        "image_id": str(row["image_id"]),
        "image": str(row.get("image", "")),
        "image_path": image_path,
        "split": split or None,
        "hallucination_type": str(row["hallucination_type"]),
        "subtype": str(row["subtype"]),
        "objects": list(row.get("objects", [])),
        "question": str(row["question"]),
        "visual_prompt": str(row["visual_prompt"]),
        "trusted_factual_text": str(row["trusted_factual_text"]),
        "trusted_prompt": str(row["trusted_prompt"]),
        "factual_fact": str(row.get("factual_fact", "")),
        "source": str(row.get("source", "after_template_v2")),
        "label": str(row.get("label", "")),
        "object_a": str(row.get("object_a", "")),
        "object_b": str(row.get("object_b", "")),
        "bbox_a": list(row.get("bbox_a", [])),
        "bbox_b": list(row.get("bbox_b", [])),
        "true_relation": str(row.get("true_relation", "")),
        "queried_relation": str(row.get("queried_relation", "")),
        "relation_bucket": str(row.get("relation_bucket", "")),
        "template_variant": str(row.get("template_variant", "")),
        "target_token_index_visual": int(branch_meta["target_token_index_visual"]),
        "target_token_index_text": int(branch_meta["target_token_index_text"]),
        "tokenized_target_index_visual": int(branch_meta["tokenized_target_index_visual"]),
        "tokenized_target_index_text": int(branch_meta["tokenized_target_index_text"]),
        "hidden_sequence_len_visual_approx": int(branch_meta["hidden_sequence_len_visual_approx"]),
        "hidden_sequence_len_text_approx": int(branch_meta["hidden_sequence_len_text_approx"]),
        "num_layers": int(branch_meta["num_layers"]),
        "num_heads": int(branch_meta["num_heads"]),
        "head_dim": int(branch_meta["head_dim"]),
        "adapter": "official_llava",
        "model_id": model_path,
        "model_name": model_name,
        "llava_repo_path": llava_repo_path,
        "conv_mode": conv_mode,
        "trusted_input_mode": trusted_input_mode,
    }


def stack_activation_items(items: list[Any], storage_dtype: str) -> Any:
    """Stack per-row activation grids into a [N,L,H,D] tensor."""

    if not items:
        return []
    torch = require_torch()
    return torch.stack(items, dim=0).to(torch_dtype(torch, storage_dtype)).cpu()


def selected_rows(rows: list[dict[str, Any]], allowed_types: set[str], max_samples: int) -> list[tuple[int, dict[str, Any]]]:
    """Filter rows by hallucination type while preserving original row indices."""

    selected: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        if allowed_types and str(row.get("hallucination_type")) not in allowed_types:
            continue
        selected.append((row_index, row))
        if int(max_samples) > 0 and len(selected) >= int(max_samples):
            break
    return selected


def main() -> int:
    """Extract official-LLaVA AFTER-template activations."""

    args = parse_args()
    try:
        torch = require_torch()
        torch.manual_seed(int(args.seed))
        pair_path = resolve_project_path(args.pair_file)
        image_root = resolve_project_path(args.image_root)
        output_path = resolve_project_path(args.output)
        metadata_path = (
            resolve_project_path(args.metadata_output)
            if str(args.metadata_output).strip()
            else output_path.with_suffix(".meta.jsonl")
        )
        manifest_path = output_path.with_suffix(".manifest.json")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace.")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")

        allowed_types = {str(item).strip() for item in args.types if str(item).strip()}
        unsupported = sorted(allowed_types - {"cat", "attr", "rel"})
        if unsupported:
            raise ValueError(f"Unsupported --types values: {unsupported}")
        rows = read_jsonl(pair_path)
        row_items = selected_rows(rows, allowed_types, int(args.max_samples))
        if not row_items:
            raise ValueError("No rows selected for extraction.")
        for source_row_index, row in row_items:
            validate_row(row, source_row_index)

        instances_json = resolve_optional_project_path(args.instances_json)
        resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json) if instances_json else None
        llava = import_official_llava(args.llava_repo_path)
        tokenizer, model, image_processor, context_len, model_name = load_official_model(args, llava)
        extractor = OfficialLlavaActivationExtractor(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            llava=llava,
            conv_mode=str(args.conv_mode),
            storage_dtype=str(args.storage_dtype),
        )

        z_text_items: list[Any] = []
        z_visual_items: list[Any] = []
        metadata_rows: list[dict[str, Any]] = []
        first_shape: list[int] | None = None
        try:
            for processed_index, (source_row_index, row) in enumerate(row_items, start=1):
                image_path = resolve_image_path(row, image_root, resolver)
                result = extractor.encode_prompt_pair(
                    image_path=image_path,
                    visual_prompt=str(row["visual_prompt"]),
                    trusted_prompt=str(row["trusted_prompt"]),
                    trusted_input_mode=str(args.trusted_input_mode),
                )
                shape = tensor_shape(result["z_text"])
                if first_shape is None:
                    first_shape = shape
                elif shape != first_shape:
                    raise RuntimeError(f"Inconsistent activation shape: {shape} != {first_shape}")
                if tensor_shape(result["z_visual"]) != first_shape:
                    raise RuntimeError(f"Visual/text activation shapes differ at row_index={source_row_index}")
                z_text_items.append(result["z_text"])
                z_visual_items.append(result["z_visual"])
                metadata_rows.append(
                    metadata_row(
                        source_row_index,
                        row,
                        image_path,
                        str(args.split),
                        result["meta"],
                        str(args.model_path),
                        model_name,
                        str(args.llava_repo_path),
                        str(args.conv_mode),
                        str(args.trusted_input_mode),
                    )
                )
                if int(args.progress_every) > 0 and processed_index % int(args.progress_every) == 0:
                    print(
                        f"[official-after-extract] processed {processed_index}/{len(row_items)} rows "
                        f"(source row {source_row_index})"
                    )
        finally:
            extractor.close()

        z_text = stack_activation_items(z_text_items, args.storage_dtype)
        z_visual = stack_activation_items(z_visual_items, args.storage_dtype)
        cache = {
            "pair_ids": [row["pair_id"] for row in metadata_rows],
            "row_indices": [row["row_index"] for row in metadata_rows],
            "image_ids": [row["image_id"] for row in metadata_rows],
            "hallucination_types": [row["hallucination_type"] for row in metadata_rows],
            "subtypes": [row["subtype"] for row in metadata_rows],
            "layers": list(range(int(first_shape[0]))) if first_shape else [],
            "metadata": metadata_rows,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_pos": z_text,
            "z_neg": z_visual,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output_path)
        write_jsonl(metadata_path, metadata_rows)
        manifest = {
            "source": "after_template_official_llava",
            "pair_file": str(pair_path),
            "pair_file_sha256": sha256_file(pair_path),
            "output": str(output_path),
            "metadata_output": str(metadata_path),
            "model_path": str(args.model_path),
            "model_base": args.model_base,
            "model_name": model_name,
            "llava_repo_path": str(args.llava_repo_path),
            "conv_mode": str(args.conv_mode),
            "context_len": int(context_len),
            "types": sorted(allowed_types) if allowed_types else ["cat", "attr", "rel"],
            "split": args.split or None,
            "layers": args.layers,
            "position_mode": args.position_mode,
            "trusted_input_mode": str(args.trusted_input_mode),
            "num_pairs": len(metadata_rows),
            "shape": [len(metadata_rows), *(first_shape or [0, 0, 0])],
            "dtype": str(args.storage_dtype),
            "seed": int(args.seed),
            "created_at": utc_now_iso(),
            "notes": [
                "Official LLaVA conv_templates/tokenizer_image_token/image_processor extraction",
                "z_visual is image + visual_prompt last-token hidden activation",
                "z_text is trusted_prompt last-token hidden activation",
                "image-token prompts capture the final expanded hidden-state position, not raw input_ids[-1]",
                "downstream direction is z_text - z_visual",
            ],
        }
        write_json(manifest_path, manifest)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote official-LLaVA AFTER-template activations to {output_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
