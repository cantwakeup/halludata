"""Teacher-forced candidate scoring and additive expert steering hooks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

from expert_data.image_resolver import CocoImageResolver
from expert_data.model_adapter import find_decoder_layers

VALID_SUBTYPES = {"cat", "cnt", "col", "rel"}
VALID_EXPERTS = ("cat", "attr", "rel")


def _stable_float(text: str) -> float:
    """Map text to a small deterministic pseudo-random float."""

    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return (value % 1000) / 1000.0


def build_llava_prefix_prompt(question: str) -> str:
    """Build the LLaVA prompt prefix before the answer text."""

    return f"USER: <image>\n{question}\nASSISTANT:"


def build_llava_full_prompt(question: str, response: str) -> str:
    """Build the full teacher-forced LLaVA prompt for one answer candidate."""

    return f"{build_llava_prefix_prompt(question)} {response}"


def parse_csv_items(text: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse comma-separated CLI text into stripped non-empty items."""

    if isinstance(text, (list, tuple)):
        values = [str(item) for item in text]
    else:
        values = str(text).split(",")
    return [value.strip() for value in values if value.strip()]


def parse_layer_spec(layer_spec: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse a layer spec like `10-20` or `10,12,14` into sorted indices."""

    if isinstance(layer_spec, (list, tuple)):
        layers = sorted({int(layer) for layer in layer_spec})
    else:
        layers_set: set[int] = set()
        for chunk in str(layer_spec).split(","):
            part = chunk.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError(f"Invalid descending layer range: {part}")
                layers_set.update(range(start, end + 1))
            else:
                layers_set.add(int(part))
        layers = sorted(layers_set)
    if not layers:
        raise ValueError("At least one steering layer must be selected")
    if min(layers) < 0:
        raise ValueError("Steering layer indices must be non-negative")
    return layers


def normalize_bool(value: str | bool) -> bool:
    """Parse a flexible CLI-style boolean value."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def route_question_to_experts(question: str, router: str, enabled_experts: tuple[str, ...]) -> tuple[str, ...]:
    """Route a benchmark question to one or more enabled expert vectors."""

    enabled = tuple(str(expert) for expert in enabled_experts if str(expert))
    if not enabled:
        raise ValueError("At least one enabled expert must be provided")
    router_name = str(router).strip().lower()
    if router_name == "no_filter":
        return enabled
    if router_name.startswith("force_"):
        expert = router_name.replace("force_", "", 1)
        if expert in enabled:
            return (expert,)
        if expert == "rel":
            rel_like = tuple(item for item in enabled if item == "rel" or item.startswith("rel_"))
            if rel_like:
                return rel_like
        return enabled
    if router_name != "rule":
        raise ValueError(f"Unsupported steering router: {router}")

    text = str(question).lower()
    attr_keywords = ("how many", "number", "count", "color", "what color", "many")
    rel_keywords = (
        "left",
        "right",
        "above",
        "below",
        "under",
        "next to",
        "behind",
        "in front of",
        "position",
        "where",
    )
    cat_keywords = ("object", "is there", "does image contain", "what is in", "what objects", "contain")
    if any(keyword in text for keyword in attr_keywords) and "attr" in enabled:
        return ("attr",)
    if any(keyword in text for keyword in rel_keywords) and "rel" in enabled:
        return ("rel",)
    if any(keyword in text for keyword in cat_keywords) and "cat" in enabled:
        return ("cat",)
    return enabled


def select_top_heads(
    head_ranking: Mapping[str, list[Mapping[str, Any]]],
    subtype: str,
    top_k: int,
) -> list[tuple[int, int]]:
    """Select the top-K `(layer, head)` pairs for one subtype from ranking JSON."""

    rows = list(head_ranking.get(str(subtype), []))
    return [
        (int(row["layer"]), int(row["head"]))
        for row in rows[: max(int(top_k), 0)]
    ]


class ExpertSteeringController:
    """Add typed expert steering vectors to selected attention heads during generation."""

    def __init__(
        self,
        model: Any,
        vector_path: str | Path,
        layers: str | list[int] | tuple[int, ...],
        alpha: float = 1.0,
        k_heads: int = 64,
        head_select: str = "norm",
        router: str = "no_filter",
        enabled_experts: tuple[str, ...] | list[str] = VALID_EXPERTS,
        apply_to: str = "last_token",
        steer_prefill: bool = False,
        steer_decode: bool = True,
        prefill_apply_to: str | None = None,
        decode_apply_to: str | None = None,
        debug_log_hook_delta: bool = False,
        debug_random_vector: bool = False,
        debug_random_seed: int = 42,
        head_map_path: str | Path | None = None,
        expert_key: str | None = None,
        seed: int = 42,
    ) -> None:
        """Load expert vectors, select heads, and register disabled hooks on a model."""

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("ExpertSteeringController requires a working torch installation.") from exc

        self._torch = torch
        self.model = model
        self.vector_path = Path(vector_path)
        self.alpha = float(alpha)
        self.k_heads = int(k_heads)
        self.head_select = str(head_select)
        self.router = str(router)
        self.enabled_experts = tuple(str(expert) for expert in enabled_experts if str(expert))
        if not self.enabled_experts:
            raise ValueError("enabled_experts must include at least one vector key")
        self.head_map_path = Path(head_map_path) if head_map_path not in (None, "") else None
        self.expert_key = str(expert_key).strip() if expert_key not in (None, "") else None
        if self.expert_key and self.expert_key not in self.enabled_experts:
            self.enabled_experts = (self.expert_key,)
        self.expert_head_map = self._load_head_map(self.head_map_path, self.expert_key)
        self.apply_to = str(apply_to)
        self.steer_prefill = bool(steer_prefill)
        self.steer_decode = bool(steer_decode)
        self.prefill_apply_to = str(prefill_apply_to or apply_to)
        self.decode_apply_to = str(decode_apply_to or "last_token")
        self.debug_log_hook_delta = bool(debug_log_hook_delta)
        self.debug_random_vector = bool(debug_random_vector)
        self.debug_random_seed = int(debug_random_seed)
        self.seed = int(seed)
        self.enabled = False
        self.current_sign = 1.0
        self.hook_call_count = 0
        self.edited_token_count = 0
        self.prefill_hook_call_count = 0
        self.decode_hook_call_count = 0
        self.prefill_edited_token_count = 0
        self.decode_edited_token_count = 0
        self.last_hook_shapes: list[dict[str, Any]] = []
        self.hook_delta_by_layer: dict[int, dict[str, float | int | list[int]]] = {}

        payload = self._load_vector_payload(self.vector_path)
        self.vector_layers = [int(layer) for layer in payload["layers"]]
        self.requested_layers = self._resolve_requested_layers(layers)
        missing_layers = [layer for layer in self.requested_layers if layer not in self.vector_layers]
        if missing_layers:
            raise ValueError(f"Requested layers are missing from vector file: {missing_layers}")
        self.num_heads = int(payload["num_heads"])
        self.head_dim = int(payload["head_dim"])
        self.hidden_size = int(payload["hidden_size"])
        self.vectors_by_expert_layer = self._index_vectors(payload["vectors"])
        if self.debug_random_vector:
            self._replace_with_debug_random_vectors()
        self.decoder_layers = list(find_decoder_layers(self.model))
        if max(self.requested_layers) >= len(self.decoder_layers):
            raise ValueError(f"Requested layer exceeds model depth: {max(self.requested_layers)} >= {len(self.decoder_layers)}")
        self.active_experts = route_question_to_experts("", self.router, self.enabled_experts)
        self.active_vectors_by_layer: dict[int, Any] = {}
        self.active_heads_by_layer: dict[int, list[int]] = {}
        self._hooks: list[Any] = []
        self.set_context("")
        self._register_hooks()

    def _load_vector_payload(self, path: Path) -> Mapping[str, Any]:
        """Load an expert-vector torch file."""

        try:
            return self._torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return self._torch.load(path, map_location="cpu")

    def _load_head_map(self, path: Path | None, expert_key: str | None) -> dict[int, list[int]] | None:
        """Load an expert-specific head map from a mining JSON file."""

        if self.head_select != "expert_map":
            return None
        if path is None:
            raise ValueError("--steer-head-map is required when head_select='expert_map'")
        if expert_key is None:
            raise ValueError("--steer-expert-key is required when head_select='expert_map'")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if expert_key not in payload:
            raise ValueError(f"Head map {path} does not contain expert key '{expert_key}'")
        heads_by_layer: dict[int, list[int]] = {}
        for row in payload[expert_key]:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                raise ValueError(f"Invalid head-map row for {expert_key}: {row!r}")
            layer = int(row[0])
            head = int(row[1])
            heads_by_layer.setdefault(layer, []).append(head)
        if not heads_by_layer:
            raise ValueError(f"Head map for expert key '{expert_key}' is empty")
        return {layer: sorted(set(heads)) for layer, heads in sorted(heads_by_layer.items())}

    def _resolve_requested_layers(self, layers: str | list[int] | tuple[int, ...]) -> list[int]:
        """Resolve requested layers, allowing expert maps to infer the hook layers."""

        if self.head_select == "expert_map":
            if self.expert_head_map is None:
                raise ValueError("expert head map must be loaded before resolving layers")
            return sorted(self.expert_head_map)
        return parse_layer_spec(layers)

    def _index_vectors(self, vectors: Mapping[str, Any]) -> dict[str, dict[int, Any]]:
        """Map each expert's layer-relative tensor rows back to original layer IDs."""

        indexed: dict[str, dict[int, Any]] = {}
        for expert in self.enabled_experts:
            if expert not in vectors:
                raise ValueError(f"Vector file is missing expert '{expert}'")
            tensor = vectors[expert].detach().cpu().float()
            if list(tensor.shape) != [len(self.vector_layers), self.num_heads, self.head_dim]:
                raise ValueError(f"Unexpected vector shape for {expert}: {list(tensor.shape)}")
            indexed[expert] = {
                int(layer): tensor[layer_index]
                for layer_index, layer in enumerate(self.vector_layers)
            }
        return indexed

    def _replace_with_debug_random_vectors(self) -> None:
        """Replace loaded vectors with deterministic random vectors for hook sanity checks."""

        generator = self._torch.Generator(device="cpu")
        generator.manual_seed(self.debug_random_seed)
        for expert in self.enabled_experts:
            for layer in self.vector_layers:
                self.vectors_by_expert_layer[expert][layer] = self._torch.randn(
                    self.num_heads,
                    self.head_dim,
                    generator=generator,
                    dtype=self._torch.float32,
                )

    def _combine_vectors(self, experts: tuple[str, ...]) -> dict[int, Any]:
        """Sum active expert vectors for each requested layer."""

        combined: dict[int, Any] = {}
        for layer in self.requested_layers:
            layer_vector = None
            for expert in experts:
                expert_vector = self.vectors_by_expert_layer[expert][layer]
                layer_vector = expert_vector.clone() if layer_vector is None else layer_vector + expert_vector
            if layer_vector is None:
                layer_vector = self._torch.zeros(self.num_heads, self.head_dim, dtype=self._torch.float32)
            combined[layer] = layer_vector
        return combined

    def _select_heads(self, vectors_by_layer: Mapping[int, Any]) -> dict[int, list[int]]:
        """Select layer-head pairs according to the configured policy."""

        if self.head_select == "all":
            return {int(layer): list(range(self.num_heads)) for layer in self.requested_layers}
        if self.head_select == "expert_map":
            if self.expert_head_map is None:
                raise ValueError("expert head map is not loaded")
            return {
                int(layer): list(heads)
                for layer, heads in self.expert_head_map.items()
                if int(layer) in self.requested_layers
            }
        all_pairs = [(int(layer), int(head)) for layer in self.requested_layers for head in range(self.num_heads)]
        if self.head_select == "random":
            rng = random.Random(self.seed)
            shuffled = list(all_pairs)
            rng.shuffle(shuffled)
            chosen = shuffled[: max(self.k_heads, 0)]
        elif self.head_select == "norm":
            scored = [
                (
                    float(vectors_by_layer[layer][head].float().norm().item()),
                    int(layer),
                    int(head),
                )
                for layer, head in all_pairs
            ]
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            chosen = [(layer, head) for _score, layer, head in scored[: max(self.k_heads, 0)]]
        else:
            raise ValueError(f"Unsupported head_select mode: {self.head_select}")
        heads_by_layer: dict[int, list[int]] = {}
        for layer, head in chosen:
            heads_by_layer.setdefault(layer, []).append(head)
        return heads_by_layer

    def set_context(self, question: str) -> None:
        """Set the current question context and recompute active experts/heads."""

        self.active_experts = route_question_to_experts(question, self.router, self.enabled_experts)
        self.active_vectors_by_layer = self._combine_vectors(self.active_experts)
        self.active_heads_by_layer = self._select_heads(self.active_vectors_by_layer)

    def enable(self) -> None:
        """Enable steering hooks for subsequent model forwards."""

        self.enabled = True

    def disable(self) -> None:
        """Disable steering hooks without removing them."""

        self.enabled = False

    def set_sign(self, sign: int | float) -> None:
        """Set the signed steering multiplier used by subsequent forwards."""

        self.current_sign = float(sign)

    def remove(self) -> None:
        """Remove all registered hooks from the model."""

        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def reset_diagnostics(self) -> None:
        """Reset hook counters without changing the active steering configuration."""

        self.hook_call_count = 0
        self.edited_token_count = 0
        self.prefill_hook_call_count = 0
        self.decode_hook_call_count = 0
        self.prefill_edited_token_count = 0
        self.decode_edited_token_count = 0
        self.last_hook_shapes = []
        self.hook_delta_by_layer = {}

    def _forward_kind(self, hidden_states: Any) -> str:
        """Classify a hook input as prefill (`T > 1`) or decode (`T == 1`)."""

        seq_len = int(hidden_states.shape[1])
        return "prefill" if seq_len > 1 else "decode"

    def _target_slice(self, hidden_states: Any) -> tuple[str, Any] | None:
        """Return the forward kind and token positions to edit for one hook call."""

        seq_len = int(hidden_states.shape[1])
        if seq_len <= 0:
            return None
        forward_kind = self._forward_kind(hidden_states)
        if forward_kind == "prefill":
            if not self.steer_prefill:
                return None
            apply_mode = self.prefill_apply_to
        else:
            if not self.steer_decode:
                return None
            apply_mode = self.decode_apply_to
        if apply_mode == "last_token":
            return forward_kind, slice(seq_len - 1, seq_len)
        if apply_mode == "all_tokens" and forward_kind == "prefill":
            return forward_kind, slice(0, seq_len)
        if apply_mode == "all_tokens" and forward_kind == "decode":
            return forward_kind, slice(seq_len - 1, seq_len)
        raise ValueError(f"Unsupported {forward_kind} apply mode: {apply_mode}")

    def _record_hook_delta(
        self,
        *,
        layer_index: int,
        hidden_states: Any,
        token_slice: Any,
        layer_heads: list[int],
        before: Any,
        after: Any,
    ) -> None:
        """Record a compact first-hit edit diagnostic for one hooked layer."""

        if not self.debug_log_hook_delta or layer_index in self.hook_delta_by_layer:
            return None
        delta = (after - before).detach().float()
        self.hook_delta_by_layer[layer_index] = {
            "layer": int(layer_index),
            "input_shape": [int(item) for item in hidden_states.shape],
            "num_heads": int(len(layer_heads)),
            "token_start": int(token_slice.start),
            "token_stop": int(token_slice.stop),
            "input_norm_before": float(before.detach().float().norm().item()),
            "edit_norm": float(delta.norm().item()),
            "input_norm_after": float(after.detach().float().norm().item()),
            "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
        }

    def _register_hooks(self) -> None:
        """Register forward-pre hooks on selected decoder attention output projections."""

        def make_hook(layer_index: int) -> Any:
            def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
                if not self.enabled or not inputs:
                    return None
                if float(self.current_sign) == 0.0:
                    return None
                layer_heads = self.active_heads_by_layer.get(layer_index)
                if not layer_heads:
                    return None
                hidden_states = inputs[0]
                target = self._target_slice(hidden_states)
                if target is None:
                    return None
                forward_kind, token_slice = target
                if int(hidden_states.shape[-1]) != self.hidden_size:
                    raise RuntimeError(
                        f"Hidden size mismatch: hook saw {hidden_states.shape[-1]}, vector file expects {self.hidden_size}"
                    )
                shaped = hidden_states.reshape(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self.num_heads,
                    self.head_dim,
                ).clone()
                vectors = self.active_vectors_by_layer[layer_index].to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                before = shaped[:, token_slice, :, :].clone() if self.debug_log_hook_delta else None
                for head in layer_heads:
                    shaped[:, token_slice, head, :] = shaped[:, token_slice, head, :] + (
                        self.current_sign * self.alpha * vectors[head]
                    )
                if before is not None:
                    after = shaped[:, token_slice, :, :]
                    self._record_hook_delta(
                        layer_index=layer_index,
                        hidden_states=hidden_states,
                        token_slice=token_slice,
                        layer_heads=layer_heads,
                        before=before,
                        after=after,
                    )
                self.hook_call_count += 1
                self.edited_token_count += len(layer_heads) * (token_slice.stop - token_slice.start)
                if forward_kind == "prefill":
                    self.prefill_hook_call_count += 1
                    self.prefill_edited_token_count += len(layer_heads) * (token_slice.stop - token_slice.start)
                else:
                    self.decode_hook_call_count += 1
                    self.decode_edited_token_count += len(layer_heads) * (token_slice.stop - token_slice.start)
                if len(self.last_hook_shapes) < 8:
                    self.last_hook_shapes.append(
                        {
                            "layer": layer_index,
                            "forward_kind": forward_kind,
                            "input_shape": [int(item) for item in hidden_states.shape],
                            "num_heads": len(layer_heads),
                            "token_start": int(token_slice.start),
                            "token_stop": int(token_slice.stop),
                        }
                    )
                return (shaped.reshape_as(hidden_states), *inputs[1:])

            return hook

        for layer_index in self.requested_layers:
            layer = self.decoder_layers[layer_index]
            self_attn = getattr(layer, "self_attn", None)
            o_proj = getattr(self_attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"Layer {layer_index} does not expose self_attn.o_proj")
            self._hooks.append(o_proj.register_forward_pre_hook(make_hook(layer_index)))

    def summary(self) -> dict[str, Any]:
        """Return diagnostic information about the active steering configuration."""

        active_head_count = sum(len(heads) for heads in self.active_heads_by_layer.values())
        return {
            "vector_path": str(self.vector_path),
            "layers": list(self.requested_layers),
            "vector_layers": list(self.vector_layers),
            "alpha": self.alpha,
            "current_sign": self.current_sign,
            "k_heads": self.k_heads,
            "head_select": self.head_select,
            "head_map_path": str(self.head_map_path) if self.head_map_path else "",
            "expert_key": self.expert_key or "",
            "router": self.router,
            "enabled_experts": list(self.enabled_experts),
            "active_experts": list(self.active_experts),
            "active_head_count": active_head_count,
            "active_heads_by_layer": {
                str(layer): len(heads)
                for layer, heads in sorted(self.active_heads_by_layer.items())
            },
            "apply_to": self.apply_to,
            "steer_prefill": self.steer_prefill,
            "steer_decode": self.steer_decode,
            "prefill_apply_to": self.prefill_apply_to,
            "decode_apply_to": self.decode_apply_to,
            "debug_log_hook_delta": self.debug_log_hook_delta,
            "debug_random_vector": self.debug_random_vector,
            "hook_call_count": self.hook_call_count,
            "edited_token_count": self.edited_token_count,
            "prefill_hook_call_count": self.prefill_hook_call_count,
            "decode_hook_call_count": self.decode_hook_call_count,
            "prefill_edited_token_count": self.prefill_edited_token_count,
            "decode_edited_token_count": self.decode_edited_token_count,
            "last_hook_shapes": list(self.last_hook_shapes),
            "hook_delta_by_layer": {
                str(layer): dict(values)
                for layer, values in sorted(self.hook_delta_by_layer.items())
            },
        }


def summarize_candidate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pairwise candidate scores for one set of scored rows."""

    if not rows:
        return {
            "num_pairs": 0,
            "pairwise_acc": 0.0,
            "mean_margin": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    margins = [float(row["score_pos"]) - float(row["score_neg"]) for row in rows]
    wins = sum(1 for margin in margins if margin > 0.0)
    ties = sum(1 for margin in margins if margin == 0.0)
    losses = len(margins) - wins - ties
    return {
        "num_pairs": len(rows),
        "pairwise_acc": wins / len(rows),
        "mean_margin": sum(margins) / len(margins),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


class MockCandidateScorer:
    """Deterministic scorer used to test steering-selection code without a real model."""

    def _candidate_alignment(self, response: str) -> float:
        """Return a deterministic pseudo-alignment for one candidate response."""

        text = str(response).lower()
        if "positive" in text or "factual" in text:
            return 1.0
        if "negative" in text or "counterfactual" in text:
            return -1.0
        return (_stable_float(text) * 2.0) - 1.0

    def score_pair(
        self,
        pair: Mapping[str, Any],
        *,
        alpha: float = 0.0,
        sign: float = 1.0,
        selected_heads: list[tuple[int, int]] | None = None,
        steering_vectors: Any | None = None,
    ) -> dict[str, float]:
        """Score candidates with one shared, text-conditioned mock steering effect."""

        del selected_heads, steering_vectors
        pair_id = str(pair["pair_id"])
        base = _stable_float(pair_id) * 0.02
        steering_strength = float(alpha) * float(sign)
        pos_alignment = self._candidate_alignment(str(pair["response_pos"]))
        neg_alignment = self._candidate_alignment(str(pair["response_neg"]))
        return {
            "score_pos": base + steering_strength * pos_alignment,
            "score_neg": base + steering_strength * neg_alignment,
        }


class LlavaCandidateScorer:
    """Score response candidates under baseline or steered LLaVA teacher-forced forward passes."""

    def __init__(
        self,
        model_id: str,
        image_root: str | Path,
        instances_json: str | Path | None = None,
        *,
        device: str = "cuda:0",
        compute_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        """Load a Hugging Face LLaVA model for candidate log-likelihood scoring."""

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
                "LlavaCandidateScorer requires working torch, transformers, and Pillow in the GPU environment."
            ) from exc

        self._torch = torch
        self._Image = Image
        self.model_id = str(model_id)
        self.device = str(device)
        self.compute_dtype = self._resolve_torch_dtype(compute_dtype)
        self.resolver = CocoImageResolver(image_root=image_root, instances_json=instances_json)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        model_classes = [candidate for candidate in (LlavaForConditionalGeneration, AutoModelForVision2Seq) if candidate]
        last_error: Exception | None = None
        model_kwargs = {
            "torch_dtype": self.compute_dtype,
            "trust_remote_code": trust_remote_code,
        }
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:  # pragma: no cover - depends on remote model availability.
                last_error = exc
        else:
            raise RuntimeError(f"Failed to load LLaVA model '{self.model_id}'.") from last_error
        self.model.to(self.device)
        self.model.eval()
        self.decoder_layers = list(find_decoder_layers(self.model))
        self.num_heads, self.hidden_size = self._resolve_head_config()
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError(f"hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}")
        self.head_dim = self.hidden_size // self.num_heads
        self._steering_state: dict[str, Any] | None = None
        self._hooks = []
        self._register_steering_hooks()

    def _resolve_torch_dtype(self, dtype_name: str) -> Any:
        """Resolve a dtype string to a torch dtype."""

        mapping = {
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "float32": self._torch.float32,
        }
        normalized = str(dtype_name).lower()
        if normalized not in mapping:
            raise ValueError(f"Unsupported compute dtype '{dtype_name}'")
        return mapping[normalized]

    def _resolve_head_config(self) -> tuple[int, int]:
        """Resolve attention-head count and hidden size from model config."""

        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None)
        num_heads = getattr(text_config, "num_attention_heads", None) or getattr(config, "num_attention_heads", None)
        hidden_size = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if num_heads is None or hidden_size is None:
            first_attn = getattr(self.decoder_layers[0], "self_attn", None)
            num_heads = num_heads or getattr(first_attn, "num_heads", None)
            hidden_size = hidden_size or getattr(first_attn, "hidden_size", None)
        if num_heads is None or hidden_size is None:
            raise RuntimeError("Could not resolve num_attention_heads and hidden_size for LLaVA scoring")
        return int(num_heads), int(hidden_size)

    def _register_steering_hooks(self) -> None:
        """Register hooks that add selected steering vectors before attention output projection."""

        def make_hook(layer_index: int) -> Any:
            def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
                state = self._steering_state
                if state is None or not inputs:
                    return None
                layer_heads = state["heads_by_layer"].get(layer_index)
                if not layer_heads:
                    return None
                hidden_states = inputs[0]
                start_idx = int(state["start_idx"])
                if start_idx >= hidden_states.shape[1]:
                    return None
                shaped = hidden_states.reshape(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self.num_heads,
                    self.head_dim,
                ).clone()
                vectors = state["vectors"][layer_index].to(device=hidden_states.device, dtype=hidden_states.dtype)
                alpha = float(state["alpha"])
                sign = float(state["sign"])
                for head in layer_heads:
                    shaped[:, start_idx:, head, :] = shaped[:, start_idx:, head, :] + (alpha * sign * vectors[head])
                return (shaped.reshape_as(hidden_states), *inputs[1:])

            return hook

        for layer_index, layer in enumerate(self.decoder_layers):
            self_attn = getattr(layer, "self_attn", None)
            o_proj = getattr(self_attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"Layer {layer_index} does not expose self_attn.o_proj")
            self._hooks.append(o_proj.register_forward_pre_hook(make_hook(layer_index)))

    def _inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Move processor outputs to the scorer device."""

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

    def _nonpad_length(self, inputs: Mapping[str, Any]) -> int:
        """Return the number of non-padding text tokens."""

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return int(inputs["input_ids"].shape[1])
        return int(attention_mask[0].sum().item())

    def _score_response(
        self,
        image_path: str,
        question: str,
        response: str,
        *,
        alpha: float,
        sign: float,
        selected_heads: list[tuple[int, int]] | None,
        steering_vectors: Any | None,
    ) -> float:
        """Return average answer-token log probability for one response candidate."""

        if not str(response).strip():
            raise ValueError("response must be non-empty for candidate scoring")
        image = self._Image.open(image_path).convert("RGB")
        prefix_prompt = build_llava_prefix_prompt(question)
        full_prompt = build_llava_full_prompt(question, response)
        prefix_inputs = self.processor(text=prefix_prompt, images=image, return_tensors="pt")
        full_inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        prefix_len = self._nonpad_length(prefix_inputs)
        full_inputs = self._inputs_to_device(full_inputs)
        target_start = max(prefix_len, 1)
        target_end = self._nonpad_length(full_inputs)
        if target_start >= target_end:
            raise RuntimeError("Could not identify answer tokens for candidate scoring")

        heads_by_layer: dict[int, list[int]] = {}
        for layer, head in selected_heads or []:
            heads_by_layer.setdefault(int(layer), []).append(int(head))
        if steering_vectors is not None and selected_heads and float(alpha) != 0.0:
            self._steering_state = {
                "vectors": steering_vectors,
                "heads_by_layer": heads_by_layer,
                "start_idx": target_start - 1,
                "alpha": float(alpha),
                "sign": float(sign),
            }
        else:
            self._steering_state = None

        with self._torch.inference_mode():
            outputs = self.model(**full_inputs, use_cache=False)
        self._steering_state = None

        logits = outputs.logits.float()
        labels = full_inputs["input_ids"][0]
        log_probs = self._torch.nn.functional.log_softmax(logits[0], dim=-1)
        token_log_probs = []
        for token_index in range(target_start, target_end):
            token_log_probs.append(log_probs[token_index - 1, labels[token_index]])
        return float(self._torch.stack(token_log_probs).mean().item())

    def score_pair(
        self,
        pair: Mapping[str, Any],
        *,
        alpha: float = 0.0,
        sign: float = 1.0,
        selected_heads: list[tuple[int, int]] | None = None,
        steering_vectors: Any | None = None,
    ) -> dict[str, float]:
        """Score positive and negative responses for one pair."""

        image_path = self.resolver.resolve(pair["image_id"])
        question = str(pair["question"])
        return {
            "score_pos": self._score_response(
                image_path,
                question,
                str(pair["response_pos"]),
                alpha=alpha,
                sign=sign,
                selected_heads=selected_heads,
                steering_vectors=steering_vectors,
            ),
            "score_neg": self._score_response(
                image_path,
                question,
                str(pair["response_neg"]),
                alpha=alpha,
                sign=sign,
                selected_heads=selected_heads,
                steering_vectors=steering_vectors,
            ),
        }
