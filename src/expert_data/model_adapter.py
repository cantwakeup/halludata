"""Activation-adapter interfaces and mock implementations for offline pilots."""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseActivationAdapter(ABC):
    """Abstract interface for adapters that expose layer-head activation vectors."""

    @abstractmethod
    def encode_pair(
        self,
        image_id: str,
        question: str,
        response: str,
        *,
        pair_id: str,
        subtype: str,
        branch: str,
    ) -> dict[str, Any]:
        """Encode one image-question-response triple into layer-head activation vectors."""


def _stable_seed(text: str) -> int:
    """Hash text deterministically into a 32-bit pseudo-random seed."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _random_vector(seed: int, dim: int) -> list[float]:
    """Generate one deterministic pseudo-random vector of a given dimension."""

    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


class MockActivationAdapter(BaseActivationAdapter):
    """Generate repeatable pseudo-random activations for scaffold and test runs."""

    def __init__(self, num_layers: int = 8, num_heads: int = 8, vector_dim: int = 8) -> None:
        """Configure the mock adapter's layer-head grid and vector dimensionality."""

        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.vector_dim = int(vector_dim)

    def _head_keys(self) -> list[str]:
        """Return all supported layer-head keys in deterministic order."""

        return [f"l{layer}_h{head}" for layer in range(self.num_layers) for head in range(self.num_heads)]

    def _subtype_basis(self, subtype: str) -> list[float]:
        """Build a subtype-specific basis direction that separates branches reproducibly."""

        subtype_to_index = {"cat": 0, "cnt": 1, "col": 2, "rel": 3}
        subtype_index = subtype_to_index.get(str(subtype), 0)
        return [
            0.0 if dimension != (subtype_index % self.vector_dim) else 1.0
            for dimension in range(self.vector_dim)
        ]

    def encode_pair(
        self,
        image_id: str,
        question: str,
        response: str,
        *,
        pair_id: str,
        subtype: str,
        branch: str,
    ) -> dict[str, Any]:
        """Encode one pair branch into deterministic pseudo-random layer-head vectors."""

        branch_sign = 1.0 if str(branch) == "pos" else -1.0
        basis = self._subtype_basis(subtype)
        layer_head_vectors: dict[str, list[float]] = {}
        for head_index, head_key in enumerate(self._head_keys()):
            seed = _stable_seed(f"{pair_id}|{image_id}|{question}|{response}|{head_key}")
            base_vector = _random_vector(seed, self.vector_dim)
            strength = 0.4 + ((head_index % self.num_heads) / max(self.num_heads, 1))
            layer_head_vectors[head_key] = [
                base_value + (branch_sign * strength * basis_value)
                for base_value, basis_value in zip(base_vector, basis)
            ]
        return {"layer_head_vectors": layer_head_vectors}


def build_llava_prompt(question: str, response: str) -> str:
    """Build the teacher-forced LLaVA prompt for one question/answer branch."""

    return f"USER: <image>\n{question}\nASSISTANT: {response}"


def find_decoder_layers(model: Any) -> Any:
    """Find decoder layers in common LLaVA/LLaMA-style Hugging Face model layouts."""

    candidates = [
        ("language_model", "model", "layers"),
        ("model", "layers"),
    ]
    for path in candidates:
        current = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            return current

    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        layers = getattr(decoder, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("Could not locate decoder layers for this model; expected a LLaVA/LLaMA-style layout.")


def _nested_getattr(root: Any, path: tuple[str, ...]) -> Any:
    """Read a nested attribute path, returning None if any segment is missing."""

    current = root
    for attribute in path:
        current = getattr(current, attribute, None)
        if current is None:
            return None
    return current


class LlavaActivationAdapter(BaseActivationAdapter):
    """Extract real LLaVA layer-head activations with teacher-forced forward passes."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda:0",
        compute_dtype: str = "bfloat16",
        storage_dtype: str = "float16",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        """Load a Hugging Face LLaVA model and register lightweight attention hooks."""

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
                "LlavaActivationAdapter requires optional dependencies: torch, transformers, and Pillow. "
                "Install a working GPU environment before running --adapter llava."
            ) from exc

        self._torch = torch
        self._Image = Image
        self.model_id = str(model_id)
        self.device = str(device)
        self.compute_dtype_name = str(compute_dtype)
        self.storage_dtype_name = str(storage_dtype)
        self.compute_dtype = self._resolve_torch_dtype(compute_dtype)
        self.storage_dtype = self._resolve_torch_dtype(storage_dtype)
        self._current_target_idx: int | None = None
        self._layer_outputs: dict[int, Any] = {}
        self._hooks: list[Any] = []

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.compute_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = {"": self.device}

        last_error: Exception | None = None
        model_classes = [candidate for candidate in (LlavaForConditionalGeneration, AutoModelForVision2Seq) if candidate]
        for model_class in model_classes:
            try:
                self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:  # pragma: no cover - depends on remote model availability.
                last_error = exc
        else:
            raise RuntimeError(f"Failed to load LLaVA model '{self.model_id}'.") from last_error

        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.decoder_layers = list(find_decoder_layers(self.model))
        self.num_heads, self.hidden_size = self._resolve_head_config()
        if self.hidden_size % self.num_heads != 0:
            raise RuntimeError(f"hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}.")
        self.head_dim = self.hidden_size // self.num_heads
        self._register_hooks()

    def _resolve_torch_dtype(self, dtype_name: str) -> Any:
        """Resolve a user-facing dtype string to a torch dtype."""

        normalized = str(dtype_name).lower()
        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype '{dtype_name}'.")
        return mapping[normalized]

    def _resolve_head_config(self) -> tuple[int, int]:
        """Resolve number of attention heads and hidden size from model config or attention modules."""

        config = getattr(self.model, "config", None)
        num_heads = _nested_getattr(config, ("text_config", "num_attention_heads"))
        hidden_size = _nested_getattr(config, ("text_config", "hidden_size"))
        if num_heads is None:
            num_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None)
        if num_heads is None and self.decoder_layers:
            num_heads = getattr(getattr(self.decoder_layers[0], "self_attn", None), "num_heads", None)
        if hidden_size is None and self.decoder_layers:
            hidden_size = getattr(getattr(self.decoder_layers[0], "self_attn", None), "hidden_size", None)
        if num_heads is None or hidden_size is None:
            raise RuntimeError("Could not resolve num_attention_heads and hidden_size from the model.")
        return int(num_heads), int(hidden_size)

    def _register_hooks(self) -> None:
        """Register o_proj forward pre-hooks that capture only the target-token activation."""

        def make_hook(layer_index: int) -> Any:
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                if self._current_target_idx is None or not inputs:
                    return
                hidden_states = inputs[0]
                target_hidden = hidden_states[:, self._current_target_idx, :].detach()
                target_hidden = target_hidden.reshape(target_hidden.shape[0], self.num_heads, self.head_dim)
                self._layer_outputs[layer_index] = target_hidden[0].to("cpu").to(self.storage_dtype)

            return hook

        for layer_index, layer in enumerate(self.decoder_layers):
            self_attn = getattr(layer, "self_attn", None)
            o_proj = getattr(self_attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"Layer {layer_index} does not expose self_attn.o_proj for activation hooks.")
            self._hooks.append(o_proj.register_forward_pre_hook(make_hook(layer_index)))

    def _inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Move processor outputs to the adapter device, casting floating tensors where safe."""

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

    def _target_token_index(self, inputs: Mapping[str, Any]) -> int:
        """Return the final non-padding token index for a tokenized teacher-forced prompt."""

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            nonzero_positions = attention_mask[0].nonzero(as_tuple=False).flatten()
            if len(nonzero_positions) == 0:
                raise RuntimeError("Cannot extract activations from an empty tokenized prompt.")
            return int(nonzero_positions[-1].item())
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor output did not contain input_ids or attention_mask.")
        return int(input_ids.shape[1] - 1)

    def _run_branch(self, image_path: str, question: str, response: str) -> tuple[Any, int]:
        """Run one teacher-forced branch and return [layers, heads, head_dim] activations."""

        if not str(response).strip():
            raise ValueError("response must be non-empty for teacher-forced activation extraction.")
        image = self._Image.open(image_path).convert("RGB")
        prompt = build_llava_prompt(question, response)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = self._inputs_to_device(inputs)
        target_idx = self._target_token_index(inputs)
        self._current_target_idx = target_idx
        self._layer_outputs = {}
        with self._torch.inference_mode():
            self.model(**inputs, use_cache=False)
        self._current_target_idx = None

        missing_layers = [index for index in range(len(self.decoder_layers)) if index not in self._layer_outputs]
        if missing_layers:
            raise RuntimeError(f"Missing hooked activations for decoder layers: {missing_layers[:5]}")
        stacked = self._torch.stack([self._layer_outputs[index] for index in range(len(self.decoder_layers))], dim=0)
        return stacked.to("cpu").to(self.storage_dtype), target_idx

    def encode_pair(
        self,
        image_path: str,
        question: str,
        response_pos: str,
        response_neg: str,
        pair_id: str | None = None,
        subtype: str | None = None,
    ) -> dict[str, Any]:
        """Encode positive and negative branches into CPU layer-head activation tensors."""

        del pair_id, subtype
        z_pos, target_pos = self._run_branch(image_path=image_path, question=question, response=response_pos)
        z_neg, target_neg = self._run_branch(image_path=image_path, question=question, response=response_neg)
        return {
            "z_pos": z_pos,
            "z_neg": z_neg,
            "meta": {
                "num_layers": int(z_pos.shape[0]),
                "num_heads": int(z_pos.shape[1]),
                "head_dim": int(z_pos.shape[2]),
                "target_token_index_pos": int(target_pos),
                "target_token_index_neg": int(target_neg),
            },
        }


def load_activation_adapter(name: str, **kwargs: Any) -> BaseActivationAdapter:
    """Load one supported activation adapter by its short name."""

    normalized_name = str(name).strip().lower()
    if normalized_name == "mock":
        return MockActivationAdapter(**kwargs)
    if normalized_name == "llava":
        return LlavaActivationAdapter(**kwargs)
    if normalized_name == "custom":
        raise ValueError("Custom activation adapters are scaffold-only for now; implement BaseActivationAdapter first.")
    raise ValueError(f"Unsupported activation adapter '{name}'")


def flatten_layer_head_vectors(activation_dict: dict[str, Any]) -> list[float]:
    """Flatten sorted layer-head vectors into one concatenated feature vector."""

    layer_head_vectors = dict(activation_dict.get("layer_head_vectors", {}))
    flattened: list[float] = []
    for head_key in sorted(layer_head_vectors):
        flattened.extend(float(value) for value in layer_head_vectors[head_key])
    return flattened

