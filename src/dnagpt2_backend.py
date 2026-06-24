"""Pinned Hugging Face DNAGPT2 token-probability backend.

Designed from ``GPTProbabilityModel.py`` and ``utils/hub.py`` in
ML-Bioinfo-CEITEC/llm_and_compression commit
2947047e905998d45eaab340bac108d95f14aebd (Apache-2.0). This implementation
uses independent autoregressive decoding, pinned revisions, tokenizer hashing,
float32 inference, and no ``trust_remote_code``.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.llm_residual_codec import TokenProbabilityBackend

DEFAULT_MODEL_REPOSITORY = "vojtam/DNAGPT2_32"
DEFAULT_MODEL_REVISION = "cc28f01babc5271d96c65499682868e1fe40baaa"
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")


def build_integer_cdf(probabilities: list[float], scale_factor: int) -> list[int]:
    """Quantize probabilities into non-zero, strictly increasing intervals."""
    if not probabilities or any(probability < 0 for probability in probabilities):
        raise ValueError("probability distribution must be non-empty and non-negative")
    total_probability = sum(probabilities)
    if total_probability <= 0:
        raise ValueError("probability distribution must have positive mass")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    cumulative = 0.0
    previous = 0
    cdf: list[int] = []
    for probability in probabilities:
        cumulative += probability / total_probability
        quantized = round(cumulative * scale_factor)
        value = max(quantized, previous + 1)
        cdf.append(value)
        previous = value
    return cdf


def trim_past_key_values(
    past_key_values: tuple[tuple[object, object], ...] | None,
    max_length: int,
) -> tuple[tuple[object, object], ...] | None:
    """Keep only the most recent cached positions for exact sliding-window reuse."""
    if past_key_values is None:
        return None
    if max_length <= 0:
        return None
    trimmed: list[tuple[object, object]] = []
    for key, value in past_key_values:
        trimmed.append((key[:, :, -max_length:, :], value[:, :, -max_length:, :]))
    return tuple(trimmed)


class HuggingFaceDNAGPT2Backend(TokenProbabilityBackend):
    """Autoregressive DNAGPT2 backend intended first for CPU reproducibility."""

    def __init__(
        self,
        model_repository: str = DEFAULT_MODEL_REPOSITORY,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        device: str = "cpu",
        context_length: int = 1024,
        cdf_scale: int = 32768,
        stride: int | None = None,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "DNAGPT2 support requires: uv sync"
            ) from error

        self._torch = torch
        self._transformers_version = transformers.__version__
        self.model_repository = model_repository
        self.revision = revision
        self.device = device
        self.context_length = context_length
        self.cdf_scale = cdf_scale
        self.stride = stride or context_length
        self._model_source = self._resolve_model_source()
        self._model_load_kwargs = {
            "trust_remote_code": False,
            "local_files_only": isinstance(self._model_source, Path),
        }
        if not isinstance(self._model_source, Path):
            self._model_load_kwargs["revision"] = revision
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_source,
            **self._model_load_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_source,
            **self._model_load_kwargs,
        ).to(device)
        self.model.eval()
        model_limit = getattr(self.model.config, "n_positions", context_length)
        if context_length <= 0 or context_length > model_limit:
            raise ValueError("context_length exceeds the model limit")
        if self.stride <= 0 or self.stride > self.context_length:
            raise ValueError("stride must be between 1 and context_length")
        self.vocab_size = self.tokenizer.vocab_size
        vocabulary = json.dumps(
            self.tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._tokenizer_sha256 = hashlib.sha256(vocabulary).hexdigest()

    def _resolve_model_source(self) -> str | Path:
        explicit_local_path = Path(self.model_repository)
        if _is_complete_model_directory(explicit_local_path):
            return explicit_local_path
        if self.model_repository == DEFAULT_MODEL_REPOSITORY:
            default_local_model_path = Path("model")
            if _is_complete_model_directory(default_local_model_path):
                return default_local_model_path
        repository_cache_name = self.model_repository.replace("/", "--")
        snapshot_relpath = Path(
            "hub",
            f"models--{repository_cache_name}",
            "snapshots",
            self.revision,
        )
        cache_roots = [
            Path(".hf-cache"),
            Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else None,
            Path.home() / ".cache" / "huggingface",
        ]
        for cache_root in cache_roots:
            if cache_root is None:
                continue
            snapshot_path = cache_root / snapshot_relpath
            if _is_complete_model_directory(snapshot_path):
                return snapshot_path
        return self.model_repository

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_repository": self.model_repository,
            "revision": self.revision,
            "tokenizer_sha256": self._tokenizer_sha256,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "cdf_scale": self.cdf_scale,
            "stride": self.stride,
            "dtype": "float32",
            "device_policy": self.device,
            "torch_version": self._torch.__version__,
            "transformers_version": self._transformers_version,
        }

    def tokenize(self, sequence: str) -> list[int]:
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        if self.detokenize(token_ids) != sequence:
            raise ValueError("DNAGPT2 tokenizer does not round-trip the residual")
        return token_ids

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def cdf_for_prefix(self, token_ids: list[int]) -> list[int]:
        if not token_ids:
            return build_integer_cdf([1.0] * self.vocab_size, self.cdf_scale)
        context = token_ids[-self.context_length :]
        input_ids = self._torch.tensor([context], dtype=self._torch.long, device=self.device)
        with self._torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0, -1, : self.vocab_size]
            probabilities = self._torch.softmax(
                logits.to(dtype=self._torch.float32), dim=-1
            ).cpu().tolist()
        return build_integer_cdf(probabilities, self.cdf_scale)

    def cdfs_for_token_sequence(self, token_ids: list[int]) -> list[list[int]]:
        if not token_ids:
            return []
        cdfs = [build_integer_cdf([1.0] * self.vocab_size, self.cdf_scale)]
        if len(token_ids) == 1:
            return cdfs
        if len(token_ids) > self.context_length:
            return [self.cdf_for_prefix(token_ids[:index]) for index in range(len(token_ids))]
        input_ids = self._torch.tensor(
            [token_ids[:-1]], dtype=self._torch.long, device=self.device
        )
        with self._torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0, :, : self.vocab_size]
            probabilities = self._torch.softmax(
                logits.to(dtype=self._torch.float32), dim=-1
            ).cpu().tolist()
        cdfs.extend(
            build_integer_cdf(distribution, self.cdf_scale)
            for distribution in probabilities
        )
        return cdfs


def _is_complete_model_directory(path: Path) -> bool:
    return path.is_dir() and all((path / filename).exists() for filename in REQUIRED_MODEL_FILES)
