"""Lazy in-process MedGemma backend using Hugging Face Transformers.

Importing this module does not import torch/transformers, inspect CUDA, download
weights, or allocate GPU memory. All heavyweight work starts on the first
``complete_json`` call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .client import (
    DEFAULT_CACHE,
    Completion,
    InvalidModelResponse,
    MedGemmaError,
    extract_json_document,
)


DEFAULT_TRANSFORMERS_MODEL = "google/medgemma-1.5-4b-it"


@dataclass(frozen=True)
class TransformersConfig:
    model: str = DEFAULT_TRANSFORMERS_MODEL
    model_revision: str = ""
    hf_token: str = ""
    quantization: str = "4bit"
    compute_dtype: str = "auto"
    max_input_tokens: int = 8_192
    max_output_tokens: int = 2_048
    model_cache_dir: Path | None = None
    response_cache_dir: Path = DEFAULT_CACHE / "transformers"
    local_files_only: bool = False

    @classmethod
    def from_environment(cls, **overrides: Any) -> "TransformersConfig":
        model_cache = os.environ.get("MEDGEMMA_MODEL_CACHE")
        values: dict[str, Any] = {
            "model": os.environ.get("MEDGEMMA_MODEL", DEFAULT_TRANSFORMERS_MODEL),
            "model_revision": os.environ.get("MEDGEMMA_MODEL_REVISION", ""),
            "hf_token": os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")),
            "quantization": os.environ.get("MEDGEMMA_QUANTIZATION", "4bit"),
            "compute_dtype": os.environ.get("MEDGEMMA_COMPUTE_DTYPE", "auto"),
            "model_cache_dir": Path(model_cache) if model_cache else None,
            "response_cache_dir": Path(
                os.environ.get(
                    "MEDGEMMA_CACHE_DIR",
                    str(DEFAULT_CACHE / "transformers"),
                )
            ),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


class TransformersMedGemmaClient:
    """Implements the same JSON-completion interface as the endpoint client."""

    def __init__(self, config: TransformersConfig):
        if config.quantization not in {"4bit", "8bit", "bf16", "fp16"}:
            raise ValueError("quantization must be 4bit, 8bit, bf16, or fp16")
        if config.compute_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("compute_dtype must be auto, bfloat16, or float16")
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._loaded_revision = config.model_revision

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _dtype(self, torch: Any) -> Any:
        if self.config.compute_dtype == "bfloat16":
            return torch.bfloat16
        if self.config.compute_dtype == "float16":
            return torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise MedGemmaError(
                "Direct MedGemma requires: pip install torch transformers "
                "accelerate bitsandbytes huggingface_hub"
            ) from error
        if not torch.cuda.is_available():
            raise MedGemmaError(
                "No CUDA GPU is available. Direct MedGemma inference is intentionally "
                "not falling back to CPU because it would be impractically slow."
            )
        dtype = self._dtype(torch)
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.config.local_files_only,
        }
        common_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
        }
        if self.config.model_revision:
            model_kwargs["revision"] = self.config.model_revision
            common_kwargs["revision"] = self.config.model_revision
        if self.config.hf_token:
            model_kwargs["token"] = self.config.hf_token
            common_kwargs["token"] = self.config.hf_token
        if self.config.model_cache_dir is not None:
            model_kwargs["cache_dir"] = str(self.config.model_cache_dir)
            common_kwargs["cache_dir"] = str(self.config.model_cache_dir)
        if self.config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif self.config.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        try:
            processor = AutoProcessor.from_pretrained(self.config.model, **common_kwargs)
            model = AutoModelForImageTextToText.from_pretrained(
                self.config.model,
                **model_kwargs,
            )
        except Exception as error:
            raise MedGemmaError(
                f"Could not load {self.config.model!r}. Confirm that model terms "
                "were accepted, HF_TOKEN is valid, dependencies support CUDA, and "
                f"the selected quantization fits memory. Original error: {error}"
            ) from error
        model.eval()
        self._torch = torch
        self._processor = processor
        self._model = model
        if not self._loaded_revision:
            self._loaded_revision = str(
                getattr(model.config, "_commit_hash", "")
                or getattr(model.config, "name_or_path", "")
            )

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        prompt_version: str,
    ) -> str:
        material = {
            "backend": "transformers",
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "quantization": self.config.quantization,
            "compute_dtype": self.config.compute_dtype,
            "max_input_tokens": self.config.max_input_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "prompt_version": prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.response_cache_dir / key[:2] / f"{key}.json"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        prompt_version: str,
        use_cache: bool = True,
    ) -> Completion:
        key = self._cache_key(system_prompt, user_prompt, schema, prompt_version)
        path = self._cache_path(key)
        if use_cache and path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return Completion(
                document=dict(cached["document"]),
                model=str(cached["model"]),
                model_revision=str(cached.get("model_revision", "")),
                response_sha256=str(cached["response_sha256"]),
                cached=True,
                usage=dict(cached.get("usage", {})),
                cache_key=key,
            )

        self._load()
        torch, processor, model = self._torch, self._processor, self._model
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        combined_prompt = f"""{system_prompt}

TASK
{user_prompt}

Return exactly one JSON object matching this JSON Schema:
{schema_text}
"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": combined_prompt}],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as error:
            raise MedGemmaError(f"MedGemma processor could not format the prompt: {error}") from error
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.config.max_input_tokens:
            raise MedGemmaError(
                f"Formatted prompt has {input_tokens} tokens, above the configured "
                f"limit of {self.config.max_input_tokens}. Reduce --max-prompt-words."
            )
        inputs = inputs.to(model.device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_output_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            output_tokens = generated[0][input_tokens:]
            content = processor.decode(output_tokens, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError as error:
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            raise MedGemmaError(
                "MedGemma ran out of GPU memory. Reduce --max-prompt-words or "
                "--max-output-tokens, or use 4bit quantization."
            ) from error
        except Exception as error:
            raise MedGemmaError(f"MedGemma generation failed: {error}") from error
        elapsed = time.perf_counter() - started
        try:
            document = extract_json_document(content)
        except InvalidModelResponse:
            raise
        response_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        completion = Completion(
            document=document,
            model=self.config.model,
            model_revision=self._loaded_revision,
            response_sha256=response_hash,
            cached=False,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": int(output_tokens.shape[-1]),
                "elapsed_seconds": elapsed,
            },
            cache_key=key,
        )
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({**asdict(completion), "created_at": time.time()}, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        return completion

