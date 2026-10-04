"""Minimal OpenAI-compatible client for a locally served MedGemma model."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "google/medgemma-27b-text-it"
DEFAULT_CACHE = Path(".data") / "v2_5" / "medgemma_cache"


class MedGemmaError(RuntimeError):
    """Base error for endpoint, protocol, or model-output failures."""


class EndpointUnavailable(MedGemmaError):
    """The configured endpoint could not be reached."""


class InvalidModelResponse(MedGemmaError):
    """The endpoint responded, but not with usable structured output."""


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    model_revision: str = ""
    api_key: str = ""
    timeout_seconds: float = 180.0
    max_output_tokens: int = 4_096
    response_mode: str = "json_schema"
    allow_remote: bool = False
    cache_dir: Path = DEFAULT_CACHE
    retries: int = 2

    @classmethod
    def from_environment(cls, **overrides: Any) -> "ClientConfig":
        values: dict[str, Any] = {
            "base_url": os.environ.get("MEDGEMMA_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
            "model_revision": os.environ.get("MEDGEMMA_MODEL_REVISION", ""),
            "api_key": os.environ.get("MEDGEMMA_API_KEY", ""),
            "response_mode": os.environ.get("MEDGEMMA_RESPONSE_MODE", "json_schema"),
            "cache_dir": Path(os.environ.get("MEDGEMMA_CACHE_DIR", str(DEFAULT_CACHE))),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


@dataclass
class Completion:
    document: dict[str, Any]
    model: str
    model_revision: str
    response_sha256: str
    cached: bool
    usage: dict[str, Any]
    cache_key: str


class JsonCompletionClient(Protocol):
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        prompt_version: str,
        use_cache: bool = True,
    ) -> Completion: ...


def is_local_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def extract_json_document(content: str) -> dict[str, Any]:
    """Parse JSON even when a server wraps it in fences or reasoning text."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise InvalidModelResponse("MedGemma response did not contain a JSON object")


class MedGemmaClient:
    def __init__(self, config: ClientConfig):
        if config.response_mode not in {"json_schema", "json_object", "prompt_only"}:
            raise ValueError("response_mode must be json_schema, json_object, or prompt_only")
        if not config.allow_remote and not is_local_url(config.base_url):
            raise ValueError(
                "Refusing to send paper text to a non-local endpoint. "
                "Use --allow-remote only after reviewing the data policy."
            )
        self.config = config

    @property
    def chat_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return f"{base}/models"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _request(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="GET" if payload is None else "POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise MedGemmaError(f"MedGemma endpoint returned HTTP {error.code}: {detail[:1000]}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise EndpointUnavailable(f"Cannot reach MedGemma endpoint at {url}: {error}") from error
        except json.JSONDecodeError as error:
            raise InvalidModelResponse(f"Endpoint returned invalid JSON: {error}") from error

    def available_models(self) -> list[str]:
        response = self._request(self.models_url)
        return [
            str(item.get("id"))
            for item in response.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        prompt_version: str,
    ) -> str:
        material = {
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "base_url": self.config.base_url,
            "response_mode": self.config.response_mode,
            "prompt_version": prompt_version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / key[:2] / f"{key}.json"

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

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "seed": 0,
        }
        if self.config.response_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rare_disease_paper_extraction",
                    "strict": True,
                    "schema": schema,
                },
            }
        elif self.config.response_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}

        response: dict[str, Any] | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self._request(self.chat_url, payload)
                break
            except EndpointUnavailable:
                if attempt >= self.config.retries:
                    raise
                time.sleep(2**attempt)
        assert response is not None
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            content = str(content)
        except (KeyError, IndexError, TypeError) as error:
            raise InvalidModelResponse("Endpoint response lacks choices[0].message.content") from error
        document = extract_json_document(content)
        response_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        completion = Completion(
            document=document,
            model=str(response.get("model") or self.config.model),
            model_revision=self.config.model_revision,
            response_sha256=response_hash,
            cached=False,
            usage=dict(response.get("usage") or {}),
            cache_key=key,
        )
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        **asdict(completion),
                        "cached": False,
                        "created_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        return completion

