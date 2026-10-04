"""Serve a local MedGemma model with an OpenAI-compatible chat endpoint.

The patient view uses MedGemma to turn everyday descriptions of symptoms into
medical terms and to explain conditions in plain language.  This service loads
the model once with Hugging Face Transformers (Apple GPU, CUDA or CPU) and
answers ``POST /v1/chat/completions`` like llama-server or vLLM, so either of
those can replace it without changing api/server.py.

Usage (from the repository root):
    python frontend/api/medgemma_server.py --model /path/to/medgemma-4b-text
    # or set MEDGEMMA_MODEL_PATH; listens on http://127.0.0.1:8766/v1

Use a text-only checkpoint (``Gemma3ForCausalLM``), e.g. medgemma-4b-text or
google/medgemma-27b-text-it.  Nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_NEW_TOKENS = 1024
MAX_BODY = 200_000


class MedGemma:
    def __init__(self, path: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.device = device
        self.name = os.path.basename(os.path.normpath(path))
        self.lock = threading.Lock()
        self.torch = torch
        print(f"Loaded {self.name} on {device} in {time.time() - started:.0f}s", flush=True)

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, int, int]:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        options: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            options["temperature"] = temperature
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        new = output[0, prompt_tokens:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), prompt_tokens, int(new.shape[0])


class Handler(BaseHTTPRequestHandler):
    model: MedGemma

    def _json(self, status: int, document: Any) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            return self._json(HTTPStatus.OK, {"object": "list", "data": [{"id": self.model.name, "object": "model"}]})
        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Unknown endpoint"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Unknown endpoint"}})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": {"message": "Request too large"}})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in body["messages"]]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": f"Invalid request: {error}"}})
        max_tokens = max(1, min(int(body.get("max_tokens") or 512), MAX_NEW_TOKENS))
        temperature = float(body.get("temperature") or 0.0)
        started = time.time()
        text, prompt_tokens, completion_tokens = self.model.chat(messages, max_tokens, temperature)
        print(f"{completion_tokens} tokens in {time.time() - started:.1f}s", flush=True)
        self._json(HTTPStatus.OK, {
            "object": "chat.completion",
            "model": self.model.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        })

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("MEDGEMMA_MODEL_PATH"), help="Path or Hugging Face ID of a text MedGemma checkpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()
    if not args.model:
        parser.error("give --model or set MEDGEMMA_MODEL_PATH")
    Handler.model = MedGemma(args.model, args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MedGemma ready at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
