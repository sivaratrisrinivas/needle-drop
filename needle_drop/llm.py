"""OpenAI-compatible chat client with JSON-schema structured output.

Works against the bundled local llama.cpp server (default, free) or any hosted
OpenAI-compatible endpoint: set NEEDLE_LLM_BASE_URL / NEEDLE_LLM_API_KEY / NEEDLE_LLM_MODEL.
Output is validated with pydantic and retried once with the validation error fed back.
"""
from __future__ import annotations

import json
import os
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from . import config
from .tracing import span

T = TypeVar("T", bound=BaseModel)

# llama.cpp's python server takes {"type":"json_object","schema":...}; OpenAI/Groq/Gemini
# take {"type":"json_schema","json_schema":{...}}. Pick with NEEDLE_LLM_JSON_MODE.
JSON_MODE = os.getenv("NEEDLE_LLM_JSON_MODE", "llamacpp")


class LLMError(RuntimeError):
    pass


_client = httpx.Client(timeout=config.LLM_TIMEOUT)


_last_ok = 0.0


def available() -> bool:
    """Is the LLM server up? A busy single-slot llama.cpp server can stall /models while it
    generates, so a recent success is remembered instead of silently falling back to heuristics."""
    global _last_ok
    if config.LLM_PROVIDER == "heuristic":
        return False
    if time.time() - _last_ok < 300:
        return True
    try:
        r = _client.get(config.LLM_BASE_URL.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, timeout=20)
        if r.status_code == 200:
            _last_ok = time.time()
            return True
        return False
    except httpx.HTTPError:
        return False


def chat(messages: list[dict], *, schema: dict | None = None, max_tokens: int = 600,
         temperature: float = 0.2, name: str = "chat", images: list[str] | None = None) -> tuple[str, dict]:
    body: dict = {"model": config.LLM_MODEL, "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature}
    if schema is not None:
        if JSON_MODE == "llamacpp":
            body["response_format"] = {"type": "json_object", "schema": schema}
        else:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": name, "schema": schema, "strict": False}}
    with span(f"llm.{name}", model=config.LLM_MODEL) as s:
        try:
            r = _client.post(config.LLM_BASE_URL.rstrip("/") + "/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {config.LLM_API_KEY}"})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(str(e)) from e
        data = r.json()
        u = data.get("usage") or {}
        s.usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), config.LLM_MODEL)
        text = data["choices"][0]["message"].get("content") or ""
        s.set(prompt=messages[-1]["content"][:1500] if isinstance(messages[-1]["content"], str) else "[multimodal]",
              output=text[:3000])
        return text, u


def structured(model: type[T], messages: list[dict], *, name: str, max_tokens: int = 600,
               temperature: float = 0.2, schema: dict | None = None) -> T:
    """Call the LLM constrained to `schema` (default: the pydantic model's schema) and validate."""
    schema = schema or model.model_json_schema()
    last_err = None
    msgs = list(messages)
    for attempt in range(2):
        text, _ = chat(msgs, schema=schema, max_tokens=max_tokens, temperature=temperature, name=name)
        try:
            return model.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content": f"That JSON was invalid: {str(e)[:500]}. Return corrected JSON only."}]
    raise LLMError(f"structured output failed for {name}: {last_err}")
