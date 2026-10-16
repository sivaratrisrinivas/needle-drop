"""Central configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("NEEDLE_DATA", ROOT / "data"))
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = DATA / "catalog.jsonl"
VECTORS_PATH = DATA / "catalog_vectors.npy"
LICENSE_DOCS = DATA / "licenses"
MODELS = Path(os.getenv("NEEDLE_MODELS", ROOT / "models"))
DB_PATH = Path(os.getenv("NEEDLE_DB", DATA / "needle.sqlite"))

# LLM provider. Any OpenAI-compatible endpoint works:
#   local llama.cpp server (default), Groq, Gemini (OpenAI-compat), OpenRouter, Ollama...
# NEEDLE_LLM=heuristic disables the LLM entirely (embedding zero-shot parser).
LLM_PROVIDER = os.getenv("NEEDLE_LLM", "openai_compat")
LLM_BASE_URL = os.getenv("NEEDLE_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
LLM_API_KEY = os.getenv("NEEDLE_LLM_API_KEY", "local")
LLM_MODEL = os.getenv("NEEDLE_LLM_MODEL", "qwen2.5-3b-instruct-q4_k_m")
LLM_TIMEOUT = float(os.getenv("NEEDLE_LLM_TIMEOUT", "180"))

# USD per 1M tokens (input, output). Local inference costs $0 in API spend; the
# table lets you see what the same run would cost on a hosted provider.
PRICES = {
    "qwen2.5-3b-instruct-q4_k_m": (0.0, 0.0),
    "local": (0.0, 0.0),
}
# Optional "what if this ran on a hosted model" comparison, set by the operator.
SHADOW_PRICE = tuple(float(x) for x in os.getenv("NEEDLE_SHADOW_PRICE", "0.10,0.40").split(","))

JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "")
WORKERS = int(os.getenv("NEEDLE_WORKERS", "1"))
