#!/usr/bin/env bash
# Start the local LLM server (OpenAI-compatible) and the Needle Drop API + UI.
set -euo pipefail
M=${NEEDLE_MODELS:-models}
export NEEDLE_EMBED_GGUF=${NEEDLE_EMBED_GGUF:-$M/nomic-embed-v1.5-f16.gguf}
export NEEDLE_VLM_GGUF=${NEEDLE_VLM_GGUF:-$M/smolvlm-500m-q8.gguf}
export NEEDLE_VLM_MMPROJ=${NEEDLE_VLM_MMPROJ:-$M/smolvlm-500m-mmproj.gguf}
if [ "${NEEDLE_LLM:-openai_compat}" != "heuristic" ] && [ -z "${NEEDLE_LLM_EXTERNAL:-}" ]; then
  python -m llama_cpp.server --model "$M/qwen2.5-3b-q4km.gguf" --model_alias qwen2.5-3b-instruct-q4_k_m \
    --port 8081 --n_ctx 8192 --n_threads "${LLM_THREADS:-$(nproc)}" > /tmp/llm.log 2>&1 &
fi
exec uvicorn needle_drop.api:app --host 0.0.0.0 --port "${PORT:-8000}"
