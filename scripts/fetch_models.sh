#!/usr/bin/env bash
# Free, open-weight models, all pulled from Docker Hub's ai/ namespace (no accounts or keys).
set -euo pipefail
M=${NEEDLE_MODELS:-models}
mkdir -p "$M"
[ -f "$M/qwen2.5-3b-q4km.gguf" ] || python scripts/pull_docker_model.py ai/qwen2.5 3B-Q4_K_M "$M/qwen2.5-3b-q4km.gguf"
[ -f "$M/nomic-embed-v1.5-f16.gguf" ] || python scripts/pull_docker_model.py ai/nomic-embed-text-v1.5 137M-F16 "$M/nomic-embed-v1.5-f16.gguf"
[ -f "$M/smolvlm-500m-q8.gguf" ] || python scripts/pull_docker_model.py ai/smolvlm 500M-Q8_0 "$M/smolvlm-500m-q8.gguf" "$M/smolvlm-500m-mmproj.gguf"
