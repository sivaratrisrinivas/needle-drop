#!/usr/bin/env bash
# MTG-Jamendo metadata + per-track licenses (metadata: CC BY-NC-SA 4.0, (c) MTG/UPF).
set -euo pipefail
mkdir -p data/raw && cd data/raw
B=https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master
for f in data/raw.meta.tsv data/raw_30s_cleantags.tsv audio_licenses.txt; do curl -sSfLO "$B/$f"; done
