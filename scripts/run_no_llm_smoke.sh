#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src.generation.analyze_claim \
  --config configs/default.yaml \
  --claim "Our bottle is 100% eco-friendly and carbon neutral." \
  --product-category "packaging" \
  --jurisdiction "general" \
  --no-llm \
  --output outputs/demo_samples/no_llm_smoke.jsonl
