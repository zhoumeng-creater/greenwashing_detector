#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m src.data.build_rule_corpus --config configs/default.yaml
python -m src.data.build_eval_claims --config configs/default.yaml
python -m src.index.build_faiss --config configs/default.yaml
python -m src.index.inspect_index --config configs/default.yaml

echo "Bootstrap complete."
