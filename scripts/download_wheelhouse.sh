#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="$ROOT/offline_bundle/wheelhouse"
mkdir -p "$WHEELHOUSE"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

python -m pip download \
  -i "$PIP_INDEX_URL" \
  -r "$ROOT/requirements.txt" \
  -d "$WHEELHOUSE"

echo "Wheelhouse written to $WHEELHOUSE"
