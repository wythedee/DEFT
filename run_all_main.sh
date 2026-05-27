#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODE:=both}"
: "${SEEDS:=0 1 2}"
: "${GPU:=0}"
: "${DATA_ROOT:?Set DATA_ROOT to the root directory containing the LMDB datasets}"
: "${OUTPUT_ROOT:=$ROOT_DIR/outputs/main}"

export MODE SEEDS GPU DATA_ROOT OUTPUT_ROOT

BACKBONES="${BACKBONES:-Codebrain CBraMod CSBrain LaBraM}"

for backbone in $BACKBONES; do
  echo "[DEFT] Running ${backbone} main experiments"
  bash "$ROOT_DIR/$backbone/scripts/train_main.sh"
done

echo "[DEFT] All requested main experiments finished."

