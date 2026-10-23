#!/usr/bin/env bash
# Train the training conditions on one GPU, one after another (completed arms are skipped).
#   ast_any = generic AST-subtree infilling baseline (needs a FIM directory built by the current prepare.py)
#   MODEL=/path/Qwen2.5-32B DATA=runs/fim SEEDS="56 123 2024" bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL:?set MODEL to local Qwen2.5-32B Base weights}"
: "${DATA:?set DATA to the frozen FIM directory (scripts/prepare_data.sh)}"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-runs/train}"
SEEDS="${SEEDS:-56}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
"$PYTHON" -m aspire.training.preflight --model "$MODEL" --max-gpu-gb "${MAX_GPU_GB:-0}"
for seed in $SEEDS; do
  for arm in ${ARMS:-plain random_fim ast_fim ast_any}; do
    dest="$OUT/seed$seed/$arm"
    if [[ -f "$dest/COMPLETE.json" ]]; then echo "skip $dest (complete)"; continue; fi
    mkdir -p "$(dirname "$dest")"
    "$PYTHON" -u -m aspire.training.train --model "$MODEL" \
      --train "$DATA/$arm.jsonl" --validation "$DATA/validation.jsonl" --output "$dest" --seed "$seed" \
      --max-gpu-gb "${MAX_GPU_GB:-0}" --batch-size 1 --gradient-accumulation 16 --learning-rate "${LR:-2e-5}" \
      --scheduler cosine --epochs 1 --rank 32 --alpha 64 --dropout 0.05 ${EXTRA:-} 2>&1 | tee "$dest.log"
  done
done
