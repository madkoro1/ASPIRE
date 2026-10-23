#!/usr/bin/env bash
# Generate the dataset, freeze the evaluation suite and build the paired FIM training sets.
#   TOKENIZER=/path/Qwen2.5-32B bash scripts/prepare_data.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
: "${TOKENIZER:?set TOKENIZER to the Qwen2.5-32B Base directory (tokenizer files)}"
RAW="${RAW:-runs/data/raw}"
SUITE="${SUITE:-runs/data/suite}"
FIM="${FIM:-runs/data/fim}"
"$PYTHON" -m aspire.data.generate --output-dir "$RAW" --train-size "${TRAIN_SIZE:-10000}" \
  --val-size "${VAL_SIZE:-1000}" --test-size-per-task "${TEST_PER_TASK:-50}" --seed "${DATA_SEED:-20260713}"
"$PYTHON" -m aspire.data.freeze --iid "$RAW/test_compositional_track_a_iid.csv" \
  --ood "$RAW/test_compositional_track_a_ood.csv" --output "$SUITE" --families paper \
  --feedback-contexts 3 --clingo-timeout 20
"$PYTHON" -m aspire.data.overlap --train "$RAW/train_compositional.csv" \
  --test "$RAW/test_compositional_track_a_iid.csv" --output "$RAW/overlap.json"
"$PYTHON" -m aspire.training.prepare --train "$RAW/train_compositional.csv" \
  --validation "$RAW/val_compositional.csv" --output "$FIM" --tokenizer "$TOKENIZER" \
  --rate 0.2 --seed 20260924 --seq-length 1024 --attempts 512
echo "suite: $SUITE   fim: $FIM"
