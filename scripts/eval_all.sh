#!/usr/bin/env bash
# All generation/refinement arms of Tables 2-6 for one trained seed.
#   MODEL=/path/Qwen2.5-32B SUITE=runs/data/suite TRAIN=runs/train SEED=56 bash scripts/eval_all.sh
# Optional API rows (OpenAI-compatible /chat/completions, key in $ASPIRE_API_KEY):
#   ENDPOINT=<url> FEWSHOT_API_MODEL=glm-5.2            Table 2, instruction-model few-shot row
#   ENDPOINT=<url> FEEDBACK_API_MODEL=Qwen2.5-32B-Instruct  optional Instruct feedback-model run (not in the paper tables)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL:?}" "${SUITE:?}" "${TRAIN:?}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-56}"
SPLIT="${SPLIT:-iid}"
OUT="${OUT:-runs/eval/seed$SEED/$SPLIT}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
COMMON=(--suite "$SUITE" --split "$SPLIT" --seed "$SEED" --max-tokens 512 --attempts 3 --clingo-timeout 20)
LOCAL=(--model "$MODEL" --max-gpu-gb "${MAX_GPU_GB:-0}")

run() { local dest="$1"; shift; [[ -f "$dest/COMPLETE.json" ]] && { echo "skip $dest"; return; }
        "$PYTHON" -u -m aspire.eval.run "${COMMON[@]}" --output "$dest" "$@" 2>&1 | tail -n 2; }

# Table 2: few-shot generation by the base model (no adapter).
run "$OUT/fewshot_base" --method fewshot --backend local "${LOCAL[@]}" --generator-style raw
# Table 2: instruction-model few-shot generation through an API (optional).
if [[ -n "${ENDPOINT:-}" && -n "${FEWSHOT_API_MODEL:-}" ]]; then
  run "$OUT/fewshot_api" --method fewshot --backend api --endpoint "$ENDPOINT" \
      --model "$FEWSHOT_API_MODEL" --generator-style chat
fi

# Tables 2, 3, 6: initial generation and ASPIRE refinement for each training condition
# (ast_any = generic AST-subtree infilling baseline).
for arm in ${ARMS:-plain random_fim ast_fim ast_any}; do
  ad="$TRAIN/seed$SEED/$arm/adapter"
  [[ -d "$ad" ]] || { echo "skip $arm: no adapter at $ad"; continue; }
  run "$OUT/$arm/base"   --method base   --backend local "${LOCAL[@]}" --adapter "$ad"
  run "$OUT/$arm/aspire" --method aspire --backend local "${LOCAL[@]}" --adapter "$ad" --initial "$OUT/$arm/base"
done
# Table 3: grammar-constrained decoding on the plain-SFT model (needs transformers-cfg).
PA="$TRAIN/seed$SEED/plain/adapter"
run "$OUT/plain_gcd/base"   --method base   --backend local "${LOCAL[@]}" --adapter "$PA" --grammar asp
run "$OUT/plain_gcd/aspire" --method aspire --backend local "${LOCAL[@]}" --adapter "$PA" --grammar asp \
    --initial "$OUT/plain_gcd/base"

AD="$TRAIN/seed$SEED/ast_fim/adapter"
INIT="$OUT/ast_fim/base"
G=(--backend local "${LOCAL[@]}" --adapter "$AD" --initial "$INIT")
# Table 6: ablations on the ASP-FIM model (code arm ast_fim) and its initial candidates.
run "$OUT/ast_fim/syntax_only"       --method syntax_only "${G[@]}"
run "$OUT/ast_fim/instance_only"     --method aspire --visible instance "${G[@]}"
run "$OUT/ast_fim/random_context"    --method aspire --selection random "${G[@]}"
run "$OUT/ast_fim/no_projection"     --method aspire --verify-projection off "${G[@]}"
run "$OUT/ast_fim/optimal_only"      --method aspire --verify-phases optimal "${G[@]}"
# Counterexample facts also shown to the generator: isolates withholding them (supplementary, not in Table 5).
run "$OUT/ast_fim/aspire_facts"      --method aspire_facts "${G[@]}"
# Table 5: feedback mechanisms under the same generator, contexts, budget and best-candidate rule.
run "$OUT/ast_fim/template"          --method template   "${G[@]}"
run "$OUT/ast_fim/self_debug"        --method self_debug "${G[@]}"
run "$OUT/ast_fim/pgs"               --method pgs        "${G[@]}"
run "$OUT/ast_fim/llm_asp"           --method llm_asp    "${G[@]}"
run "$OUT/ast_fim/aspire_keep_last"  --method aspire --keep last "${G[@]}"
# Optional: Instruct feedback model through an API, local generator (optional).
if [[ -n "${ENDPOINT:-}" && -n "${FEEDBACK_API_MODEL:-}" ]]; then
  run "$OUT/ast_fim/aspire_instruct_fb" --method aspire "${G[@]}" --feedback-backend api \
      --endpoint "$ENDPOINT" --feedback-model "$FEEDBACK_API_MODEL" --feedback-style chat
fi

# Paired McNemar / bootstrap CI (overall and per group) against the initial
# candidates and against the full method.
"$PYTHON" -m aspire.eval.report --root "$OUT" --output "$OUT/report" \
  --reference ast_fim/base --reference ast_fim/aspire --reference plain/base \
  --overlap-train "${TRAIN_CSV:-runs/data/raw/train_compositional.csv}" --suite "$SUITE" --split "$SPLIT"

# CPU analyses (no model): failure taxonomy of initial generations and counterexample
# hard-coding of refinement runs; then judge accuracy of the verifier (Table 4).
ANALYSE=()
for d in plain/base random_fim/base ast_fim/base ast_any/base plain_gcd/base \
         ast_fim/aspire ast_fim/template ast_fim/pgs ast_fim/aspire_facts; do
  [[ -f "$OUT/$d/COMPLETE.json" ]] && ANALYSE+=("$OUT/$d")
done
"$PYTHON" -m aspire.eval.errors --suite "$SUITE" --split "$SPLIT" --output "$OUT/report/errors" "${ANALYSE[@]}"
"$PYTHON" -m aspire.eval.verifier_bench --suite "$SUITE" --split "$SPLIT" --output "$OUT/report/verifier" \
  --per-family "${BENCH_PER_FAMILY:-10}" --probes "${BENCH_PROBES:-50}"
