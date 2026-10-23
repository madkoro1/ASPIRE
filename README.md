# ASPIRE

**ASP** generation with AST-guided **I**nfilling and semantic counterexample **RE**finement.

This repository is the code for the paper *基于 AST 补全与反例精炼的答案集程序生成方法*.
The paper calls the AST-guided training strategy **ASP-FIM**; its code arm is `ast_fim`
(the name AST-FIM in the paper refers only to the baseline of Gong et al. [18], code arm `ast_any`).
It contains the dataset generator, the evaluation suite builder, the three training conditions
(SFT / Random-FIM / ASP-FIM, plus the generic AST-subtree baseline), the multi-context projection verifier, the refinement loop, and
the refinement baselines.

## Layout

| Path | Paper section | Content |
|---|---|---|
| `aspire/data/generate.py`, `vocab.py` | 3.1 | Medical-rescue NL→ASP generator (Track A families, disjoint train/test vocabularies) |
| `aspire/data/overlap.py` | 3.1 | Train/test exact and predicate-renamed program overlap |
| `aspire/data/freeze.py`, `suite.py` | 2.1, 2.4, 3.1 | Freezes the test suite into public (question, F0, interface, feedback contexts) and private (held-out contexts; reference kept for audit) files; every context stores its offline expected output |
| `aspire/families.py` | Table 1 | The 13 evaluated families and their easy/mid/hard groups |
| `aspire/training/fim.py`, `asp_ast.py` | 2.2, Eq. 1 | Clingo AST spans (complete rule, body literal, comparison literal), token-matched random spans, answer-only encoding |
| `aspire/training/prepare.py` | 2.3, 3.2 | Builds the paired `plain` / `random_fim` / `ast_fim` training sets (requested ρ = 0.2; paired slots that fail fall back to full programs, actual 18.97%) |
| `aspire/training/train.py` | 2.3, Eq. 2, 3.2 | BF16 LoRA training (unquantized base weights); loss = answer+EOS NLL summed over an update / supervised tokens in that update |
| `aspire/verify/solver.py`, `oracle.py` | 2.4–2.5, Eq. 3–6 | Projection of all stable models, feasible/optimal phases, V(P,F) against frozen expected outputs, R(P), visible score s(P) |
| `aspire/refine/loop.py` | 2.5 | Semantic feedback refinement and all refinement baselines in one loop |
| `aspire/models/prompts.py` | 2.5, 3.5 | Every prompt, with its source |
| `aspire/models/backend.py` | 3.2 | Local HF + LoRA (adapter disabled for the feedback role), OpenAI-compatible API, offline mock |
| `aspire/eval/run.py`, `report.py` | 3.3–3.6, Eq. 7 | Resumable runs; strict / held-out / visible pass rates, OF, fixes/regressions, McNemar, bootstrap CI, costs |

## Methods (`--method`)

All refinement methods share the same loop: candidates are verified on the visible contexts
C_vis = {F0} ∪ C_fb, the best candidate by visible score is kept (strictly higher replaces),
and the loop stops on a full visible pass, an empty program, an unfinished verification or after
T rounds. Only the way the next candidate is produced differs.

| Method | Table | What produces the next candidate | Source |
|---|---|---|---|
| `base` | 2, 3, 5, 6 | Generator, one pass | — |
| `fewshot` | 2 | Model without adapter, k training-vocabulary exemplars | — |
| `aspire` | 2, 3, 5, 6 | Verified diagnosis → feedback LLM (five fields Category / Rule / Violated condition / Phase / Fix direction, checked by the program; invalid outputs rejected) → generator | this paper |
| `template` | 5 | Verified diagnosis rendered by template → generator | this paper (control) |
| `syntax_only` | 6 | `aspire`, but only parse/grounding errors are diagnosed | this paper (ablation) |
| `self_debug` | 5 | Generator explains its program line by line, then fixes it given one failing visible context | Chen et al., ICLR 2024 (UT+Expl, App. G.7) |
| `pgs` | 5 | Feedback LLM states the violated property; generator repairs with the PGS repair prompt | He et al., arXiv 2506.18315 (Tester feedback + Table 13 only) |
| `llm_asp` | 5 | Generator sees clingo output on F0 (no expected output) and answers UPDATE or PASS | Ishay & Lee, Findings ACL 2026 (official repo prompt) |
| `aspire_facts` | supplementary | `aspire`, but the generator also receives the counterexample facts | this paper (ablation) |

Invalid feedback: one retry that states the rejection reason; if it is rejected again the round
is consumed and the next round uses the next failing visible context (stop `feedback_exhausted`
when none is left).

Ablation switches: `--visible instance` (only F0), `--selection random`, `--keep last`,
`--verify-projection off` (whole answer sets during refinement), `--verify-phases optimal`
(skip the feasible phase during refinement). Final scoring always uses the standard verifier.
`--grammar asp` decodes generator calls under an ASP grammar (`transformers-cfg`).

Training arms: `plain`, `random_fim`, `ast_fim`, and `ast_any` (generic AST-subtree infilling:
uniform over all AST nodes, in the same slots as `ast_fim`, length not matched).

Adaptations made for every baseline: budget T = 3 (papers use 10 or 5), greedy decoding, best
candidate by visible score, ASP instead of Python. `pgs` omits property checkers and input
synthesis; `llm_asp` uses a short ASP guide instead of the authors' resource document. These
choices are recorded in each run's `run_config.json` (`source`).

## Install

```bash
python -m pip install -r requirements.txt  # Linux/CUDA 12.8 server runtime, including PyTorch
python -m unittest discover -s tests -q
```

The requirements file contains the full supplied server dependency snapshot, including
core packages, transitive dependencies, and NVIDIA CUDA runtime packages, with every
package version pinned. `pytest` is optional and is not included in that snapshot.
For CPU-only verification/data/API use, install
`clingo==5.8.0 pandas==2.2.3 requests==2.32.3`.

## Reproduce

```bash
# 1. data: 10k/1k training rows, 50 test rows per family, frozen suite, paired FIM sets
TOKENIZER=/models/Qwen2.5-32B bash scripts/prepare_data.sh

# 2. training: plain / random_fim / ast_fim for each seed (one GPU)
MODEL=/models/Qwen2.5-32B DATA=runs/data/fim SEEDS="56 123 2024" bash scripts/train_all.sh

# 3. evaluation: every local row of Tables 2-6 for one seed, then the report
MODEL=/models/Qwen2.5-32B SUITE=runs/data/suite TRAIN=runs/train SEED=56 bash scripts/eval_all.sh
# with the two API rows as well
ENDPOINT="$URL" FEWSHOT_API_MODEL=glm-5.2 FEEDBACK_API_MODEL=Qwen2.5-32B-Instruct \
  MODEL=/models/Qwen2.5-32B SUITE=runs/data/suite TRAIN=runs/train SEED=56 bash scripts/eval_all.sh
```

Single runs:

```bash
python -m aspire.eval.run --suite runs/data/suite --method base --backend local \
  --model /models/Qwen2.5-32B --adapter runs/train/seed56/ast_fim/adapter --output runs/eval/base
python -m aspire.eval.run --suite runs/data/suite --method aspire --backend local \
  --model /models/Qwen2.5-32B --adapter runs/train/seed56/ast_fim/adapter \
  --initial runs/eval/base --output runs/eval/aspire
python -m aspire.eval.report --root runs/eval --output runs/eval/report --reference base --reference aspire
```

Optional rows that use an API (OpenAI-compatible `/chat/completions`, key in `ASPIRE_API_KEY`):

```bash
# instruction model few-shot (Table 2)
python -m aspire.eval.run --suite runs/data/suite --method fewshot --backend api \
  --endpoint "$URL" --model glm-5.2 --generator-style chat --output runs/eval/fewshot_api
# Instruct feedback model with the local generator (optional, not in the paper tables)
python -m aspire.eval.run ... --method aspire --feedback-backend api --endpoint "$URL" \
  --feedback-model Qwen2.5-32B-Instruct --feedback-style chat --initial runs/eval/base --output runs/eval/aspire_instruct_fb
```

A plumbing check without a GPU: `--backend mock --model mock --limit 13`.

Supplementary experiments (reviewer requests). Existing completed runs are skipped.

```bash
# a. rebuild the FIM sets into a NEW directory (adds ast_any; plain/random_fim/ast_fim are
#    regenerated identically - compare their sha256 with the old directory before reuse)
python -m aspire.training.prepare --train runs/data/raw/train_compositional.csv \
  --validation runs/data/raw/val_compositional.csv --output runs/data/fim_v2 \
  --tokenizer /models/Qwen2.5-32B --rate 0.2 --seed 20260924 --seq-length 1024 --attempts 512
# b. train only the new arm
ARMS=ast_any MODEL=/models/Qwen2.5-32B DATA=runs/data/fim_v2 bash scripts/train_all.sh
# c. new evaluation rows (ast_any, grammar-constrained decoding, no_projection, optimal_only,
#    aspire_facts) and the CPU analyses (failure taxonomy, hard-coding, verifier benchmark)
pip install transformers-cfg==0.2.7
MODEL=/models/Qwen2.5-32B SUITE=runs/data/suite TRAIN=runs/train SEED=56 bash scripts/eval_all.sh
```

CPU analyses alone: `python -m aspire.eval.errors --suite S --output O RUN...` and
`python -m aspire.eval.verifier_bench --suite S --output O [--per-family 50 --probes 50]`.

## Released results

Paper numbers are traceable only through the run directories. For each table
row, the result bundle (distributed separately from this repository; `runs/`
is git-ignored) contains `run_config.json` (method, seed, suite_id, model and
adapter hashes, code hashes), `protocol_samples.jsonl` (per-sample programs,
traces and verdicts), `generation_cache.jsonl` (every model call) and
`COMPLETE.json`; the suite and FIM directories carry their own manifests with
SHA-256 hashes. `ROWS.md` in the bundle maps every row of Tables 2–6 to its run
directory and seed.

## Protocol notes

- The paper evaluates 13 families (`--families paper`, default). The generator also produces three
  single-construct L1 decision controls, which are excluded.
- Training-data syntax is wider than the Track A test syntax: training rows (atomic and seen
  composite templates) include brace choice rules, `#count` aggregates, conditional literals and
  variable-weight weak constraints; only the Track A evaluation families are restricted to
  disjunctive choice and fixed-weight weak constraints. AST spans never split aggregate or
  conditional-literal internals; such rules can only be masked as a complete rule.
- Test families have their own task identifiers and question texts, but some reference programs
  share templates with training templates (e.g. negative filtering, closure, join→negative filter);
  identical reference strings between training and test are possible; `aspire.data.overlap`
  (run by `scripts/prepare_data.sh`) reports exact Q/A, exact program and predicate-renamed
  program overlap per family in `overlap.json`.
- Expected outputs are frozen per context at suite-freeze time (schema 2). The runner verifies
  candidates against them only; it never executes the reference program, so contexts labelled by
  hand (facts + expected projections) can be verified in the same way.
- Held-out contexts are read only by `aspire.verify.oracle.evaluate`, after every inference decision.
- Feedback-context facts are never shown to the generator; the feedback LLM sees them, and its output
  is rejected if it contains ASP code, a ground atom, or a non-numeric constant that occurs only in
  the counterexample, if any of the five fields is missing, if the category is not one of the five
  categories, or if the phase disagrees with the verifier's diagnosis. A rejected feedback consumes
  the round; P_{k+1} = P_k and the generator is not called.
- Suite schema 2 replaces the id-shift contexts of five families (negative filtering, join,
  join→negative filter, candidate→negative filter, join→status→blocked) with discriminative contexts;
  results on schema-1 suites must not be mixed with schema-2 results.
- Bounded multi-context agreement is not a proof of equivalence on all inputs.
- Runs are resumable: every model call is cached in `generation_cache.jsonl`, every sample result in
  `protocol_samples.jsonl`; changing the configuration of an existing run directory is refused.

## License

Code: MIT (see `LICENSE`).
