#!/usr/bin/env bash
# Classic PPO-RNN chunked budget multiseed pipeline (flat + unmasked meta).
#
# One 20M run per (model, seed); checkpoints at 100k…20m along the way.
#
# Examples:
#   ./scripts/run_baseline_rnn_budget_multiseed_pipeline.sh
#   SEEDS="0 1" BUDGETS=100k,1m,20m ./scripts/run_baseline_rnn_budget_multiseed_pipeline.sh
#   SKIP_TRAIN=1 ./scripts/run_baseline_rnn_budget_multiseed_pipeline.sh
#
# Environment:
#   SEEDS / BUDGETS / SKIP_TRAIN / SKIP_EVAL / FORCE_EVAL / REPORT_DIR
#   SKIP_IF_CHECKPOINT=1 (default) — skip train if final-budget checkpoint exists
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpu_train_helpers.sh
source "$(dirname "$0")/gpu_train_helpers.sh"

PYTHON="$(detect_python)"
configure_gpu_env

SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGETS="${BUDGETS:-100k,500k,1m,2m,5m,10m,20m}"
REPORT_DIR="${REPORT_DIR:-runs/benchmark_reports/baseline_migration}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SKIP_IF_CHECKPOINT="${SKIP_IF_CHECKPOINT:-1}"

# Final budget label (last in comma list) used for skip checks.
FINAL_BUDGET="${BUDGETS##*,}"
FINAL_BUDGET="${FINAL_BUDGET// /}"

mkdir -p "${REPORT_DIR}"

echo "=== Classic PPO-RNN chunked budget multiseed ==="
echo "Seeds:   ${SEEDS}"
echo "Budgets: ${BUDGETS} (single run to ${FINAL_BUDGET} per model/seed)"
echo "Report:  ${REPORT_DIR}"

train_one() {
  local seed="$1"
  local kind="$2" # flat | meta_no_mask
  local runs_policy label
  if [[ "${kind}" == "flat" ]]; then
    runs_policy="baselines_classic_flat_ppo_rnn_${FINAL_BUDGET}"
    label="Classic flat PPO-RNN (${FINAL_BUDGET}, seed ${seed})"
    if [[ "${SKIP_IF_CHECKPOINT}" == "1" ]] \
      && maybe_skip_train "${label}" "${runs_policy}" "${seed}" "" "${PYTHON}"; then
      return 0
    fi
    run_gpu_train "${label}" \
      "${PYTHON}" -m ppo.flat_ppo_rnn \
      --preset classic \
      --chunked-budgets \
      --budgets "${BUDGETS}" \
      --seed "${seed}"
  else
    runs_policy="baselines_classic_meta_ppo_rnn_no_mask_${FINAL_BUDGET}"
    label="Classic meta PPO-RNN no-mask (${FINAL_BUDGET}, seed ${seed})"
    if [[ "${SKIP_IF_CHECKPOINT}" == "1" ]] \
      && maybe_skip_train "${label}" "${runs_policy}" "${seed}" "" "${PYTHON}"; then
      return 0
    fi
    run_gpu_train "${label}" \
      "${PYTHON}" -m ppo.meta_ppo_rnn \
      --no-use-action-mask \
      --chunked-budgets \
      --budgets "${BUDGETS}" \
      --seed "${seed}"
  fi
}

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo ""
  echo ">>> Training (chunked)"
  for seed in ${SEEDS}; do
    train_one "${seed}" flat
    train_one "${seed}" meta_no_mask
  done
else
  echo ""
  echo ">>> Skipping training (SKIP_TRAIN=1)"
fi

for seed in ${SEEDS}; do
  echo ""
  echo ">>> Seed ${seed}: greedy eval + per-seed metrics"

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    EVAL_ARGS=(--seed "${seed}" --budgets "${BUDGETS}")
    if [[ "${FORCE_EVAL}" == "1" ]]; then
      EVAL_ARGS+=(--force)
    fi
    "${PYTHON}" -m evaluation.batch_greedy_eval_by_budget "${EVAL_ARGS[@]}"
  else
    echo "    Skipping greedy eval (SKIP_EVAL=1)"
  fi

  "${PYTHON}" -m evaluation.aggregate_training_return_by_budget \
    --seed "${seed}" \
    --budgets "${BUDGETS}" \
    --output "${REPORT_DIR}/training_return_by_budget_seed${seed}.json"

  "${PYTHON}" -m evaluation.aggregate_greedy_vs_training_by_budget \
    --seed "${seed}" \
    --budgets "${BUDGETS}" \
    --output "${REPORT_DIR}/greedy_vs_training_by_budget_seed${seed}.json"
done

SEEDS_CSV="$(echo "${SEEDS}" | tr ' ' ',')"

echo ""
echo ">>> Multiseed aggregate"
"${PYTHON}" -m evaluation.aggregate_greedy_vs_training_multiseed \
  --seeds "${SEEDS_CSV}" \
  --report-dir "${REPORT_DIR}" \
  --output "${REPORT_DIR}/greedy_vs_training_multiseed.json"

echo ""
echo ">>> RNN pair plots"
for seed in ${SEEDS}; do
  "${PYTHON}" -m evaluation.plot_growth_by_budget \
    --input "${REPORT_DIR}/greedy_vs_training_by_budget_seed${seed}.json" \
    --output "${REPORT_DIR}/growth_classic_rnn_pair_greedy_seed${seed}.png" \
    --metric greedy_eval_pct_max \
    --models classic_rnn

  "${PYTHON}" -m evaluation.plot_growth_by_budget \
    --input "${REPORT_DIR}/greedy_vs_training_by_budget_seed${seed}.json" \
    --output "${REPORT_DIR}/growth_classic_rnn_pair_training_seed${seed}.png" \
    --metric training_return_pct_max \
    --models classic_rnn
done

"${PYTHON}" -m evaluation.plot_growth_by_budget \
  --input "${REPORT_DIR}/greedy_vs_training_multiseed.json" \
  --output "${REPORT_DIR}/growth_classic_rnn_pair_multiseed_greedy.png" \
  --metric greedy_eval_pct_max \
  --models classic_rnn \
  --multiseed

"${PYTHON}" -m evaluation.plot_growth_by_budget \
  --input "${REPORT_DIR}/greedy_vs_training_multiseed.json" \
  --output "${REPORT_DIR}/growth_classic_rnn_pair_multiseed_training.png" \
  --metric training_return_pct_max \
  --models classic_rnn \
  --multiseed

echo ""
echo "Done."
echo "  Multiseed JSON: ${REPORT_DIR}/greedy_vs_training_multiseed.json"
echo "  RNN plots: ${REPORT_DIR}/growth_classic_rnn_pair_multiseed_*.png"
