#!/usr/bin/env bash
# Full multi-seed baseline budget pipeline:
#   train (optional) -> greedy eval -> per-seed metrics -> multiseed aggregate -> plots
#
# Examples:
#   ./scripts/run_baseline_budget_multiseed_pipeline.sh
#   SKIP_TRAIN=1 ./scripts/run_baseline_budget_multiseed_pipeline.sh
#   SEEDS="0 1 2 3 4" BUDGETS=100k,500k,1m,2m,5m,10m,20m ./scripts/run_baseline_budget_multiseed_pipeline.sh
#
# Environment:
#   SEEDS          Space-separated training seeds (default: 0 1 2 3 4)
#   BUDGETS        Comma-separated budget labels (default: all seven)
#   SKIP_TRAIN=1   Skip training; only eval + metrics + plots
#   SKIP_EVAL=1    Skip greedy eval (metrics/plots from existing eval JSON)
#   SKIP_SEEDS     Space-separated seeds to skip during training
#   FORCE_EVAL=1   Re-run greedy eval even if JSON exists
#   REPORT_DIR     Output directory (default: runs/benchmark_reports/baseline_migration)
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
fi

SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGETS="${BUDGETS:-100k,500k,1m,2m,5m,10m,20m}"
REPORT_DIR="${REPORT_DIR:-runs/benchmark_reports/baseline_migration}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

mkdir -p "${REPORT_DIR}"

echo "=== Baseline budget multiseed pipeline ==="
echo "Seeds:   ${SEEDS}"
echo "Budgets: ${BUDGETS}"
echo "Report:  ${REPORT_DIR}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo ""
  echo ">>> Training"
  # shellcheck disable=SC2086
  SKIP_SEEDS="${SKIP_SEEDS:-}" ./scripts/run_baseline_budget_sweep_multi_seed.sh ${SEEDS}
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
echo ">>> Plots"
for seed in ${SEEDS}; do
  "${PYTHON}" -m evaluation.plot_growth_by_budget \
    --input "${REPORT_DIR}/greedy_vs_training_by_budget_seed${seed}.json" \
    --output "${REPORT_DIR}/growth_by_budget_seed${seed}.png" \
    --metric greedy_eval_pct_max

  "${PYTHON}" -m evaluation.plot_growth_by_budget \
    --input "${REPORT_DIR}/greedy_vs_training_by_budget_seed${seed}.json" \
    --output "${REPORT_DIR}/growth_by_budget_training_seed${seed}.png" \
    --metric training_return_pct_max
done

"${PYTHON}" -m evaluation.plot_growth_by_budget \
  --input "${REPORT_DIR}/greedy_vs_training_multiseed.json" \
  --output "${REPORT_DIR}/growth_by_budget_multiseed_greedy.png" \
  --metric greedy_eval_pct_max \
  --multiseed

"${PYTHON}" -m evaluation.plot_growth_by_budget \
  --input "${REPORT_DIR}/greedy_vs_training_multiseed.json" \
  --output "${REPORT_DIR}/growth_by_budget_multiseed_training.png" \
  --metric training_return_pct_max \
  --multiseed

echo ""
echo "Done."
echo "  Per-seed JSON: ${REPORT_DIR}/greedy_vs_training_by_budget_seed<seed>.json"
echo "  Multiseed JSON: ${REPORT_DIR}/greedy_vs_training_multiseed.json"
echo "  Multiseed plot: ${REPORT_DIR}/growth_by_budget_multiseed_greedy.png"
