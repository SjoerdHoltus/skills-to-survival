#!/usr/bin/env bash
# Train non-RNN baselines across multiple seeds (same budgets as single-seed sweep).
#
# Examples:
#   ./scripts/run_baseline_budget_sweep_multi_seed.sh
#   SEEDS="0 1 2 3 4" BUDGETS=100k,500k,1m,2m,5m,10m,20m ./scripts/run_baseline_budget_sweep_multi_seed.sh
#   SKIP_SEEDS="42" ./scripts/run_baseline_budget_sweep_multi_seed.sh   # skip seeds already done
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi

BUDGETS="${BUDGETS:-100k,500k,1m,2m,5m,10m,20m}"
SKIP_SEEDS="${SKIP_SEEDS:-}"

should_skip() {
  local seed="$1"
  for s in ${SKIP_SEEDS}; do
    [[ "${s}" == "${seed}" ]] && return 0
  done
  return 1
}

echo "Training seeds: ${SEEDS[*]}"
echo "Budgets: ${BUDGETS}"
echo "Skip seeds: ${SKIP_SEEDS:-<none>}"

for seed in "${SEEDS[@]}"; do
  if should_skip "${seed}"; then
    echo "=== Skipping seed ${seed} (SKIP_SEEDS) ==="
    continue
  fi
  echo ""
  echo "============================================================"
  echo " SEED ${seed}"
  echo "============================================================"
  SEED="${seed}" BUDGETS="${BUDGETS}" ./scripts/run_baseline_budget_sweep.sh
done

echo ""
echo "Training complete for seeds: ${SEEDS[*]}"
