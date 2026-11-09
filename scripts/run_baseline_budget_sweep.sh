#!/usr/bin/env bash
# Train non-RNN baselines with checkpoints for one or more budgets.
#
# Budget labels: 100k, 500k, 1m, 2m, 5m, 10m, 20m (all env-step budgets)
#
# Examples:
#   ./scripts/run_baseline_budget_sweep.sh                              # all seven budgets
#   BUDGETS=100k,500k,2m,5m,20m ./scripts/run_baseline_budget_sweep.sh
#   BUDGETS=100k ./scripts/run_baseline_budget_sweep.sh
#
# Environment (GPU / resume):
#   SKIP_IF_CHECKPOINT=1     Skip jobs that already have a checkpoint (default: 1)
#   TRAIN_COOLDOWN_SEC=10    Pause between successful GPU jobs (default: 10)
#   TRAIN_GPU_RETRIES=3      Retry CUDA failures (default: 3)
#   TRAIN_GPU_RETRY_WAIT_SEC=20
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpu_train_helpers.sh
source "$(dirname "$0")/gpu_train_helpers.sh"

PYTHON="$(detect_python)"
configure_gpu_env

SEED="${SEED:-42}"
BUDGETS="${BUDGETS:-100k,500k,1m,2m,5m,10m,20m}"
EXTRA=( "$@" )

IFS=',' read -r -a BUDGET_LIST <<< "${BUDGETS}"

train_flat() {
  local preset="$1"
  local budget="$2"
  local width="${3:-}"
  local runs_policy="baselines_${preset}_flat_ppo_${budget}"
  if [[ "${preset}" == "extended" ]]; then
    runs_policy="baselines_extended_ppo_${budget}"
  fi
  local label="Classic flat PPO ${width} (${budget})"
  if [[ "${preset}" == "extended" ]]; then
    label="Extended flat PPO 512 (${budget})"
  fi

  if maybe_skip_train "${label}" "${runs_policy}" "${SEED}" "${width}" "${PYTHON}"; then
    return 0
  fi

  local cmd=(
    "${PYTHON}" -m ppo.flat_ppo
    --preset "${preset}"
    --budget "${budget}"
    --seed "${SEED}"
  )
  if [[ -n "${width}" ]]; then
    cmd+=(--layer-size "${width}")
  fi
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    cmd+=("${EXTRA[@]}")
  fi
  run_gpu_train "${label}" "${cmd[@]}"
}

train_meta() {
  local budget="$1"
  local use_mask="$2"
  local runs_policy="baselines_classic_meta_ppo_${budget}"
  local label="Classic meta PPO 128 + mask (${budget})"
  if [[ "${use_mask}" == "0" ]]; then
    runs_policy="baselines_classic_meta_ppo_no_mask_${budget}"
    label="Classic meta PPO 128 no mask (${budget})"
  fi

  if maybe_skip_train "${label}" "${runs_policy}" "${SEED}" "" "${PYTHON}"; then
    return 0
  fi

  local cmd=(
    "${PYTHON}" -m ppo.meta_ppo
    --budget "${budget}"
    --seed "${SEED}"
  )
  if [[ "${use_mask}" == "0" ]]; then
    cmd+=(--no-use-action-mask)
  fi
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    cmd+=("${EXTRA[@]}")
  fi
  run_gpu_train "${label}" "${cmd[@]}"
}

train_budget() {
  local budget="$1"
  echo ""
  echo "################################################################"
  echo "# Budget: ${budget}  |  seed: ${SEED}"
  echo "################################################################"

  train_flat extended "${budget}"
  for WIDTH in 512 256 128; do
    train_flat classic "${budget}" "${WIDTH}"
  done
  train_meta "${budget}" 1
  train_meta "${budget}" 0
}

for budget in "${BUDGET_LIST[@]}"; do
  budget="$(echo "${budget}" | tr -d '[:space:]')"
  [[ -n "${budget}" ]] || continue
  train_budget "${budget}"
done

echo ""
echo "Done. Runs under runs/baselines_*_{${BUDGETS}}/"
echo "Aggregate comparison:"
echo "  ${PYTHON} -m evaluation.aggregate_training_return_by_budget --seed ${SEED}"
