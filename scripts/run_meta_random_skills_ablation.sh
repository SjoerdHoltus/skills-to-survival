#!/usr/bin/env bash
# Ablation: Classic unmasked Meta-PPO with Flax-default-init (untrained) frozen skills.
# Trains for BUDGET env steps (default 20m) over SEEDS (default 0 1 2 3 4), then
# greedy-evals each run — skill-quality control vs trained OpenES skills.
#
# Examples:
#   ./scripts/run_meta_random_skills_ablation.sh
#   BUDGET=100k SEEDS="0 1" ./scripts/run_meta_random_skills_ablation.sh
#   SKIP_EVAL=1 ./scripts/run_meta_random_skills_ablation.sh
#
# Environment:
#   SEEDS                Space-separated training seeds (default: 0 1 2 3 4)
#   BUDGET               Env-step budget label (default: 20m)
#   RANDOM_SKILLS_SEED   Fixed skill-init seed across meta seeds (default: 0)
#   SKIP_EVAL=1          Skip greedy eval
#   SKIP_IF_CHECKPOINT=1 Skip train when a checkpoint already exists (default: 1)
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpu_train_helpers.sh
source "$(dirname "$0")/gpu_train_helpers.sh"

PYTHON="$(detect_python)"
configure_gpu_env

SEEDS="${SEEDS:-0 1 2 3 4}"
BUDGET="${BUDGET:-20m}"
RANDOM_SKILLS_SEED="${RANDOM_SKILLS_SEED:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_IF_CHECKPOINT="${SKIP_IF_CHECKPOINT:-1}"

RUNS_POLICY="baselines_classic_meta_ppo_no_mask_random_skills_${BUDGET}"

echo "=== Meta random-skills ablation: budget=${BUDGET}, seeds=${SEEDS}, skills_seed=${RANDOM_SKILLS_SEED} ==="

find_run_dir() {
  local seed="$1"
  "${PYTHON}" -c "
from glob import glob
import json, os
root='runs/${RUNS_POLICY}'
cands=sorted(glob(root+'/run_*'))
for d in reversed(cands):
    cfg=os.path.join(d,'config.json')
    w=os.path.join(d,'weights','params.msgpack')
    if not (os.path.isfile(cfg) and os.path.isfile(w)):
        continue
    hyp=json.load(open(cfg)).get('hyperparameters') or {}
    if int(hyp.get('seed',-1))==${seed} and bool(hyp.get('random_skills', False)):
        print(d); break
"
}

for seed in ${SEEDS}; do
  LABEL="Classic meta PPO 128 no mask random skills (${BUDGET}, seed=${seed})"
  echo ""
  echo "################################################################"
  echo "# Seed ${seed}  |  budget ${BUDGET}"
  echo "################################################################"

  if [[ "${SKIP_IF_CHECKPOINT}" == "1" ]] && maybe_skip_train "${LABEL}" "${RUNS_POLICY}" "${seed}" "" "${PYTHON}"; then
    :
  else
    run_gpu_train "${LABEL}" \
      "${PYTHON}" -m ppo.meta_ppo \
      --no-use-action-mask \
      --random-skills \
      --random-skills-seed "${RANDOM_SKILLS_SEED}" \
      --budget "${BUDGET}" \
      --seed "${seed}"
  fi

  RUN_DIR="$(find_run_dir "${seed}")"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "ERROR: could not find run dir under runs/${RUNS_POLICY} for seed ${seed}"
    exit 1
  fi
  echo "Run dir: ${RUN_DIR}"

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    gpu_cooldown
    echo ">>> Greedy eval (seed ${seed})"
    OUT="${RUN_DIR}/evals/greedy_eval_seed42_128ep.json"
    "${PYTHON}" -m evaluation.benchmark_eval_official_ppo \
      --policy meta \
      --run-dir "${RUN_DIR}" \
      --num-episodes 128 \
      --master-seed 42 \
      --eval-action-mask auto \
      --fast \
      --output "${OUT}"
    echo "Eval: ${OUT}"
  fi
done

echo ""
echo "Done. Seeds ${SEEDS} under runs/${RUNS_POLICY}/"
