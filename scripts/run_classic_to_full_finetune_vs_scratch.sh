#!/usr/bin/env bash
# Matched fine-tune vs scratch on Full Craftax (Craftax Extended).
# Drink / food / fight / wood / stone / ore @ G generations;
# Classic DEFAULT_SKILL_WEIGHTS for fine-tune init.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=gpu_train_helpers.sh
source scripts/gpu_train_helpers.sh

PY="$(detect_python)"
configure_gpu_env

GENS="${GENS:-50}"
SEED="${SEED:-42}"
# Comma list of skills (default: all six)
ONLY="${ONLY:-drink_water,find_food,monster_fight,find_wood,collect_stone,collect_ore}"
OUT_DIR="${OUT_DIR:-runs/benchmark_reports/classic_to_full_finetune_vs_scratch}"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/manifest.csv"
LOG="${OUT_DIR}/runner.log"

want() {
  [[ ",${ONLY}," == *",$1,"* ]]
}

DRINK_W="checkpoints/skills/drink_water.npy"
FOOD_W="checkpoints/skills/find_food.npy"
FIGHT_W="checkpoints/skills/monster_fight.npy"
WOOD_W="checkpoints/skills/find_wood.npy"
STONE_W="checkpoints/skills/collect_stone.npy"
ORE_W="checkpoints/skills/collect_ore.npy"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "skill,arm,seed,generations,resume,status,run_dir,started_at,finished_at" >"${MANIFEST}"
fi

newest_run_dir() {
  local folder="$1"
  local before="$2"
  local py="$3"
  "${py}" - <<PY
import os, glob
folder = "runs/${folder}"
before = float("${before}")
cands = []
for p in glob.glob(os.path.join(folder, "run_*")):
    if not os.path.isdir(p):
        continue
    m = os.path.getmtime(p)
    if m >= before - 1.0:
        cands.append((m, p))
print(max(cands)[1] if cands else "")
PY
}

run_one() {
  local skill="$1"
  local arm="$2"
  local module="$3"
  local folder="$4"
  shift 4
  local started finished status run_dir resume="null"
  local -a cmd=("${PY}" -m "${module}" --variant craftax --generations "${GENS}" --seed "${SEED}")

  if [[ "${arm}" == "finetune" ]]; then
    case "${skill}" in
      drink_water) resume="${DRINK_W}" ;;
      find_food) resume="${FOOD_W}" ;;
      monster_fight) resume="${FIGHT_W}" ;;
      find_wood) resume="${WOOD_W}" ;;
      collect_stone) resume="${STONE_W}" ;;
      collect_ore) resume="${ORE_W}" ;;
    esac
    cmd+=(--resume "${resume}")
  fi
  cmd+=("$@")

  started="$(date -Iseconds)"
  local t0
  t0="$(date +%s)"
  echo "[$(date -Iseconds)] START ${skill}/${arm}: ${cmd[*]}" | tee -a "${LOG}"
  # `|| status=$?` so set -e does not abort when run_gpu_train returns non-zero.
  status=0
  run_gpu_train "${skill}_${arm}" "${cmd[@]}" || status=$?
  finished="$(date -Iseconds)"
  run_dir="$(newest_run_dir "${folder}" "${t0}" "${PY}")"
  echo "${skill},${arm},${SEED},${GENS},${resume},${status},${run_dir},${started},${finished}" >>"${MANIFEST}"
  echo "[$(date -Iseconds)] DONE ${skill}/${arm} status=${status} run_dir=${run_dir}" | tee -a "${LOG}"
  if [[ "${status}" -ne 0 ]]; then
    echo "WARN: ${skill}/${arm} failed; continuing" | tee -a "${LOG}"
  fi
}

{
  echo "=== classic→full fine-tune vs scratch  gens=${GENS} seed=${SEED} only=${ONLY} ==="
  date -Iseconds

  # Drink: Classic defaults (pop 100, 64 rollouts, horizon 20)
  if want drink_water; then
    run_one drink_water scratch skills.drink_water drink_water_craftax \
      --pop-size 100 --num-rollouts 64
    run_one drink_water finetune skills.drink_water drink_water_craftax \
      --pop-size 100 --num-rollouts 64
  fi

  # Food / fight: Classic moving finals → Full moving (matched deployment setting).
  # ponytail: 128 rollouts × 90 steps OOMs on 6GB; keep arms matched at 64.
  if want find_food; then
    run_one find_food scratch skills.find_food find_food_craftax \
      --pop-size 100 --num-rollouts 64 --cow-mode moving
    run_one find_food finetune skills.find_food find_food_craftax \
      --pop-size 100 --num-rollouts 64 --cow-mode moving
  fi

  if want monster_fight; then
    run_one monster_fight scratch skills.monster_fight monster_fight_craftax \
      --pop-size 100 --num-rollouts 64 --zombie-mode moving
    run_one monster_fight finetune skills.monster_fight monster_fight_craftax \
      --pop-size 100 --num-rollouts 64 --zombie-mode moving
  fi

  # Gathering: same matched budget as drink (pop 100, 64 rollouts).
  if want find_wood; then
    run_one find_wood scratch skills.find_wood find_wood_craftax \
      --pop-size 100 --num-rollouts 64
    run_one find_wood finetune skills.find_wood find_wood_craftax \
      --pop-size 100 --num-rollouts 64
  fi

  if want collect_stone; then
    run_one collect_stone scratch skills.collect_stone collect_stone_craftax \
      --pop-size 100 --num-rollouts 64
    run_one collect_stone finetune skills.collect_stone collect_stone_craftax \
      --pop-size 100 --num-rollouts 64
  fi

  if want collect_ore; then
    run_one collect_ore scratch skills.collect_ore collect_ore_craftax \
      --pop-size 100 --num-rollouts 64
    run_one collect_ore finetune skills.collect_ore collect_ore_craftax \
      --pop-size 100 --num-rollouts 64
  fi

  echo "=== all jobs finished ==="
  date -Iseconds
} 2>&1 | tee -a "${LOG}"
