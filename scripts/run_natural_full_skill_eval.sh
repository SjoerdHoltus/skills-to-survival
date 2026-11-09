#!/usr/bin/env bash
# Natural Full Craftax skill eval: Classic zero-shot + 50-gen fine-tunes.
set -euo pipefail
cd "$(dirname "$0")/.."
# Prefer GPU; unset CPU pin from Classic recipe. Override with JAX_PLATFORMS=cpu if needed.
unset JAX_PLATFORMS || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
PY=.venv/bin/python3
OUT=runs/benchmark_reports
mkdir -p "$OUT"

SEEDS="${SEEDS:-1024}"
BATCH="${BATCH:-32}"
STEPS="${STEPS:-128}"

echo "=== classic → natural Full $(date -Iseconds) ==="
"$PY" -c 'import jax; print("devices:", jax.devices())'
"$PY" -m evaluation.batch_eval_skills_natural_craftax \
  --weight-set classic \
  --num-seeds "$SEEDS" --batch-size "$BATCH" --steps "$STEPS" --seed 0 \
  --output-json "$OUT/skill_natural_eval_full_classic_zeroshot.json"

echo "=== finetune50 → natural Full $(date -Iseconds) ==="
"$PY" -m evaluation.batch_eval_skills_natural_craftax \
  --weight-set finetune50 \
  --num-seeds "$SEEDS" --batch-size "$BATCH" --steps "$STEPS" --seed 0 \
  --output-json "$OUT/skill_natural_eval_full_finetune50.json"

"$PY" - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

a = load("runs/benchmark_reports/skill_natural_eval_full_classic_zeroshot.json")
b = load("runs/benchmark_reports/skill_natural_eval_full_finetune50.json")
rows = {r["skill"]: r for r in a["results"]}
ft = {r["skill"]: r for r in b["results"]}
lines = [
    "# Natural Full Craftax skill eval",
    "",
    "Protocol mirrors Classic natural eval: unmodified Extended maps, light staging,",
    "greedy skill-only rollouts, N=1024, horizon=128.",
    "",
    "| Skill | Classic→Full success | succ\\|ever | FT50 success | succ\\|ever |",
    "|-------|---------------------:|-----------:|-------------:|-----------:|",
]
for skill in ("find_wood", "drink_water", "find_food", "monster_fight", "collect_stone", "collect_ore"):
    c = rows.get(skill)
    f = ft.get(skill)
    if c is None:
        continue
    c_s = f"{100*c['success_rate']:.1f}%"
    c_e = f"{100*c['success_rate_given_ever_saw_target']:.1f}%"
    if f:
        f_s = f"{100*f['success_rate']:.1f}%"
        f_e = f"{100*f['success_rate_given_ever_saw_target']:.1f}%"
    else:
        f_s, f_e = "—", "—"
    lines.append(f"| {skill} | {c_s} | {c_e} | {f_s} | {f_e} |")
lines += [
    "",
    "Classic zeroshot JSON: `skill_natural_eval_full_classic_zeroshot.json`",
    "Fine-tune50 JSON: `skill_natural_eval_full_finetune50.json`",
]
Path("runs/benchmark_reports/skill_natural_eval_full_REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "=== done $(date -Iseconds) ==="
