#!/usr/bin/env bash
# Run novel-melee zero-shot one type per process (avoids JAX cross-type JIT glitches).
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
PY=.venv/bin/python3
OUT=runs/benchmark_reports/fight_novel_melee_types
mkdir -p "$OUT"
: >"$OUT/run.log"
: >"$OUT/raw_rows.csv"

header_written=0
for tid in 0 1 2; do
  echo "=== type $tid $(date -Iseconds) ===" | tee -a "$OUT/run.log"
  tmp=$(mktemp -d)
  "$PY" -m evaluation.eval_fight_novel_melee_types \
    --num-rollouts 256 --seeds 0,1,2,3,4 --types "$tid" \
    --out-dir "$tmp" 2>&1 | tee -a "$OUT/run.log"
  if [[ ! -f "$tmp/raw_rows.csv" ]]; then
    echo "ERROR: no raw_rows for type $tid" | tee -a "$OUT/run.log"
    exit 1
  fi
  if [[ "$header_written" -eq 0 ]]; then
    cat "$tmp/raw_rows.csv" >"$OUT/raw_rows.csv"
    header_written=1
  else
    tail -n +2 "$tmp/raw_rows.csv" >>"$OUT/raw_rows.csv"
  fi
  rm -rf "$tmp"
done

# Rebuild summary + REPORT from merged CSV
"$PY" - <<'PY' | tee -a "$OUT/run.log"
import csv, json, numpy as np
from pathlib import Path
out = Path("runs/benchmark_reports/fight_novel_melee_types")
rows = list(csv.DictReader(out.joinpath("raw_rows.csv").open()))
summary = []
for tid in sorted({int(r["melee_type_id"]) for r in rows}):
    name = next(r["melee_type_name"] for r in rows if int(r["melee_type_id"]) == tid)
    hp = int(next(r["melee_hp"] for r in rows if int(r["melee_type_id"]) == tid))
    entry = {"melee_type_id": tid, "melee_type_name": name, "melee_hp": hp}
    for src in ("classic", "untrained"):
        ks = [float(r["kill_rate"]) for r in rows if int(r["melee_type_id"]) == tid and r["weight_source"] == src]
        ss = [float(r["score"]) for r in rows if int(r["melee_type_id"]) == tid and r["weight_source"] == src]
        entry[f"{src}_kill_mean"] = float(np.mean(ks))
        entry[f"{src}_kill_std"] = float(np.std(ks))
        entry[f"{src}_score_mean"] = float(np.mean(ss))
        entry[f"{src}_score_std"] = float(np.std(ss))
    summary.append(entry)
z0 = next(s for s in summary if s["melee_type_id"] == 0)["classic_kill_mean"]
for s in summary:
    s["retention_vs_zombie"] = float(s["classic_kill_mean"] / z0) if z0 > 1e-8 else float("nan")
out.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
lines = [
    "# Classic fight → Full novel melee types (zero-shot)",
    "",
    "Classic hierarchy `monster_fight` weights, greedy Full bootstrap, moving melee.",
    "Obs sums melee channels (no type identity). Types 0/1/2 = zombie / gnome / orc.",
    "",
    "| type_id | name | HP | Classic kill | Untrained kill | vs zombie |",
    "|--------:|------|---:|-------------:|---------------:|----------:|",
]
for s in summary:
    lines.append(
        f"| {s['melee_type_id']} | {s['melee_type_name']} | {s['melee_hp']} | "
        f"{s['classic_kill_mean']:.3f}±{s['classic_kill_std']:.3f} | "
        f"{s['untrained_kill_mean']:.3f}±{s['untrained_kill_std']:.3f} | "
        f"{s['retention_vs_zombie']:.2f} |"
    )
out.joinpath("REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo "=== done $(date -Iseconds) ===" | tee -a "$OUT/run.log"
