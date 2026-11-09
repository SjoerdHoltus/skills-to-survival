#!/usr/bin/env python3
"""Paired seed-level tests for thesis greedy eval (training seeds 0--4).

Unit of analysis is the per-seed mean normalized return (%). Episodes are not
pooled across seeds (avoids pseudoreplication). Pairing is by training-seed ID;
greedy worlds already share master seed 42 within each whisker payload.

Writes JSON + LaTeX. No scipy: Student-t p-values via incomplete beta.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from statistics import mean, stdev

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT_DIR = os.path.join(_PROJECT_ROOT, "runs/benchmark_reports/baseline_migration")
TABLE_DIR = os.path.join(_PROJECT_ROOT, "figures", "output")

POLICY_LABEL = {
    "flat_512": "Flat PPO",
    "meta_no_mask": "Meta-PPO",
    "flat_rnn": "Flat PPO-RNN",
    "meta_rnn_no_mask": "Meta-PPO-RNN",
}

EXPECTED_SEEDS = [0, 1, 2, 3, 4]

# Thesis contrasts: (a_key, b_key, short label for a−b)
CONTRASTS = (
    ("flat_512", "meta_no_mask", "Flat $-$ Meta"),
    ("meta_rnn_no_mask", "flat_512", "Meta-RNN $-$ Flat"),
    ("meta_rnn_no_mask", "flat_rnn", "Meta-RNN $-$ Flat-RNN"),
    ("flat_512", "flat_rnn", "Flat $-$ Flat-RNN"),
    ("meta_rnn_no_mask", "meta_no_mask", "Meta-RNN $-$ Meta"),
)


def _betacf(a: float, b: float, x: float, *, max_iter: int = 200) -> float:
    """Continued fraction for incomplete beta (Lentz)."""
    tiny = 1e-30
    am = 1.0
    bm = 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    if abs(bz) < tiny:
        bz = tiny
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am, bm, az, bz = ap, bp, app, bpp
        if abs(bpp) < tiny:
            bpp = tiny
        az /= bpp
        bz = 1.0
        am /= bpp
        bm /= bpp
        if abs(az - am) < 3e-7 * abs(az):
            return az
    return az


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - (math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / b) * _betacf(
        b, a, 1.0 - x
    )


def student_t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided P(|T_df| >= |t|)."""
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    x = df / (df + t * t)
    return _betainc(0.5 * df, 0.5, x)


def student_t_critical_two_sided(df: int, alpha: float = 0.05) -> float:
    """Two-sided critical value via bisection (stdlib only)."""
    lo, hi = 0.0, 100.0
    target = alpha
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_sf_two_sided(mid, float(df)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_mean_diff(a: list[float], b: list[float], *, alpha: float = 0.05) -> dict:
    """Paired mean difference a−b with Student-t CI and two-sided p."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        raise ValueError("need n>=2 paired values")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    md = mean(diffs)
    sd = stdev(diffs)
    se = sd / math.sqrt(n)
    df = n - 1
    t_stat = md / se if se > 0 else float("inf") if md != 0 else 0.0
    t_crit = student_t_critical_two_sided(df, alpha)
    half = t_crit * se
    p = student_t_sf_two_sided(t_stat, float(df)) if math.isfinite(t_stat) else 0.0
    return {
        "n": n,
        "diffs": diffs,
        "mean_diff": md,
        "std_diff": sd,
        "se_diff": se,
        "df": df,
        "t_stat": float(t_stat),
        "t_crit": float(t_crit),
        "ci_low": md - half,
        "ci_high": md + half,
        "alpha": alpha,
        "p_two_sided": float(p),
        "ci_excludes_zero": (md - half) > 0.0 or (md + half) < 0.0,
    }


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _seed_aligned_values(
    block_a: dict, block_b: dict
) -> tuple[list[float], list[float], list[int]]:
    """Align by training seed ID (not list position)."""
    by_a = {int(r["seed"]): float(r["greedy_eval_pct_max"]) for r in block_a["per_seed_mean_pct"]}
    by_b = {int(r["seed"]): float(r["greedy_eval_pct_max"]) for r in block_b["per_seed_mean_pct"]}
    if sorted(by_a) != EXPECTED_SEEDS or sorted(by_b) != EXPECTED_SEEDS:
        raise ValueError(
            f"expected seeds {EXPECTED_SEEDS}, found {sorted(by_a)} and {sorted(by_b)}"
        )
    seeds = sorted(set(by_a) & set(by_b))
    return [by_a[s] for s in seeds], [by_b[s] for s in seeds], seeds


def analyze_whisker(path: str) -> dict:
    payload = _load(path)
    classic = payload["classic"]
    rows = []
    for a_key, b_key, label in CONTRASTS:
        a_vals, b_vals, seeds = _seed_aligned_values(classic[a_key], classic[b_key])
        stats = paired_mean_diff(a_vals, b_vals)
        rows.append(
            {
                "contrast": label,
                "a_key": a_key,
                "b_key": b_key,
                "a_label": POLICY_LABEL[a_key],
                "b_label": POLICY_LABEL[b_key],
                "seeds": seeds,
                "a_values": a_vals,
                "b_values": b_vals,
                **{k: v for k, v in stats.items() if k != "diffs"},
                "diffs": stats["diffs"],
            }
        )
    return {
        "source": os.path.abspath(path),
        "budget": payload.get("budget"),
        "num_episodes": payload.get("num_episodes"),
        "master_seed": payload.get("master_seed"),
        "seeds": payload.get("seeds"),
        "unit": "per_training_seed_mean_normalized_return_pct",
        "test": "paired_student_t_two_sided",
        "alpha": 0.05,
        "contrasts": rows,
    }


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p{<}0.001"
    return f"p{{=}}{p:.3f}"


def write_latex(results_by_ep: dict[int, dict], output_path: str) -> None:
    """One table: rows=contrasts, columns=128ep / 256ep mean diff [CI] (p)."""
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Paired mean differences in greedy normalized environment "
        "return (percentage points) after twenty million training steps. "
        "Each contrast is matched on training seeds $0$--$4$; evaluation "
        "uses master seed~$42$. Cells report mean difference with a "
        "Student-$t$ 95\\% CI and two-sided paired $t$-test $p$-value "
        "($n{=}5$, $\\mathrm{df}{=}4$). The analysis unit is the "
        "per-seed mean over evaluation episodes. Intervals and $p$-values "
        "are unadjusted for multiple comparisons.}",
        "\\label{tab:greedy_paired_tests_20m}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Contrast & 128 episodes & 256 episodes \\\\",
        "\\midrule",
    ]
    order = [c[2] for c in CONTRASTS]
    by_label = {
        ep: {r["contrast"]: r for r in block["contrasts"]} for ep, block in results_by_ep.items()
    }
    for label in order:
        cells = []
        for ep in (128, 256):
            r = by_label[ep][label]
            cells.append(
                f"${r['mean_diff']:+.2f}\\;"
                f"[{r['ci_low']:.2f},\\,{r['ci_high']:.2f}]\\;"
                f"({_fmt_p(r['p_two_sided'])})$"
            )
        # Escape already uses $-$ in labels
        lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _self_check() -> None:
    # Known: two-sided t critical for df=4 at alpha=0.05 is ~2.776
    tcrit = student_t_critical_two_sided(4, 0.05)
    assert abs(tcrit - 2.776445) < 1e-3, tcrit
    p = student_t_sf_two_sided(tcrit, 4.0)
    assert abs(p - 0.05) < 1e-3, p
    # Paired: identical series → mean_diff=0, p=1
    s = paired_mean_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(s["mean_diff"]) < 1e-12 and abs(s["p_two_sided"] - 1.0) < 1e-9
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        default=REPORT_DIR,
        help="Directory with greedy_whisker_thesis_quartet_20m_*ep.json",
    )
    parser.add_argument(
        "--latex",
        default=os.path.join(TABLE_DIR, "greedy_paired_tests_20m.tex"),
    )
    parser.add_argument(
        "--json-out",
        default=os.path.join(REPORT_DIR, "greedy_paired_tests_20m.json"),
    )
    parser.add_argument("--check", action="store_true", help="Run stdlib t self-check")
    args = parser.parse_args()

    if args.check:
        _self_check()
        return

    results_by_ep: dict[int, dict] = {}
    for n_ep in (128, 256):
        path = os.path.join(args.report_dir, f"greedy_whisker_thesis_quartet_20m_{n_ep}ep.json")
        block = analyze_whisker(path)
        results_by_ep[n_ep] = block
        print(f"=== {n_ep} ep (master_seed={block['master_seed']}) ===")
        for r in block["contrasts"]:
            print(
                f"  {r['contrast']}: Δ={r['mean_diff']:+.2f} "
                f"95%CI=[{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
                f"t={r['t_stat']:.2f} p={r['p_two_sided']:.4f}"
            )

    out = {
        "benchmark": "greedy_paired_tests",
        "by_num_episodes": {str(k): v for k, v in results_by_ep.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.json_out}")

    write_latex(results_by_ep, args.latex)
    print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
