"""
TensorBoard logging utility for ES training scripts.

Also mirrors all logged scalars to JSON and CSV, and overwrites a standalone
``fitness_overview.png`` in the run directory every generation (worst/best band
+ mean line). The PNG is intentionally kept out of TensorBoard so tfevents
files stay small and load quickly.

Usage:
    from training.tb_logger import TBLogger

    logger = TBLogger("runs/find_wood_symbolic", json_path="runs/.../training_metrics.json")
    # Also writes training_metrics.csv and fitness_overview.png next to the JSON.
    for gen in range(generations):
        ...
        logger.log_generation(gen, fitnesses)
    logger.close()
"""

import csv
import json
import os

import numpy as np
from tensorboardX import SummaryWriter

# Headless matplotlib for the fitness_overview.png file. Lazily guarded so
# environments without matplotlib still work (the overview PNG is skipped).
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - diagnostics only
    _HAS_MPL = False


class TBLogger:
    def __init__(
        self,
        log_dir: str = "runs/training",
        json_path: str | None = None,
        overview_png_path: str | None = None,
    ):
        """TensorBoard logger + standalone overview PNG.

        ``overview_png_path`` is a regular PNG on the filesystem that is
        **overwritten every generation** with a worst/mean/best plot. It is
        never embedded in tfevents (TensorBoard's log is append-only, so
        "overwriting" there would just keep growing the file on disk).

        Defaults to ``<run_dir>/fitness_overview.png`` when ``json_path`` is
        provided; set ``overview_png_path=False`` to disable the PNG entirely.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to {log_dir}")

        self.json_path = json_path
        self.csv_path = os.path.splitext(json_path)[0] + ".csv" if json_path else None
        self._metrics_generations = [] if json_path else None
        if json_path:
            print(f"Metrics JSON: {json_path}")
            print(f"Metrics CSV:  {self.csv_path}")

        # Running history for the overwritten overview PNG.
        self._gen_history: list[int] = []
        self._best_history: list[float] = []
        self._mean_history: list[float] = []
        self._worst_history: list[float] = []

        # Resolve overview PNG path: explicit > derived from json_path > none.
        if overview_png_path is False:
            resolved_png = None
        elif overview_png_path is None and json_path is not None:
            resolved_png = os.path.join(
                os.path.dirname(os.path.abspath(json_path)),
                "fitness_overview.png",
            )
        else:
            resolved_png = overview_png_path
        self.overview_png_path = resolved_png
        if self.overview_png_path:
            print(f"Overview PNG (overwritten per gen): {self.overview_png_path}")
        elif not _HAS_MPL:
            print("matplotlib not available: fitness overview PNG disabled")

    _CSV_BASE_FIELDS = [
        "generation",
        "fitness/best",
        "fitness/mean",
        "fitness/worst",
        "fitness/std",
        "fitness/median",
    ]

    def _flush_metrics_files(self):
        if not self.json_path:
            return
        out_dir = os.path.dirname(os.path.abspath(self.json_path))
        os.makedirs(out_dir, exist_ok=True)

        payload = {
            "schema": "training_metrics_v1",
            "generations": self._metrics_generations,
        }
        tmp_json = self.json_path + ".tmp"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_json, self.json_path)

        extra_keys = set()
        base_set = set(self._CSV_BASE_FIELDS)
        for row in self._metrics_generations:
            extra_keys |= set(row.keys()) - base_set
        fieldnames = self._CSV_BASE_FIELDS + sorted(extra_keys)

        tmp_csv = self.csv_path + ".tmp"
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._metrics_generations)
        os.replace(tmp_csv, self.csv_path)

    def log_generation(self, gen, fitnesses, extra=None):
        """Log standard ES metrics for one generation.

        Args:
            gen: generation number
            fitnesses: array-like of fitness values for the population
            extra: optional dict of additional scalar metrics to log
        """
        f = np.asarray(fitnesses)
        best = float(f.max())
        worst = float(f.min())
        mean = float(f.mean())
        self.writer.add_scalar("fitness/best", best, gen)
        self.writer.add_scalar("fitness/mean", mean, gen)
        self.writer.add_scalar("fitness/worst", worst, gen)
        self.writer.add_scalar("fitness/std", float(f.std()), gen)
        self.writer.add_scalar("fitness/median", float(np.median(f)), gen)
        # Cheap scalar proxy for population diversity (complements the PNG).
        self.writer.add_scalar("fitness/spread", best - worst, gen)

        if extra:
            for key, value in extra.items():
                self.writer.add_scalar(key, float(value), gen)

        self._gen_history.append(int(gen))
        self._best_history.append(best)
        self._mean_history.append(mean)
        self._worst_history.append(worst)
        self._overwrite_overview_png()

        if self.json_path is not None:
            row = {
                "generation": int(gen),
                "fitness/best": float(f.max()),
                "fitness/mean": float(f.mean()),
                "fitness/worst": float(f.min()),
                "fitness/std": float(f.std()),
                "fitness/median": float(np.median(f)),
            }
            if extra:
                for key, value in extra.items():
                    row[key] = float(value)
            self._metrics_generations.append(row)
            self._flush_metrics_files()

    def _overwrite_overview_png(self) -> None:
        """Render worst/best band + mean line, overwriting the PNG file.

        TensorBoard is deliberately not touched here: tfevents is append-only,
        so re-logging the figure every generation would bloat TB startup time.
        """
        if not _HAS_MPL or not self.overview_png_path or not self._gen_history:
            return
        gens = self._gen_history
        worst = self._worst_history
        best = self._best_history
        mean = self._mean_history

        fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=90)
        ax.fill_between(
            gens,
            worst,
            best,
            color="tab:blue",
            alpha=0.20,
            linewidth=0,
            label="worst–best",
        )
        ax.plot(gens, best, color="tab:green", linewidth=1.0, label="best")
        ax.plot(gens, worst, color="tab:red", linewidth=1.0, label="worst")
        ax.plot(gens, mean, color="tab:blue", linewidth=1.6, label="mean")
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness")
        ax.set_title("Population fitness (worst / mean / best)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout(pad=0.4)

        try:
            os.makedirs(os.path.dirname(self.overview_png_path), exist_ok=True)
            tmp = self.overview_png_path + ".tmp"
            fig.savefig(tmp, format="png")
            os.replace(tmp, self.overview_png_path)
        except Exception as exc:  # pragma: no cover - I/O edge cases
            print(f"warning: failed to save overview PNG: {exc}")
        finally:
            plt.close(fig)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, float(value), step)

    def log_histogram(self, tag, values, step, bins="tensorflow"):
        """Log a histogram (e.g. discrete meta-action ids or returns)."""
        arr = np.asarray(values)
        self.writer.add_histogram(tag, arr, step, bins=bins)

    def close(self):
        if self.json_path:
            self._flush_metrics_files()
        self.writer.close()
