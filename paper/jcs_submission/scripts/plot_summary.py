"""Regenerate the summary figure from the versioned paper plot data."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plot_data.csv"
OUT = ROOT / "figures"


def load_rows():
    with DATA.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows, dataset, condition, methods):
    chosen = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition]
    by_method = {r["method"]: r for r in chosen}
    baseline = float(by_method["data_only"]["baseline_rel_l2"])
    return [
        (method, 100.0 * (baseline - float(by_method[method]["mean_rel_l2"])) / baseline)
        for method in methods
    ]


def main():
    rows = load_rows()
    panels = [
        ("Burgers", "locked_holdout", ["PDE", "gradient", "curvature"],
         ["PDE", "Gradient", "Curvature"]),
        ("Allen-Cahn", "dense", ["full_curvature", "interval", "pointwise"],
         ["Full curvature", "Interval", "Pointwise"]),
        ("Allen-Cahn", "sparse", ["full_curvature", "interval", "pointwise"],
         ["Full curvature", "Interval", "Pointwise"]),
    ]
    colors = {"PDE": "#2878B5", "gradient": "#55A868", "curvature": "#C44E52",
              "full_curvature": "#8172B3", "interval": "#DD8452", "pointwise": "#64B5CD"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.25), constrained_layout=True)
    for ax, (dataset, condition, methods, labels) in zip(axes, panels):
        values = select(rows, dataset, condition, methods)
        xs = list(range(len(values)))
        gains = [v for _, v in values]
        bars = ax.bar(xs, gains, color=[colors[m] for m in methods], width=0.68,
                      edgecolor="#333333", linewidth=0.45)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(xs, labels, rotation=23, ha="right")
        ax.set_ylabel("Improvement over data only (%)")
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title({
            ("Burgers", "locked_holdout"): "(a) Burgers holdout",
            ("Allen-Cahn", "dense"): "(b) Allen-Cahn, dense",
            ("Allen-Cahn", "sparse"): "(c) Allen-Cahn, sparse",
        }[(dataset, condition)], fontsize=9)
        low, high = min(gains), max(gains)
        pad = max(0.8, (high - low) * 0.18)
        ax.set_ylim(min(-0.7, low - pad), high + pad)
        for rect, val in zip(bars, gains):
            if val >= 0:
                y, va = val + 0.12, "bottom"
            else:
                y, va = val - 0.12, "top"
            ax.text(rect.get_x() + rect.get_width() / 2, y, f"{val:+.2f}",
                    ha="center", va=va, fontsize=7.5)
    fig.suptitle("Field-error improvement; panels are normalized within each condition", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "relative_gains.pdf", bbox_inches="tight")
    fig.savefig(OUT / "relative_gains.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
