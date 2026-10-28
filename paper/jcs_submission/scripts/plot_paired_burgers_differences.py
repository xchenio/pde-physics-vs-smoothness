"""Export and plot paired trajectory-level differences from the locked Burgers holdout."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper" / "jcs_submission"
RUNS = ROOT / "results" / "fno_regularization_confirmation_v1" / "runs"
OUT = PAPER / "figures"
CSV_OUT = PAPER / "paired_burgers_differences.csv"


def load_seed_means():
    arms = {name: [] for name in ("none", "physics", "gradient", "curvature")}
    identity = None
    for arm in arms:
        for seed in range(3):
            path = RUNS / f"{arm}_s{seed}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["seed"] != seed:
                raise ValueError(f"Unexpected seed in {path}")
            current = (record["identity"]["plan"], record["identity"]["data_sha256"])
            if identity is None:
                identity = current
            elif current != identity:
                raise ValueError("Run files do not share a plan and confirmation dataset")
            values = np.asarray(record["per_sample"], dtype=float)
            if values.shape != (500,) or not np.isfinite(values).all():
                raise ValueError(f"Expected 500 finite trajectory errors in {path}")
            arms[arm].append(values)
    means = {name: np.mean(values, axis=0) for name, values in arms.items()}
    return means, identity


def main():
    means, identity = load_seed_means()
    deltas = {
        "physics_minus_none": means["physics"] - means["none"],
        "physics_minus_gradient": means["physics"] - means["gradient"],
        "physics_minus_curvature": means["physics"] - means["curvature"],
    }
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["confirmation_order", *deltas.keys()])
        for i in range(500):
            writer.writerow([i + 1, *(f"{values[i]:.12g}" for values in deltas.values())])

    colors = {
        "physics_minus_none": "#2878B5",
        "physics_minus_gradient": "#55A868",
        "physics_minus_curvature": "#C44E52",
    }
    labels = {
        "physics_minus_none": "PDE $-$ data only",
        "physics_minus_gradient": "PDE $-$ gradient",
        "physics_minus_curvature": "PDE $-$ curvature",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    all_values = np.concatenate(list(deltas.values()))
    span = float(all_values.max() - all_values.min())
    for name, values in deltas.items():
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        mean = float(values.mean())
        ax.step(x, y, where="post", color=colors[name], linewidth=1.65,
                label=f"{labels[name]} (mean {mean:+.4f})")
        ax.axvline(mean, color=colors[name], linestyle="--", linewidth=0.9, alpha=0.75)
    ax.axvline(0.0, color="#222222", linewidth=0.9)
    ax.set_xlabel("Per-trajectory relative-$L^2$ difference (PDE minus comparator)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xlim(float(all_values.min() - 0.05 * span),
                float(all_values.max() + 0.05 * span))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Locked Burgers holdout: paired differences across 500 trajectories", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "paired_burgers_differences.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paired_burgers_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plan hash: {identity[0]}")
    print(f"Data SHA-256: {identity[1]}")
    for name, values in deltas.items():
        print(f"{name}: mean={values.mean():+.8f}; fraction(PDE better)={(values < 0).mean():.3f}")
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
