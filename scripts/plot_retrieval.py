"""
plot_retrieval.py — Bar chart of paired top-1 retrieval accuracy with 95% CI.

Produces Figure 2: two-panel bar chart comparing normalisation methods
for Phikon-v2 and Midnight-12k. Error bars show 95% confidence intervals
of the mean across paired comparisons (273 cross-scanner, 546 cross-staining).

Usage:
    python plot_retrieval.py --out figures/fig2.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Mean accuracy (%) and 95% CI half-widths from paired comparisons
# CI = 1.96 * std / sqrt(n); n=273 cross-scanner, n=546 cross-staining
DATA = {
    "cross_scanner": {
        "Phikon-v2":    {"mean": [70.12, 72.90, 51.82], "ci": [1.55, 1.67, 2.22]},
        "Midnight-12k": {"mean": [56.96, 55.90, 51.35], "ci": [1.32, 1.44, 1.59]},
    },
    "cross_staining": {
        "Phikon-v2":    {"mean": [33.78, 32.06, 24.03], "ci": [1.21, 1.25, 1.12]},
        "Midnight-12k": {"mean": [29.02, 26.76, 25.75], "ci": [0.88, 0.85, 0.97]},
    },
}
CONDITIONS = ["No normalisation", "Reinhard", "Macenko"]


def plot_panel(ax, data, title, y_max):
    x = np.arange(len(CONDITIONS))
    width = 0.35

    bars1 = ax.bar(x - width / 2, data["Phikon-v2"]["mean"], width,
                   yerr=data["Phikon-v2"]["ci"], capsize=3,
                   label="Phikon-v2", color="#2166AC", alpha=0.8,
                   error_kw={"linewidth": 1.0})
    bars2 = ax.bar(x + width / 2, data["Midnight-12k"]["mean"], width,
                   yerr=data["Midnight-12k"]["ci"], capsize=3,
                   label="Midnight-12k", color="#B2182B", alpha=0.8,
                   error_kw={"linewidth": 1.0})

    ax.set_ylabel("Top-1 retrieval accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=9)
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/fig2.pdf")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_panel(ax1, DATA["cross_scanner"], "(a) Cross-scanner", 90)
    plot_panel(ax2, DATA["cross_staining"], "(b) Cross-staining", 50)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
