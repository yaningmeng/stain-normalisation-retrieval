"""
plot_umap.py — UMAP projection of slide-level mean embeddings.

Produces Figure 3: two-panel UMAP coloured by scanner and staining condition,
with centroid markers sized by within-group spread.

Usage:
    python plot_umap.py \
        --feature-dir features/midnight_none \
        --out figures/fig3.pdf
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP


def load_slide_embeddings(feature_dir):
    """Compute mean embedding per slide from tile-level features."""
    files = sorted(Path(feature_dir).glob("*.h5"))
    embeddings, scanners, stainers = [], [], []

    for f in files:
        with h5py.File(f, "r") as h:
            embeddings.append(h["features"][:].mean(axis=0))
            scanners.append(str(h.attrs["scanner"]))
            stainers.append(str(h.attrs["stainer"]))

    return np.stack(embeddings), scanners, stainers


def compute_spread(points):
    """RMS distance from centroid."""
    centroid = points.mean(axis=0)
    return np.sqrt(((points - centroid) ** 2).sum(axis=1).mean())


def plot_umap(X_2d, labels, ax, title, cmap_name="Set1"):
    """Scatter plot with centroid markers sized by spread."""
    unique = sorted(set(labels))
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, len(unique)))

    for i, label in enumerate(unique):
        mask = [l == label for l in labels]
        pts = X_2d[mask]
        centroid = pts.mean(axis=0)
        spread = compute_spread(pts)

        ax.scatter(pts[:, 0], pts[:, 1], s=15, color=colors[i], alpha=0.6)
        ax.scatter(
            centroid[0], centroid[1],
            s=spread * 200, color=colors[i],
            edgecolors="black", linewidths=0.5,
            label=f"{label} ({spread:.1f})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Scanner (spread)" if "scanner" in title.lower() else "Stainer (spread)",
              fontsize=7, loc="upper right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out", default="figures/fig3.pdf")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, scanners, stainers = load_slide_embeddings(args.feature_dir)
    print(f"{X.shape[0]} slides, {X.shape[1]}-d embeddings")

    reducer = UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    X_2d = reducer.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_umap(X_2d, scanners, ax1, "(a) Coloured by scanner")
    plot_umap(X_2d, stainers, ax2, "(b) Coloured by staining condition")

    # Print summary statistics
    print("\n=== Scanner centroids ===")
    centroids = []
    for s in sorted(set(scanners)):
        pts = X_2d[[sc == s for sc in scanners]]
        c = pts.mean(axis=0)
        centroids.append(c)
        print(f"  {s}: spread={compute_spread(pts):.2f}")

    centroids = np.stack(centroids)
    from itertools import combinations
    dists = [np.linalg.norm(centroids[i] - centroids[j])
             for i, j in combinations(range(len(centroids)), 2)]
    print(f"  Mean inter-centroid distance: {np.mean(dists):.2f}")
    spreads = [compute_spread(X_2d[[sc == s for sc in scanners]])
               for s in sorted(set(scanners))]
    print(f"  Mean within-group spread: {np.mean(spreads):.2f}")

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=300)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
