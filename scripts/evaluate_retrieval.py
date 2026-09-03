"""
evaluate_retrieval.py — Paired top-1 nearest-neighbour retrieval evaluation.

Given a directory of per-slide HDF5 feature files (produced by extract_features.py),
computes cross-scanner and cross-staining retrieval accuracy.

Usage:
    python evaluate_retrieval.py \
        --feature-dir features/phikon_v2_none \
        --run-name phikon_v2_none \
        --out-dir results
"""

import argparse
import glob
import itertools

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_embeddings(feature_dir, device="cuda"):
    """Load and L2-normalise per-slide embeddings from HDF5 files."""
    files = sorted(glob.glob(f"{feature_dir}/*.h5"))
    assert len(files) == 91, f"Expected 91 files, found {len(files)}"

    embeddings = {}
    reference_tile_ids = None

    for file in files:
        with h5py.File(file, "r") as h:
            stainer = str(h.attrs["stainer"])
            scanner = str(h.attrs["scanner"])
            tile_ids = h["tile_id"][:]

            if reference_tile_ids is None:
                reference_tile_ids = tile_ids
            else:
                assert np.array_equal(reference_tile_ids, tile_ids)

            x = torch.from_numpy(h["features"][:]).to(device)
            embeddings[(stainer, scanner)] = F.normalize(x, dim=1)

    return embeddings, reference_tile_ids


def compute_retrieval(embeddings, reference_tile_ids, device="cuda"):
    """Compute paired top-1 retrieval for cross-scanner and cross-staining axes."""
    stainers = sorted({k[0] for k in embeddings})
    scanners = sorted({k[1] for k in embeddings})
    truth = torch.arange(len(reference_tile_ids), device=device)

    def score(a, b):
        similarity = a @ b.T
        forward = (similarity.argmax(dim=1) == truth).float().mean().item()
        reverse = (similarity.argmax(dim=0) == truth).float().mean().item()
        return forward, reverse, (forward + reverse) / 2

    results = []

    with torch.inference_mode():
        # Cross-scanner: fix stainer, vary scanner
        for stainer in stainers:
            for a, b in itertools.combinations(scanners, 2):
                fwd, rev, mean = score(
                    embeddings[(stainer, a)], embeddings[(stainer, b)]
                )
                results.append({
                    "axis": "cross-scanner",
                    "fixed_condition": stainer,
                    "condition_a": a,
                    "condition_b": b,
                    "forward": fwd,
                    "reverse": rev,
                    "top1_mean": mean,
                })

        # Cross-staining: fix scanner, vary stainer
        for scanner in scanners:
            for a, b in itertools.combinations(stainers, 2):
                fwd, rev, mean = score(
                    embeddings[(a, scanner)], embeddings[(b, scanner)]
                )
                results.append({
                    "axis": "cross-staining",
                    "fixed_condition": scanner,
                    "condition_a": a,
                    "condition_b": b,
                    "forward": fwd,
                    "reverse": rev,
                    "top1_mean": mean,
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    embeddings, tile_ids = load_embeddings(args.feature_dir, device)
    df = compute_retrieval(embeddings, tile_ids, device)
    df["top1_percent"] = df["top1_mean"] * 100

    summary = df.groupby("axis")["top1_percent"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    print(f"\n{args.run_name}")
    print(summary.round(2))

    df.to_csv(f"{args.out_dir}/{args.run_name}.csv", index=False)
    print(f"Saved to {args.out_dir}/{args.run_name}.csv")


if __name__ == "__main__":
    main()
