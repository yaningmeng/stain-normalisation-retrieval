#!/usr/bin/env python3
"""
extract_features.py — turn PLISM tiles into foundation-model embeddings,
optionally recolouring the images first.

Reads the per-slide HDF5 files written by ``plism_loader.py`` and writes one
HDF5 of features per slide. The point of the ``--normalise`` flag is that the
SAME tiles can be encoded with and without image-space stain normalisation, so
the two runs are directly comparable.

Encoders (both ungated: no HuggingFace token, no licence to accept)
-------------------------------------------------------------------
  phikon_v2   owkin/phikon-v2     1024-d   CLS token
              Ranked #15 of 16 on the PLISM robustness leaderboard, and one of
              the five encoders used in this laboratory's existing work.

  midnight    kaiko-ai/midnight   3072-d   concat(CLS, mean patch tokens)
              Midnight-12k. Ranked #6 of 16 — the best ungated model on that
              leaderboard — and MIT licensed.

Both are DINOv2-family models loaded through ``transformers``. Note that
``timm`` is not involved: it is a separate library used by UNI-v2 and Virchow,
and it is not part of TRIDENT (TRIDENT merely depends on it).

Stain normalisation uses torch-staintools rather than torchstain, because it is
the only one of the two that implements Vahadane — the method behind this
laboratory's own published stain-vector analysis — and because it normalises a
whole batch on the GPU, so tiles never leave the device between normalisation
and encoding.

Usage
-----
    python extract_features.py --tiles-dir tiles --model phikon_v2 \
        --normalise none     --out-dir features/phikon_v2_raw

    python extract_features.py --tiles-dir tiles --model phikon_v2 \
        --normalise macenko  --out-dir features/phikon_v2_macenko

    python extract_features.py --tiles-dir tiles --model phikon_v2 \
        --normalise vahadane --out-dir features/phikon_v2_vahadane

    python extract_features.py --tiles-dir tiles --model midnight \
        --normalise none     --out-dir features/midnight_raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #

ENCODERS = {
    "phikon_v2": {
        "repo": "owkin/phikon-v2",
        "mean": (0.485, 0.456, 0.406),      # ImageNet statistics
        "std": (0.229, 0.224, 0.225),
        "pool": "cls",
        "dim": 1024,
    },
    "midnight": {
        "repo": "kaiko-ai/midnight",
        "mean": (0.5, 0.5, 0.5),            # stated on the model card
        "std": (0.5, 0.5, 0.5),
        "pool": "cls_plus_mean",
        "dim": 3072,
    },
}


def load_encoder(name: str, device: torch.device):
    from transformers import AutoModel

    spec = ENCODERS[name]
    model = AutoModel.from_pretrained(spec["repo"]).to(device).eval()
    return model, spec


def pool_tokens(hidden: torch.Tensor, how: str) -> torch.Tensor:
    """hidden: (B, 1 + n_patches, D) -> (B, D) or (B, 2D)."""
    if how == "cls":
        return hidden[:, 0, :]
    if how == "cls_plus_mean":
        return torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=-1)
    raise ValueError(how)


# --------------------------------------------------------------------------- #
# stain normalisation
# --------------------------------------------------------------------------- #

def build_normaliser(method: str, target: np.ndarray, device: torch.device):
    """Fit a torch-staintools normaliser to one reference tile.

    torch-staintools is used rather than torchstain for two reasons: it is the
    only one of the two that implements Vahadane — the method this laboratory's
    own published stain-vector analysis uses — and it normalises a whole batch
    on the GPU, so tiles never leave the device between normalisation and
    encoding. ``target`` is HWC uint8; the library wants BCHW float in [0, 1].
    """
    if method == "none":
        return None

    from torch_staintools.normalizer import NormalizerBuilder

    kwargs = {}
    if method in {"macenko", "vahadane"}:
        kwargs["concentration_solver"] = "qr"

    normaliser = NormalizerBuilder.build(method, **kwargs).to(device)
    reference = (
        torch.from_numpy(target).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    )
    normaliser.fit(reference)
    return normaliser


def apply_normaliser(normaliser, batch: torch.Tensor) -> tuple[torch.Tensor, int]:
    """batch: BCHW float in [0, 1] -> same, plus a count of tiles left unchanged.

    Stain normalisation genuinely fails on some tiles: nearly blank background
    has no stain vectors to estimate, and the result comes back non-finite.
    Silently dropping those tiles would give the normalised and unnormalised
    runs different tile counts for the same slide, which quietly invalidates
    every paired comparison built on top. So the original is kept and counted.
    """
    if normaliser is None:
        return batch, 0

    try:
        out = normaliser(batch)
    except Exception:                                    # noqa: BLE001
        return batch, len(batch)

    bad = ~torch.isfinite(out).all(dim=(1, 2, 3))
    if bad.any():
        out = out.clone()
        out[bad] = batch[bad]
    return out.clamp(0.0, 1.0), int(bad.sum())


# --------------------------------------------------------------------------- #
# main pass
# --------------------------------------------------------------------------- #

def reference_tile(tiles_dir: Path, slide: str | None) -> np.ndarray:
    """One fixed reference tile, so every slide is normalised towards the same target."""
    files = sorted(tiles_dir.glob("*.h5"))
    if not files:
        sys.exit(f"No .h5 tile files in {tiles_dir}. Run plism_loader.py fetch first.")
    chosen = next((f for f in files if f.stem == slide), files[0])
    with h5py.File(chosen, "r") as h5:
        tile = h5["tiles"][0]
    print(f"stain reference: first tile of {chosen.stem}")
    return np.asarray(tile)


@torch.no_grad()
def encode_slide(
    path: Path,
    model,
    spec: dict,
    normaliser,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(path, "r") as h5:
        tiles = h5["tiles"][:]
        tile_ids = h5["tile_id"][:]

    mean = torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], device=device).view(1, 3, 1, 1)

    features, n_unchanged = [], 0
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        batch = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float().div_(255.0)

        batch, n_bad = apply_normaliser(normaliser, batch)
        n_unchanged += n_bad

        batch = (batch - mean) / std
        hidden = model(batch).last_hidden_state
        features.append(pool_tokens(hidden, spec["pool"]).float().cpu().numpy())

    return np.concatenate(features), tile_ids, n_unchanged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(ENCODERS), default="phikon_v2")
    parser.add_argument("--normalise", choices=["none", "macenko", "vahadane", "reinhard"], default="none")
    parser.add_argument("--reference-slide", default=None, help="slide whose first tile is the stain target")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  normalise={args.normalise}  device={device}")
    model, spec = load_encoder(args.model, device)
    normaliser = (
        build_normaliser(args.normalise, reference_tile(args.tiles_dir, args.reference_slide), device)
        if args.normalise != "none"
        else None
    )

    slides = sorted(args.tiles_dir.glob("*.h5"))
    total_unchanged = 0

    for path in slides:
        features, tile_ids, n_unchanged = encode_slide(
            path, model, spec, normaliser, device, args.batch_size
        )
        total_unchanged += n_unchanged

        out = args.out_dir / path.name
        with h5py.File(out, "w") as h5, h5py.File(path, "r") as src:
            h5.create_dataset("features", data=features.astype(np.float32))
            h5.create_dataset("tile_id", data=tile_ids)
            h5.attrs["slide_id"] = src.attrs["slide_id"]
            h5.attrs["stainer"] = src.attrs["stainer"]
            h5.attrs["scanner"] = src.attrs["scanner"]
            h5.attrs["model"] = args.model
            h5.attrs["normalise"] = args.normalise
        note = "" if args.normalise == "none" else f"  ({n_unchanged} tiles unchanged by normaliser)"
        print(f"  {out.name}  {features.shape}{note}")

    if args.normalise != "none":
        print(f"\n{total_unchanged} tiles could not be normalised and were left as-is.")
        print("They are still encoded, so paired comparisons keep matching tile counts.")
    print(f"\nwrote {len(slides)} feature files to {args.out_dir}/")


if __name__ == "__main__":
    main()
