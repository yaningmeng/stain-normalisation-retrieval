#!/usr/bin/env python3
"""
plism_loader.py — fetch a small, reproducible subset of the PLISM tile dataset.

PLISM (https://huggingface.co/datasets/owkin/plism-dataset-tiles, CC BY 4.0) is
a tissue microarray of 46 human organs, stained under 13 conditions and scanned
on 7 scanners: 91 whole-slide images cut into 16,278 tiles each, 1,481,298
tiles in total, about 146 GB. Every slide carries the identical list of
``tile_id`` values in the identical order, so "the same tile" is comparable
across all 91 slides.

Nobody should download 146 GB. This script reads only the parquet row groups it
actually needs, over HTTP, and writes one HDF5 file per slide.

Three things about this dataset are easy to get wrong, and all three are
handled here.

1.  Shards do not all hold the same number of rows (some 5,055, some 5,056), so
    arithmetic that assumes a fixed shard size silently drifts by a whole
    slide. This script reads the real row counts from the parquet footers and
    then verifies that the total is exactly 91 x 16,278 before trusting them.

2.  Row groups hold about 99 rows and are transferred whole, so 400 scattered
    tiles would cost 400 row groups while 4 contiguous blocks of 100 cost about
    8. Tiles are therefore selected as contiguous blocks. (This is also why
    pulling the official scattered 460-tile benchmark subset costs essentially
    the entire dataset.)

3.  The 7 scanners image the SAME physical section, but the 13 staining
    conditions are SERIAL SECTIONS — different physical slices of the block,
    with a reported median registration error of ~43 microns against a
    112-micron tile. Comparisons across staining conditions therefore mix
    colour with genuine morphological difference. Use ``--stainer`` to stay on
    the clean cross-scanner axis.

Usage
-----
    # once, a few minutes: read the footer of every shard
    python plism_loader.py build-index

    # what is actually in the dataset?
    python plism_loader.py summarise

    # smoke test: 4 slides, 100 tiles each
    python plism_loader.py fetch --tiles-per-slide 100 --max-slides 4 --out-dir tiles_smoke

    # the clean cross-scanner axis: one staining condition, all 7 scanners
    python plism_loader.py fetch --tiles-per-slide 400 --stainer GMH --out-dir tiles_gmh

    # the full 91-slide design
    python plism_loader.py fetch --tiles-per-slide 400 --out-dir tiles
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "owkin/plism-dataset-tiles"
N_SHARDS = 293
N_SLIDES = 91
TILES_PER_SLIDE = 16278

DEFAULT_INDEX = Path("plism_index.parquet")
DEFAULT_SLIDES = Path("plism_slides.csv")

# Roughly 99 rows of ~113 KB each; used only to print an estimate.
MB_PER_ROW_GROUP = 11


# --------------------------------------------------------------------------- #
# remote access
# --------------------------------------------------------------------------- #

def _filesystem():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _shard_path(shard: int) -> str:
    return f"datasets/{REPO}/data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"


# --------------------------------------------------------------------------- #
# index: where does each row group sit in the global row stream?
# --------------------------------------------------------------------------- #

def _shard_row_groups(shard: int) -> tuple[int, list[int]]:
    """Row-group sizes of one shard, read from the parquet footer only.

    A footer read costs about a second. Reading a shard's metadata *columns*
    costs nearly two minutes, because that is 51 scattered range requests — so
    this keeps ``build-index`` at minutes rather than hours.
    """
    fs = _filesystem()
    with fs.open(_shard_path(shard), "rb") as handle:
        meta = pq.ParquetFile(handle).metadata
        return shard, [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]


def build_index(index_path: Path, slides_path: Path, workers: int = 8) -> None:
    """Record the global position of every row group, then identify the slides."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    sizes: dict[int, list[int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_shard_row_groups, s) for s in range(N_SHARDS)]
        for future in tqdm(futures, desc="reading shard footers", unit="shard"):
            shard, row_groups = future.result()
            sizes[shard] = row_groups

    records, cursor = [], 0
    for shard in range(N_SHARDS):
        for row_group, n_rows in enumerate(sizes[shard]):
            records.append((shard, row_group, cursor, n_rows))
            cursor += n_rows

    index = pd.DataFrame(records, columns=["shard", "row_group", "global_start", "num_rows"])

    expected = N_SLIDES * TILES_PER_SLIDE
    if cursor != expected:
        raise RuntimeError(
            f"total rows {cursor:,} != {N_SLIDES} x {TILES_PER_SLIDE} = {expected:,}. "
            "The layout assumption behind this loader is wrong; do not trust any "
            "subset it produces."
        )

    index.to_parquet(index_path, index=False)
    print(f"wrote {index_path}: {len(index):,} row groups, {cursor:,} rows verified")

    slides = _read_slide_table(index)
    slides.to_csv(slides_path, index=False)
    print(f"wrote {slides_path}: {len(slides)} slides")


def _locate(index: pd.DataFrame, global_rows: np.ndarray) -> pd.DataFrame:
    """Global row numbers -> the row group holding each one."""
    starts = index.global_start.to_numpy()
    pos = np.searchsorted(starts, global_rows, side="right") - 1
    return pd.DataFrame(
        {
            "global_row": global_rows,
            "shard": index.shard.to_numpy()[pos],
            "row_group": index.row_group.to_numpy()[pos],
            "group_start": starts[pos],
        }
    )


def _read_slide_table(index: pd.DataFrame) -> pd.DataFrame:
    """Identify all 91 slides by reading one small row group per slide."""
    from tqdm import tqdm

    fs = _filesystem()
    located = _locate(index, np.arange(N_SLIDES, dtype=np.int64) * TILES_PER_SLIDE)

    rows = []
    for rank, entry in tqdm(
        list(located.iterrows()), desc="identifying slides", unit="slide", total=N_SLIDES
    ):
        with fs.open(_shard_path(int(entry.shard)), "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(
                int(entry.row_group), columns=["slide_id", "stainer", "scanner"]
            )
        record = table.to_pandas().iloc[int(entry.global_row - entry.group_start)]
        rows.append((rank, record.slide_id, record.stainer, record.scanner))

    return pd.DataFrame(rows, columns=["slide_rank", "slide_id", "stainer", "scanner"])


def load_index(index_path: Path, slides_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not index_path.exists() or not slides_path.exists():
        sys.exit(
            f"No index yet. Build it once with:\n    python {Path(__file__).name} build-index"
        )
    return pd.read_parquet(index_path), pd.read_csv(slides_path)


# --------------------------------------------------------------------------- #
# subset selection
# --------------------------------------------------------------------------- #

def select_tile_ranks(n_tiles: int, block: int = 100) -> np.ndarray:
    """Pick tile ranks as evenly spread CONTIGUOUS blocks, to keep transfer small."""
    n_tiles = min(n_tiles, TILES_PER_SLIDE)
    n_blocks = max(1, int(round(n_tiles / block)))
    block = int(np.ceil(n_tiles / n_blocks))
    starts = np.linspace(0, TILES_PER_SLIDE - block, n_blocks).astype(int)
    return np.unique(np.concatenate([np.arange(s, s + block) for s in starts]))[:n_tiles]


def build_plan(
    index: pd.DataFrame,
    slides: pd.DataFrame,
    tiles_per_slide: int,
    stainer: str | None,
    scanner: str | None,
    max_slides: int | None,
) -> pd.DataFrame:
    chosen = slides
    if stainer:
        chosen = chosen[chosen.stainer == stainer]
    if scanner:
        chosen = chosen[chosen.scanner == scanner]
    if chosen.empty:
        sys.exit("No slides matched that --stainer / --scanner combination.")
    if max_slides:
        chosen = chosen.head(max_slides)

    ranks = select_tile_ranks(tiles_per_slide)
    grid = chosen.assign(key=1).merge(
        pd.DataFrame({"tile_rank": ranks, "key": 1}), on="key"
    ).drop(columns="key")
    grid["global_row"] = grid.slide_rank * TILES_PER_SLIDE + grid.tile_rank

    plan = grid.merge(_locate(index, grid.global_row.to_numpy()), on="global_row")

    n_groups = plan.groupby(["shard", "row_group"]).ngroups
    print(
        f"{chosen.slide_id.nunique()} slides x {len(ranks)} tiles = {len(plan):,} images, "
        f"touching {n_groups:,} row groups "
        f"(~{n_groups * MB_PER_ROW_GROUP / 1024:.1f} GB of transfer)"
    )
    return plan


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

def _decode(cell) -> np.ndarray:
    from PIL import Image

    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def fetch(plan: pd.DataFrame, out_dir: Path) -> None:
    import h5py
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    fs = _filesystem()
    collected: dict[str, dict[str, np.ndarray]] = {}

    groups = list(plan.groupby(["shard", "row_group"]))
    for (shard, row_group), rows in tqdm(groups, desc="reading row groups", unit="rg"):
        with fs.open(_shard_path(int(shard)), "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(
                int(row_group), columns=["slide_id", "tile_id", "png"]
            )
        frame = table.to_pandas()
        start = int(rows.group_start.iloc[0])

        for _, want in rows.iterrows():
            record = frame.iloc[int(want.global_row - start)]
            if record.slide_id != want.slide_id:
                raise RuntimeError(
                    f"expected {want.slide_id} at global row {want.global_row} "
                    f"but found {record.slide_id}. Rebuild the index."
                )
            collected.setdefault(record.slide_id, {})[record.tile_id] = _decode(record.png)

    meta = plan.drop_duplicates("slide_id").set_index("slide_id")
    for slide_id, tiles in sorted(collected.items()):
        tile_ids = sorted(tiles)
        stack = np.stack([tiles[t] for t in tile_ids])
        with h5py.File(out_dir / f"{slide_id}.h5", "w") as h5:
            h5.create_dataset("tiles", data=stack, compression="gzip", compression_opts=1)
            h5.create_dataset("tile_id", data=np.array(tile_ids, dtype="S32"))
            h5.attrs["slide_id"] = slide_id
            h5.attrs["stainer"] = str(meta.loc[slide_id, "stainer"])
            h5.attrs["scanner"] = str(meta.loc[slide_id, "scanner"])
        print(f"  {slide_id}.h5  {stack.shape}")

    (
        plan.drop_duplicates("slide_id")[["slide_id", "stainer", "scanner"]]
        .sort_values("slide_id")
        .to_csv(out_dir / "slides.csv", index=False)
    )
    print(f"\nwrote {len(collected)} slides to {out_dir}/")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def summarise(slides: pd.DataFrame) -> None:
    print(f"{len(slides)} slides, {TILES_PER_SLIDE:,} tiles each\n")
    print(f"stainers ({slides.stainer.nunique()}): {sorted(slides.stainer.unique())}")
    print(f"scanners ({slides.scanner.nunique()}): {sorted(slides.scanner.unique())}\n")
    print("slides per (stainer, scanner) cell:")
    print(pd.crosstab(slides.stainer, slides.scanner))
    print(
        "\nThe 7 scanners image the same physical section: that axis is clean.\n"
        "The 13 stainers are serial sections: colour AND morphology differ there."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--slides", type=Path, default=DEFAULT_SLIDES)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-index", help="read every shard footer once (a few minutes)")
    sub.add_parser("summarise", help="describe the experimental design")

    get = sub.add_parser("fetch", help="download a tile subset")
    get.add_argument("--tiles-per-slide", type=int, default=400)
    get.add_argument("--stainer", default=None, help="restrict to one staining condition")
    get.add_argument("--scanner", default=None, help="restrict to one scanner")
    get.add_argument("--max-slides", type=int, default=None, help="cap slides, for smoke tests")
    get.add_argument("--out-dir", type=Path, default=Path("tiles"))

    args = parser.parse_args()

    if args.command == "build-index":
        build_index(args.index, args.slides)
        return

    index, slides = load_index(args.index, args.slides)

    if args.command == "summarise":
        summarise(slides)
    elif args.command == "fetch":
        plan = build_plan(
            index, slides, args.tiles_per_slide, args.stainer, args.scanner, args.max_slides
        )
        fetch(plan, args.out_dir)


if __name__ == "__main__":
    main()
