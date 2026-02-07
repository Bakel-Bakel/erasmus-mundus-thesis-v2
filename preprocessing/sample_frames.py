#!/usr/bin/env python3
"""
Uniformly sample K images from a folder of frames named like:
  frame_000358_00_11_56_716.jpg

Sampling is done by frame index (the 000358 part), equally spaced across the dataset.
Outputs are copied into an output folder.

Usage:
  python sample_frames.py --in-dir /path/to/LTsealine --out-dir /path/to/sample_200 --count 200
"""

import argparse
import re
import shutil
from pathlib import Path
import numpy as np
import time

FRAME_RE = re.compile(r"frame_(\d+)_")  # captures the frame number after "frame_"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_frame_number(p: Path) -> int:
    m = FRAME_RE.search(p.name)
    if not m:
        raise ValueError(f"Could not parse frame number from filename: {p.name}")
    return int(m.group(1))


def list_sorted_frames(in_dir: Path):
    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"No images found in {in_dir}")

    # Sort by extracted frame number, then name as tie-breaker
    files.sort(key=lambda p: (extract_frame_number(p), p.name))
    return files


def uniform_sample_indices(n: int, k: int):
    """
    Returns k indices from [0, n-1] uniformly spaced.
    Uses linspace then rounds to nearest, then unique-ifies while preserving order.
    """
    if k <= 0:
        raise ValueError("--count must be > 0")
    if k >= n:
        return list(range(n))

    raw = np.linspace(0, n - 1, num=k)
    idx = np.round(raw).astype(int).tolist()

    # Ensure uniqueness in case rounding causes duplicates
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            out.append(i)
            seen.add(i)

    # If uniqueness reduced count, fill missing by stepping through gaps
    if len(out) < k:
        needed = k - len(out)
        # pick extra indices from the remaining ones, spread out
        remaining = [i for i in range(n) if i not in seen]
        extra = np.round(np.linspace(0, len(remaining) - 1, num=needed)).astype(int)
        for j in extra:
            out.append(remaining[int(j)])

        out.sort()

    return out


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError("mode must be 'copy' or 'symlink'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path, help="Folder containing extracted frames")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output folder to write sampled frames")
    ap.add_argument("--count", required=True, type=int, help="Number of images to sample")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                    help="copy = duplicate files, symlink = lightweight links (Linux)")
    ap.add_argument("--prefix", default="", help="Optional prefix added to output filenames")
    args = ap.parse_args()

    in_dir = args.in_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    files = list_sorted_frames(in_dir)
    n = len(files)

    k = args.count
    idxs = uniform_sample_indices(n, k)
    sampled = [files[i] for i in idxs]

    print(f"Found {n} images in {in_dir}")
    print(f"Sampling {len(sampled)} images uniformly -> {out_dir} (mode={args.mode})")

    for i, src in enumerate(sampled, start=1):
        # keep original filename, optionally with prefix
        dst_name = f"{args.prefix}{src.name}" if args.prefix else src.name
        dst = out_dir / dst_name
        copy_or_link(src, dst, args.mode)

        if i == 1 or i % 50 == 0 or i == len(sampled):
            fn = extract_frame_number(src)
            print(f"[{i}/{len(sampled)}] frame={fn}  ->  {dst.name}")

    dt = time.time() - t0
    first_fn = extract_frame_number(sampled[0])
    last_fn = extract_frame_number(sampled[-1])
    print(f"Done in {dt:.2f}s. Frame range in sample: {first_fn} .. {last_fn}")


if __name__ == "__main__":
    main()
