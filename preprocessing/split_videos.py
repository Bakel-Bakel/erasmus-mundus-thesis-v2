#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffprobe_duration_seconds(input_path: str) -> float:
    """
    Returns duration in seconds using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        input_path
    ]
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    data = json.loads(p.stdout)
    dur = float(data["format"]["duration"])
    return dur

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def format_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def split_video_stream_copy(
    input_path: str,
    out_dir: str,
    total_minutes: float | None,
    chunk_minutes: float,
    prefix: str,
) -> None:
    in_path = Path(input_path)
    out_dir_p = Path(out_dir)
    ensure_dir(out_dir_p)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    ext = in_path.suffix  # keep same container extension
    if ext == "":
        # fallback if no extension; you can choose one, but we default to .mp4
        ext = ".mp4"

    total_sec_video = ffprobe_duration_seconds(str(in_path))
    chunk_sec = float(chunk_minutes) * 60.0
    if chunk_sec <= 0:
        raise ValueError("--chunk-minutes must be > 0")

    # how much of the video to process
    if total_minutes is None:
        total_sec_process = total_sec_video
    else:
        total_sec_process = min(float(total_minutes) * 60.0, total_sec_video)

    if total_sec_process <= 0:
        raise ValueError("Nothing to process (total minutes resolved to 0).")

    n_chunks = int(math.ceil(total_sec_process / chunk_sec))

    print("=== Split Job ===")
    print(f"Input:             {in_path}")
    print(f"Container ext:     {ext}")
    print(f"Video duration:    {format_hms(total_sec_video)} ({total_sec_video:.2f}s)")
    print(f"Process duration:  {format_hms(total_sec_process)} ({total_sec_process:.2f}s)")
    print(f"Chunk size:        {format_hms(chunk_sec)} ({chunk_sec:.2f}s)")
    print(f"Output directory:  {out_dir_p.resolve()}")
    print(f"Chunks to create:  {n_chunks}")
    print("=================\n")

    start_time = time.perf_counter()

    for i in range(n_chunks):
        seg_start = i * chunk_sec
        seg_len = min(chunk_sec, total_sec_process - seg_start)
        if seg_len <= 0:
            break

        out_name = f"{prefix}_{i+1:04d}{ext}"
        out_path = out_dir_p / out_name

        print(f"[{i+1}/{n_chunks}] Splitting chunk {i+1}...")
        print(f"  - start: {format_hms(seg_start)}  len: {format_hms(seg_len)}")
        print(f"  - out:   {out_path.name}")

        # Stream copy (no re-encode), keep original codec/quality.
        # -ss after -i for accurate seeking (required for stream copy to work correctly)
        # This is slower but ensures all chunks are created properly.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(in_path),
            "-ss", str(seg_start),
            "-t", str(seg_len),
            "-map", "0",          # include all streams (video/audio/subs) if possible
            "-c", "copy",         # preserve codec
            "-avoid_negative_ts", "make_zero",
            str(out_path)
        ]

        p = run_cmd(cmd)
        if p.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed on chunk {i+1}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{p.stderr}"
            )

    elapsed = time.perf_counter() - start_time
    print("\nDone.")
    print(f"Total runtime: {elapsed:.2f} seconds ({format_hms(elapsed)})")

def main():
    ap = argparse.ArgumentParser(
        description="Split a video into chunks while preserving codec/container via FFmpeg stream copy."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input video path")
    ap.add_argument("--out-dir", default="splits", help="Output directory (default: splits)")
    ap.add_argument("--total-minutes", type=float, default=None,
                    help="How many minutes of the input to split (from start). Omit to split full video.")
    ap.add_argument("--chunk-minutes", type=float, required=True,
                    help="Length of each smaller video in minutes (e.g. 1 for 1-minute chunks)")
    ap.add_argument("--prefix", default="part", help="Output filename prefix (default: part)")
    args = ap.parse_args()

    split_video_stream_copy(
        input_path=args.inp,
        out_dir=args.out_dir,
        total_minutes=args.total_minutes,
        chunk_minutes=args.chunk_minutes,
        prefix=args.prefix,
    )

if __name__ == "__main__":
    main()
