#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import cv2


def format_timestamp_ms(ms: float) -> str:
    """Format milliseconds to HH_MM_SS_mmm."""
    ms = max(0.0, ms)
    total_ms = int(round(ms))
    h = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    m = rem // 60_000
    rem %= 60_000
    s = rem // 1_000
    mmm = rem % 1_000
    return f"{h:02d}_{m:02d}_{s:02d}_{mmm:03d}"


def main():
    ap = argparse.ArgumentParser(description="Extract frames and save as FrameNumber_Timestamp images.")
    ap.add_argument("--in", dest="inp", required=True, help="Input video path")
    ap.add_argument("--out-dir", default="frames", help="Folder to save frames (default: frames)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png",
                    help="Image format to save (default: png)")
    ap.add_argument("--prefix", default="frame", help="Filename prefix (default: frame)")
    ap.add_argument("--every", type=int, default=1,
                    help="Save every Nth frame (default: 1 = save all frames)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Resize factor (e.g., 0.5 for half size). Default: 1.0")
    ap.add_argument("--start-ms", type=float, default=0.0,
                    help="Optional offset to add to timestamps (milliseconds). Default: 0")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise SystemExit(f"Input video not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {in_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # sometimes 0 depending on codec
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.every <= 0:
        raise SystemExit("--every must be >= 1")

    pad = max(6, len(str(total_frames))) if total_frames > 0 else 6

    print("=== Frame Extraction ===")
    print(f"Input:        {in_path}")
    print(f"Output dir:   {out_dir.resolve()}")
    print(f"Format:       {args.format}")
    print(f"Every Nth:    {args.every}")
    print(f"Scale:        {args.scale}")
    print(f"FPS (info):   {fps}")
    print(f"Total frames: {total_frames if total_frames > 0 else 'Unknown'}")
    print("Name format:  PREFIX_FRAME#_HH_MM_SS_mmm")
    print("========================\n")

    start = time.perf_counter()
    read_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        read_idx += 1

        # Save every Nth frame
        if (read_idx - 1) % args.every != 0:
            continue

        if args.scale != 1.0:
            new_w = int(frame.shape[1] * args.scale)
            new_h = int(frame.shape[0] * args.scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Timestamp from decode position
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) + args.start_ms
        stamp = format_timestamp_ms(t_ms)

        # Filename = frame_<index>_<timestamp>.png
        filename = f"{args.prefix}_{saved:0{pad}d}_{stamp}.{args.format}"
        out_path = out_dir / filename

        if args.format == "jpg":
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(str(out_path), frame)

        saved += 1

        if total_frames > 0:
            print(f"\rSaved {saved} images | Reading frame {read_idx}/{total_frames}", end="")
        else:
            print(f"\rSaved {saved} images | Reading frame {read_idx}", end="")

    cap.release()
    elapsed = time.perf_counter() - start
    print("\n\nDone.")
    print(f"Saved images: {saved}")
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
