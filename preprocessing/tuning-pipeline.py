#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np


# --------------------------
# Core pipeline pieces
# --------------------------

def white_balance_lab(img_bgr: np.ndarray, a_shift: int, b_shift: int) -> np.ndarray:
    """LAB white balance using A/B channel shifting (matches the blog-style approach)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = cv2.split(lab)

    A = np.clip(A + int(a_shift), 0, 255).astype(np.uint8)
    B = np.clip(B + int(b_shift), 0, 255).astype(np.uint8)
    L = np.clip(L, 0, 255).astype(np.uint8)

    lab2 = cv2.merge([L, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def red_boost(img_bgr: np.ndarray, red_strength_0_1: float) -> np.ndarray:
    """Boost red channel by blending original red with equalized red."""
    rs = float(np.clip(red_strength_0_1, 0.0, 1.0))
    b, g, r = cv2.split(img_bgr)
    r_eq = cv2.equalizeHist(r)
    r_out = cv2.addWeighted(r, 1.0 - rs, r_eq, rs, 0.0)
    return cv2.merge([b, g, r_out])


def clahe_l_channel(img_bgr: np.ndarray, clip_limit: float, tile_grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel in LAB."""
    clip_limit = float(max(0.0, clip_limit))
    tile_grid = int(max(2, tile_grid))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _min_filter(gray: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    return cv2.erode(gray, np.ones((k, k), np.uint8))


def _dark_channel(img_bgr: np.ndarray, r: int) -> np.ndarray:
    b, g, rch = cv2.split(img_bgr)
    m = cv2.min(cv2.min(b, g), rch)
    return _min_filter(m, r)


def dehaze_dark_channel(img_bgr: np.ndarray, omega: float, t_min: float = 0.15, r: int = 7) -> np.ndarray:
    """
    Dark-channel-inspired dehaze.
    omega: strength (0..1)
    t_min: prevents extreme amplification
    r: dark channel radius
    """
    omega = float(np.clip(omega, 0.0, 1.0))
    t_min = float(np.clip(t_min, 0.01, 1.0))
    r = int(max(1, r))

    I = img_bgr.astype(np.float32) / 255.0
    dark = _dark_channel((I * 255).astype(np.uint8), r=r).astype(np.float32) / 255.0

    # Atmospheric light A from brightest dark-channel pixels
    flat_dark = dark.reshape(-1)
    flat_I = I.reshape(-1, 3)

    n = max(50, int(0.001 * flat_dark.size))  # top 0.1% (min 50 pixels)
    idx = np.argsort(flat_dark)[-n:]
    A = flat_I[idx].mean(axis=0)  # (3,)

    A_safe = np.maximum(A, 1e-6)
    I_norm = I / A_safe
    dark_norm = _dark_channel((np.clip(I_norm, 0, 1) * 255).astype(np.uint8), r=r).astype(np.float32) / 255.0

    t = 1.0 - omega * dark_norm
    t = np.clip(t, t_min, 1.0)

    J = (I - A) / t[..., None] + A
    J = np.clip(J, 0.0, 1.0)
    return (J * 255).astype(np.uint8)


def sharpen_unsharp(img_bgr: np.ndarray) -> np.ndarray:
    """Matches the blog snippet: blur 3x3, addWeighted(img, 1.2, blur, -0.2, 0)."""
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    out = cv2.addWeighted(img_bgr, 1.2, blur, -0.2, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correct(img_bgr: np.ndarray, g: float) -> np.ndarray:
    """Matches the blog snippet: LUT with inv = 1/g."""
    g = float(max(0.01, g))
    inv = 1.0 / g
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def enhance_underwater(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """
    Pipeline order matching the blog layout you screenshotted:
    WB(LAB) -> Red boost -> CLAHE -> Dehaze -> Sharpen -> Gamma
    """
    out = img_bgr
    out = white_balance_lab(out, params["A_SHIFT"], params["B_SHIFT"])
    out = red_boost(out, params["RED_STRENGTH"])
    out = clahe_l_channel(out, params["CLAHE_CLIP"], tile_grid=params["CLAHE_GRID"])
    out = dehaze_dark_channel(out, params["OMEGA"], t_min=params["T_MIN"], r=params["DARK_R"])
    out = sharpen_unsharp(out)
    out = gamma_correct(out, params["GAMMA"])
    return out


# --------------------------
# Trackbar mapping (like the blog)
# --------------------------
""" 
        "A_SHIFT": 0,           # -50..+50
        "B_SHIFT": -10,         # -50..+50 (negative often helps blue/green cast)
        "OMEGA": 0.90,          # 0..1
        "CLAHE_CLIP": 2.0,      # 0..5
        "CLAHE_GRID": 8,        # tile grid size
        "RED_STRENGTH": 0.35,   # 0..1
        "T_MIN": 0.15,          # 0.01..1
        "DARK_R": 7,            # 5..15 typical
        "GAMMA": 1.10,  


        "A_SHIFT": 18,           # -50..+50
        "B_SHIFT": 0,         # -50..+50 (negative often helps blue/green cast)
        "OMEGA": 0.50,          # 0..1
        "CLAHE_CLIP": 4.0,      # 0..5
        "CLAHE_GRID": 8,        # tile grid size
        "RED_STRENGTH": 0.74,   # 0..1
        "T_MIN": 1.00,          # 0.01..1
        "DARK_R": 25,            # 5..15 typical
        "GAMMA": 0.98, """ 
def make_default_params():
    # Reasonable defaults close to typical underwater tuning.
    return {
        "A_SHIFT": 18,           # -50..+50
        "B_SHIFT": 0,         # -50..+50 (negative often helps blue/green cast)
        "OMEGA": 0.90,          # 0..1
        "CLAHE_CLIP": 2.5,      # 0..5
        "CLAHE_GRID": 8,        # tile grid size
        "RED_STRENGTH": 0.4,   # 0..1
        "T_MIN": 0.35,          # 0.01..1
        "DARK_R": 15,            # 5..15 typical
        "GAMMA": 1.1, 
                # >1 brightens shadows more than highlights
    }


def read_params_from_trackbars(window_name: str, params: dict) -> dict:
    # 0..100 mapped to -50..+50
    params["A_SHIFT"] = cv2.getTrackbarPos("A Shift", window_name) - 50
    params["B_SHIFT"] = cv2.getTrackbarPos("B Shift", window_name) - 50

    # 0..100 mapped to 0..1
    params["OMEGA"] = cv2.getTrackbarPos("Omega", window_name) / 100.0

    # 0..50 mapped to 0..5
    params["CLAHE_CLIP"] = cv2.getTrackbarPos("CLAHE Clip", window_name) / 10.0

    # 0..100 mapped to 0..1
    params["RED_STRENGTH"] = cv2.getTrackbarPos("Red Boost", window_name) / 100.0

    # extra knobs (optional but useful)
    params["T_MIN"] = max(0.01, cv2.getTrackbarPos("t_min x100", window_name) / 100.0)  # 0.01..1
    params["DARK_R"] = max(1, cv2.getTrackbarPos("Dark r", window_name))                 # >=1
    params["GAMMA"] = max(0.10, cv2.getTrackbarPos("Gamma x100", window_name) / 100.0)   # >=0.1

    return params


def setup_trackbars(window_name: str, params: dict):
    def _noop(x):  # callback required by OpenCV
        pass

    cv2.createTrackbar("A Shift", window_name, params["A_SHIFT"] + 50, 100, _noop)
    cv2.createTrackbar("B Shift", window_name, params["B_SHIFT"] + 50, 100, _noop)
    cv2.createTrackbar("Omega", window_name, int(params["OMEGA"] * 100), 100, _noop)
    cv2.createTrackbar("CLAHE Clip", window_name, int(params["CLAHE_CLIP"] * 10), 50, _noop)
    cv2.createTrackbar("Red Boost", window_name, int(params["RED_STRENGTH"] * 100), 100, _noop)

    # Extra useful controls
    cv2.createTrackbar("t_min x100", window_name, int(params["T_MIN"] * 100), 100, _noop)
    cv2.createTrackbar("Dark r", window_name, int(params["DARK_R"]), 25, _noop)          # up to 25
    cv2.createTrackbar("Gamma x100", window_name, int(params["GAMMA"] * 100), 300, _noop) # up to 3.00


def draw_params_overlay(img: np.ndarray, params: dict) -> np.ndarray:
    """Draw parameter values as text overlay on the image."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    
    # Create semi-transparent dark background for text (more opaque for better visibility)
    overlay_bg = overlay.copy()
    cv2.rectangle(overlay_bg, (10, 10), (350, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, overlay_bg, 0.5, 0, overlay)
    
    # Text settings - using white text with black outline for maximum visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    outline_color = (0, 0, 0)  # Black outline
    thickness = 2
    outline_thickness = 3
    line_height = 25
    y_start = 35
    
    # Draw parameter labels and values
    lines = [
        f"A Shift: {params['A_SHIFT']:+3d}",
        f"B Shift: {params['B_SHIFT']:+3d}",
        f"Omega: {params['OMEGA']:.2f}",
        f"CLAHE Clip: {params['CLAHE_CLIP']:.2f}",
        f"CLAHE Grid: {params['CLAHE_GRID']}",
        f"Red Boost: {params['RED_STRENGTH']:.2f}",
        f"t_min: {params['T_MIN']:.2f}",
        f"Dark r: {params['DARK_R']}",
        f"Gamma: {params['GAMMA']:.2f}",
    ]
    
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        # Draw black outline first for better visibility
        cv2.putText(overlay, line, (15, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        # Then draw white text on top
        cv2.putText(overlay, line, (15, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return overlay


# --------------------------
# Folder processing helpers
# --------------------------

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def process_folder(in_dir: Path, out_dir: Path, params: dict, ext: str = "jpg", quality: int = 95):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in in_dir.rglob("*") if p.is_file() and is_image_file(p)])
    if not images:
        raise SystemExit(f"No images found in: {in_dir}")

    t0 = time.time()
    for i, p in enumerate(images, start=1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] unreadable: {p}")
            continue

        enhanced = enhance_underwater(img, params)

        rel = p.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix("." + ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if ext == "jpg":
            cv2.imwrite(str(out_path), enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))])
        else:
            cv2.imwrite(str(out_path), enhanced)

        if i % 50 == 0 or i == 1 or i == len(images):
            print(f"[{i}/{len(images)}] saved {out_path.name}")

    dt = time.time() - t0
    print(f"Done. Processed {len(images)} images in {dt:.1f}s ({len(images)/max(dt,1e-6):.2f} img/s).")


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-image", help="Path to one representative image for tuning (recommended)")
    ap.add_argument("--in-dir", help="Folder of images to batch enhance")
    ap.add_argument("--out-dir", help="Output folder for enhanced images")
    ap.add_argument("--save-params", default="enhance_params.json", help="Where to save tuned params JSON")
    ap.add_argument("--load-params", help="Load params JSON instead of tuning")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--quality", type=int, default=95)

    args = ap.parse_args()

    params = make_default_params()

    # Load params if provided
    if args.load_params:
        with open(args.load_params, "r", encoding="utf-8") as f:
            params.update(json.load(f))
        print("Loaded params:", params)

    # Interactive tuning
    if args.tune_image:
        img = cv2.imread(args.tune_image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read tune image: {args.tune_image}")

        win_enh = "Enhanced"
        win_org = "Original"

        cv2.namedWindow(win_enh, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_enh, 1000, 700)
        cv2.namedWindow(win_org, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_org, 1000, 700)

        setup_trackbars(win_enh, params)

        print("Tuning controls:")
        print("- Adjust sliders until it looks like the blog result you like.")
        print("- Press 's' to SAVE params, 'q' to quit.")

        while True:
            params = read_params_from_trackbars(win_enh, params)
            enhanced = enhance_underwater(img, params)
            
            # Add parameter overlay to enhanced image
            enhanced_with_overlay = draw_params_overlay(enhanced, params)

            cv2.imshow(win_org, img)
            cv2.imshow(win_enh, enhanced_with_overlay)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                with open(args.save_params, "w", encoding="utf-8") as f:
                    json.dump(params, f, indent=2)
                print("Saved params to:", args.save_params)
                print(params)

        cv2.destroyAllWindows()

    # Batch processing
    if args.in_dir and args.out_dir:
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        process_folder(in_dir, out_dir, params, ext=args.ext, quality=args.quality)
    else:
        if not args.tune_image and not args.load_params:
            print("Nothing to do. Provide --tune-image and/or --in-dir + --out-dir.")


if __name__ == "__main__":
    main()
