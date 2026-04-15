"""
Video Frame Prediction Visualiser
For every .mp4 in test-videos/:
  - Sample 100 random frames
  - For each frame produce a 4-panel figure:
      Panel 1 — Original frame
      Panel 2 — CLAHE-enhanced equivalent
      Panel 3 — Predicted mask (grayscale)
      Panel 4 — Original frame with pipe highlighted in yellow
  - Save as  video_predictions/<video_stem>/frame_XXXXX.png

Usage:
    python3 predict_video_frames.py
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────── CONFIGURATION ────────────────────────────────── #
MODEL_PATH  = 'models-new-final/best_pipe_unet.pth'
VIDEO_DIR   = 'test-videos'
OUTPUT_DIR  = 'video_predictions'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE    = 512
THRESHOLD   = 0.5
N_FRAMES    = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────── MODEL ────────────────────────────────────────── #
def load_model():
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        classes=1,
        activation='sigmoid',
        decoder_attention_type='scse',
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────── UNDERWATER ENHANCEMENT PIPELINE ─────────────────────── #
# Exact pipeline from tuning-pipeline.py:
# WB (LAB A/B shift) → Red boost → CLAHE → Dehaze → Sharpen → Gamma

ENHANCE_PARAMS = {
    "A_SHIFT":       18,
    "B_SHIFT":        0,
    "OMEGA":       0.90,
    "CLAHE_CLIP":  2.5,
    "CLAHE_GRID":    8,
    "RED_STRENGTH": 0.4,
    "T_MIN":       0.35,
    "DARK_R":        15,
    "GAMMA":        1.1,
}

def _white_balance_lab(img_bgr, a_shift, b_shift):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = cv2.split(lab)
    A = np.clip(A + int(a_shift), 0, 255).astype(np.uint8)
    B = np.clip(B + int(b_shift), 0, 255).astype(np.uint8)
    L = np.clip(L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def _red_boost(img_bgr, red_strength):
    b, g, r = cv2.split(img_bgr)
    r_eq  = cv2.equalizeHist(r)
    r_out = cv2.addWeighted(r, 1.0 - red_strength, r_eq, red_strength, 0.0)
    return cv2.merge([b, g, r_out])

def _clahe_l(img_bgr, clip_limit, tile_grid):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                             tileGridSize=(int(tile_grid), int(tile_grid)))
    return cv2.cvtColor(cv2.merge([clahe.apply(L), A, B]), cv2.COLOR_LAB2BGR)

def _min_filter(gray, r):
    k = 2 * r + 1
    return cv2.erode(gray, np.ones((k, k), np.uint8))

def _dark_channel(img_bgr, r):
    b, g, rc = cv2.split(img_bgr)
    return _min_filter(cv2.min(cv2.min(b, g), rc), r)

def _dehaze(img_bgr, omega, t_min, r):
    I    = img_bgr.astype(np.float32) / 255.0
    dark = _dark_channel((I * 255).astype(np.uint8), r).astype(np.float32) / 255.0
    flat_dark = dark.reshape(-1)
    flat_I    = I.reshape(-1, 3)
    n   = max(50, int(0.001 * flat_dark.size))
    idx = np.argsort(flat_dark)[-n:]
    A   = flat_I[idx].mean(axis=0)
    A_safe  = np.maximum(A, 1e-6)
    I_norm  = I / A_safe
    dark_n  = _dark_channel(
        (np.clip(I_norm, 0, 1) * 255).astype(np.uint8), r
    ).astype(np.float32) / 255.0
    t = np.clip(1.0 - omega * dark_n, t_min, 1.0)
    J = np.clip((I - A) / t[..., None] + A, 0.0, 1.0)
    return (J * 255).astype(np.uint8)

def _sharpen(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    return np.clip(cv2.addWeighted(img_bgr, 1.2, blur, -0.2, 0), 0, 255).astype(np.uint8)

def _gamma(img_bgr, g):
    inv   = 1.0 / max(0.01, g)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)

def enhance_underwater(img_bgr):
    """Full 6-step underwater enhancement pipeline (tuning-pipeline.py defaults)."""
    p = ENHANCE_PARAMS
    out = _white_balance_lab(img_bgr, p["A_SHIFT"], p["B_SHIFT"])
    out = _red_boost(out, p["RED_STRENGTH"])
    out = _clahe_l(out, p["CLAHE_CLIP"], p["CLAHE_GRID"])
    out = _dehaze(out, p["OMEGA"], p["T_MIN"], p["DARK_R"])
    out = _sharpen(out)
    out = _gamma(out, p["GAMMA"])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ─────────────────────────── PREPROCESSING ────────────────────────────────── #
def preprocess(img_bgr):
    """Returns (img_rgb_512, tensor_for_model)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(resized).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return resized, x.to(DEVICE)


# ─────────────────────────── YELLOW OVERLAY ───────────────────────────────── #
def yellow_overlay(img_rgb, mask_bin, alpha=0.55):
    overlay = img_rgb.copy().astype(np.float32)
    yellow  = np.array([255, 220, 0], dtype=np.float32)
    overlay[mask_bin.astype(bool)] = (
        (1 - alpha) * overlay[mask_bin.astype(bool)] + alpha * yellow
    )
    return overlay.clip(0, 255).astype(np.uint8)


# ─────────────────────────── FRAME SAMPLER ────────────────────────────────── #
def sample_frames(video_path, n):
    """Return list of (frame_index, frame_bgr) tuples, randomly sampled."""
    cap        = cv2.VideoCapture(str(video_path))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sample   = min(n, total)
    indices    = sorted(random.sample(range(total), n_sample))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames, total


# ─────────────────────────── SAVE ONE FIGURE ──────────────────────────────── #
def save_figure(img_rgb, enhanced_rgb, mask_bin, overlay_rgb,
                out_path, frame_idx, video_name):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#1a1a1a')

    panels = [
        (img_rgb,           None,   'Original Frame'),
        (enhanced_rgb,      None,   'Enhanced (WB+CLAHE+Dehaze)'),
        (mask_bin * 255,    'gray', 'Predicted Mask'),
        (overlay_rgb,       None,   'Pipe Overlay (yellow)'),
    ]

    for ax, (im, cmap, title) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.axis('off')
        ax.set_title(title, color='white', fontsize=12,
                     fontweight='bold', pad=7)

    fig.suptitle(f'{video_name}  —  frame {frame_idx:06d}',
                 color='#aaaaaa', fontsize=9, y=0.02)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────── MAIN ─────────────────────────────────────────── #
def main():
    print(f"Loading model from {MODEL_PATH}  (device: {DEVICE})")
    model = load_model()

    videos = sorted(Path(VIDEO_DIR).glob('*.mp4'))
    if not videos:
        print(f"No .mp4 files found in {VIDEO_DIR}/"); return

    print(f"Found {len(videos)} video(s). Sampling {N_FRAMES} frames each.\n")

    for video_path in videos:
        stem    = video_path.stem
        out_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(out_dir, exist_ok=True)

        print(f"── {video_path.name}")
        frames, total_frames = sample_frames(video_path, N_FRAMES)
        print(f"   {total_frames} total frames  →  sampling {len(frames)}")

        for j, (frame_idx, frame_bgr) in enumerate(frames, 1):
            img_rgb, x   = preprocess(frame_bgr)
            # Enhance the already-resized frame so it matches the prediction resolution
            frame_bgr_512 = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
            enhanced_rgb  = enhance_underwater(frame_bgr_512)

            with torch.no_grad():
                prob     = model(x).squeeze().cpu().numpy()
            mask_bin = (prob > THRESHOLD).astype(np.uint8)
            overlay  = yellow_overlay(img_rgb, mask_bin)

            out_path = os.path.join(out_dir, f'frame_{frame_idx:06d}.png')
            save_figure(img_rgb, enhanced_rgb, mask_bin, overlay,
                        out_path, frame_idx, stem)

            if j % 10 == 0 or j == len(frames):
                print(f"   [{j:3d}/{len(frames)}] frame {frame_idx:06d} → {out_path}")

        print(f"   Done — {len(frames)} images saved to {out_dir}/\n")

    print(f"All videos processed. Results in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
