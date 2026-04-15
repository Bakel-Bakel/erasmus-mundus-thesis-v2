"""
Prediction Visualiser
Loads the trained U-Net, picks 20 random training images, and saves a
3-panel figure for each:
  Panel 1 — Original image
  Panel 2 — Predicted mask (grayscale)
  Panel 3 — Original image with pipe highlighted in yellow
Output: prediction_samples/sample_001.png … sample_020.png
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
IMAGE_DIR   = 'merged_dataset/images/Train'
OUTPUT_DIR  = 'prediction_samples'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE    = 512
THRESHOLD   = 0.5
N_SAMPLES   = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ─────────────────────────── PREPROCESSING ────────────────────────────────── #
def preprocess(img_bgr):
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
    """Blend a yellow colour over predicted pipe pixels."""
    overlay = img_rgb.copy().astype(np.float32)
    yellow  = np.array([255, 220, 0], dtype=np.float32)
    pipe    = mask_bin.astype(bool)
    overlay[pipe] = (1 - alpha) * overlay[pipe] + alpha * yellow
    return overlay.clip(0, 255).astype(np.uint8)

# ─────────────────────────── MAIN ─────────────────────────────────────────── #
def main():
    print(f"Loading model from {MODEL_PATH}  (device: {DEVICE})")
    model = load_model()

    all_images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    sample_names = random.sample(all_images, min(N_SAMPLES, len(all_images)))
    print(f"Generating {len(sample_names)} prediction samples → {OUTPUT_DIR}/\n")

    for i, name in enumerate(sample_names, 1):
        img_path = os.path.join(IMAGE_DIR, name)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [skip] could not read {name}")
            continue

        img_rgb, x = preprocess(img_bgr)

        with torch.no_grad():
            prob     = model(x).squeeze().cpu().numpy()
        mask_bin = (prob > THRESHOLD).astype(np.uint8)

        overlay = yellow_overlay(img_rgb, mask_bin)

        # ── 3-panel figure ──
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#1a1a1a')

        titles = ['Original Image', 'Predicted Mask', 'Pipe Overlay (yellow)']
        images = [img_rgb, mask_bin * 255, overlay]
        cmaps  = [None, 'gray', None]

        for ax, title, im, cmap in zip(axes, titles, images, cmaps):
            ax.imshow(im, cmap=cmap)
            ax.axis('off')
            ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

        stem = Path(name).stem
        fig.suptitle(stem, color='#aaaaaa', fontsize=9, y=0.02)
        plt.tight_layout(pad=0.5)

        out_path = os.path.join(OUTPUT_DIR, f'sample_{i:03d}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [{i:02d}/{len(sample_names)}] {out_path}  ({stem})")

    print(f"\nDone. {len(sample_names)} images saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
