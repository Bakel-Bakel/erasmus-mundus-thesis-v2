"""
U-Net Pipe Segmentation — Training Script with Full Metrics Tracking
Produces after every run:
  training_results/
    01_loss_curves.png
    02_iou_dice_curves.png
    03_lr_schedule.png
    04_precision_recall_curves.png
    05_f1_specificity_curves.png
    06_metric_heatmap.png
    07_best_epoch_confusion_matrix.png
    08_gradient_norm_curve.png
    09_train_val_gap.png
    10_per_epoch_summary_table.png
    training_report.md
    training_history.json
"""

import os
import json
import time
import random
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# ─────────────────────────── CONFIGURATION ────────────────────────────────── #
ENCODER          = 'resnet34'
ENCODER_WEIGHTS  = 'imagenet'
ACTIVATION       = 'sigmoid'
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
LR               = 0.0001
EPOCHS           = 100
IMG_SIZE         = 512
BATCH_SIZE       = 8
VAL_SPLIT        = 0.2
RANDOM_SEED      = 42
NUM_WORKERS      = 4 if torch.cuda.is_available() else 0

IMAGE_DIR        = 'merged_dataset/images/Train'
MASK_DIR         = 'merged_dataset/masks/Train'
OUTPUT_DIR       = 'training_results-final'
CHECKPOINT_DIR   = 'models-new-final'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

STYLE = {
    'train_color': '#2196F3',
    'val_color':   '#F44336',
    'grid_alpha':  0.25,
    'fig_dpi':     150,
    'title_size':  13,
    'label_size':  11,
}


# ─────────────────────────── DATASET ──────────────────────────────────────── #
class PipeDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None):
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.images_fps, self.masks_fps = [], []
        for f in image_files:
            img_path  = os.path.join(images_dir, f)
            mask_path = os.path.join(masks_dir, Path(f).stem + '.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.images_fps.append(img_path)
                self.masks_fps.append(mask_path)
        self.augmentation = augmentation
        print(f"  Dataset: {len(self.images_fps)} valid image-mask pairs")

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.masks_fps[i], 0)
        mask  = np.where(mask > 128, 1.0, 0.0).astype(np.float32)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


def get_augmentation(train=True):
    if train:
        transforms = [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    return A.Compose(transforms)


# ─────────────────────────── METRICS ──────────────────────────────────────── #
def batch_metrics(pred_prob, masks, threshold=0.5):
    """Returns dict of scalar metrics for one batch."""
    pred = (pred_prob > threshold).float()
    gt   = masks

    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()

    iou         = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice        = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision   = (tp + 1e-6) / (tp + fp + 1e-6)
    recall      = (tp + 1e-6) / (tp + fn + 1e-6)
    f1          = 2 * precision * recall / (precision + recall + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        'iou': iou, 'dice': dice, 'precision': precision,
        'recall': recall, 'f1': f1, 'specificity': specificity,
        'accuracy': accuracy, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def avg_metrics(metric_list):
    keys = ['iou', 'dice', 'precision', 'recall', 'f1', 'specificity', 'accuracy']
    result = {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
    # Also sum pixel-level CM totals across all batches
    for cm_key in ['tp', 'fp', 'fn', 'tn']:
        result[cm_key] = float(sum(m[cm_key] for m in metric_list))
    return result


# ─────────────────────────── TRAIN / VAL LOOPS ────────────────────────────── #
def run_epoch(model, loader, optimizer, loss_fn, train, device, grad_clip=1.0):
    model.train() if train else model.eval()
    total_loss, all_metrics, grad_norms = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss  = loss_fn(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norms.append(norm.item())
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                all_metrics.append(batch_metrics(preds.detach(), masks))

    avg = avg_metrics(all_metrics)
    avg['loss'] = total_loss / len(loader)
    if train:
        avg['grad_norm'] = float(np.mean(grad_norms)) if grad_norms else 0.0
    return avg


# ═══════════════════════════ PLOTTING FUNCTIONS ═══════════════════════════════ #
def fig_save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    plt.close(fig)
    return path


def _ep(history):
    return list(range(1, len(history['train_loss']) + 1))


# ── 1. Loss Curves ────────────────────────────────────────────────────────── #
def plot_loss_curves(history):
    ep = _ep(history)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, history['train_loss'], color=STYLE['train_color'], lw=2, label='Train Loss')
    ax.plot(ep, history['val_loss'],   color=STYLE['val_color'],   lw=2, label='Val Loss')

    best_ep = int(np.argmin(history['val_loss'])) + 1
    ax.axvline(best_ep, color='gray', ls='--', lw=1.5, alpha=0.7,
               label=f'Best val epoch = {best_ep}')

    ax.set_title('Training & Validation Loss', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice Loss')
    ax.legend(); ax.grid(alpha=STYLE['grid_alpha'])
    plt.tight_layout()
    return fig_save(fig, '01_loss_curves.png')


# ── 2. IoU & Dice Curves ──────────────────────────────────────────────────── #
def plot_iou_dice_curves(history):
    ep = _ep(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('IoU and Dice Score Curves', fontsize=STYLE['title_size'], fontweight='bold')

    for ax, key, title in [(axes[0], 'iou', 'IoU'), (axes[1], 'dice', 'Dice')]:
        ax.plot(ep, history[f'train_{key}'], color=STYLE['train_color'], lw=2, label=f'Train {title}')
        ax.plot(ep, history[f'val_{key}'],   color=STYLE['val_color'],   lw=2, label=f'Val {title}')
        best_ep = int(np.argmax(history[f'val_{key}'])) + 1
        ax.axvline(best_ep, color='gray', ls='--', lw=1.5, alpha=0.7,
                   label=f'Best val epoch = {best_ep}')
        ax.set_title(title, fontsize=STYLE['title_size'])
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=STYLE['grid_alpha'])

    plt.tight_layout()
    return fig_save(fig, '02_iou_dice_curves.png')


# ── 3. Learning Rate Schedule ─────────────────────────────────────────────── #
def plot_lr_schedule(history):
    ep = _ep(history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(ep, history['lr'], color='#9C27B0', lw=2, marker='o', ms=3)
    ax.set_title('Learning Rate Schedule', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate (log scale)')
    ax.grid(alpha=STYLE['grid_alpha'])
    plt.tight_layout()
    return fig_save(fig, '03_lr_schedule.png')


# ── 4. Precision & Recall Curves ──────────────────────────────────────────── #
def plot_precision_recall_curves(history):
    ep = _ep(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Precision and Recall over Training', fontsize=STYLE['title_size'], fontweight='bold')

    for ax, key, title in [(axes[0], 'precision', 'Precision'), (axes[1], 'recall', 'Recall')]:
        ax.plot(ep, history[f'train_{key}'], color=STYLE['train_color'], lw=2, label=f'Train')
        ax.plot(ep, history[f'val_{key}'],   color=STYLE['val_color'],   lw=2, label=f'Val')
        ax.set_title(title, fontsize=STYLE['title_size'])
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=STYLE['grid_alpha'])

    plt.tight_layout()
    return fig_save(fig, '04_precision_recall_curves.png')


# ── 5. F1 & Specificity Curves ────────────────────────────────────────────── #
def plot_f1_specificity_curves(history):
    ep = _ep(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('F1 Score and Specificity over Training', fontsize=STYLE['title_size'], fontweight='bold')

    for ax, key, title in [(axes[0], 'f1', 'F1 Score'), (axes[1], 'specificity', 'Specificity')]:
        ax.plot(ep, history[f'train_{key}'], color=STYLE['train_color'], lw=2, label='Train')
        ax.plot(ep, history[f'val_{key}'],   color=STYLE['val_color'],   lw=2, label='Val')
        ax.set_title(title, fontsize=STYLE['title_size'])
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=STYLE['grid_alpha'])

    plt.tight_layout()
    return fig_save(fig, '05_f1_specificity_curves.png')


# ── 6. Metric Heatmap ─────────────────────────────────────────────────────── #
def plot_metric_heatmap(history):
    metrics = ['val_iou', 'val_dice', 'val_precision', 'val_recall', 'val_f1', 'val_specificity']
    labels  = ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'Specificity']
    ep = _ep(history)

    data = np.array([[history[m][i] for i in range(len(ep))] for m in metrics])

    fig, ax = plt.subplots(figsize=(max(12, len(ep) * 0.3), 5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[0.5, len(ep) + 0.5, len(metrics) - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label='Score')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Epoch'); ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title('Validation Metric Heatmap over Training',
                 fontsize=STYLE['title_size'], fontweight='bold')
    plt.tight_layout()
    return fig_save(fig, '06_metric_heatmap.png')


# ── 7. Best-Epoch Confusion Matrix ────────────────────────────────────────── #
def plot_best_epoch_confusion_matrix(history):
    best_ep = int(np.argmax(history['val_iou']))
    tp = history['val_tp'][best_ep]
    fp = history['val_fp'][best_ep]
    fn = history['val_fn'][best_ep]
    tn = history['val_tn'][best_ep]

    cm = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Confusion Matrix at Best Epoch (Epoch {best_ep + 1})',
                 fontsize=STYLE['title_size'], fontweight='bold')

    ConfusionMatrixDisplay(cm, display_labels=['Background', 'Pipe']).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Raw Counts', fontsize=11)

    ConfusionMatrixDisplay(cm_norm, display_labels=['Background', 'Pipe']).plot(
        ax=axes[1], colorbar=False, cmap='Blues', values_format='.3f')
    axes[1].set_title('Normalised', fontsize=11)

    total = tp + fp + fn + tn
    prec  = tp / (tp + fp + 1e-6)
    rec   = tp / (tp + fn + 1e-6)
    f1    = 2 * prec * rec / (prec + rec + 1e-6)
    acc   = (tp + tn) / (total + 1e-6)

    fig.text(0.98, 0.5,
             f"Epoch {best_ep + 1}\n\nTP={tp:,.0f}\nFP={fp:,.0f}\nFN={fn:,.0f}\nTN={tn:,.0f}\n\n"
             f"Acc:  {acc:.4f}\nPrec: {prec:.4f}\nRec:  {rec:.4f}\nF1:   {f1:.4f}",
             ha='right', va='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig_save(fig, '07_best_epoch_confusion_matrix.png')


# ── 8. Gradient Norm ──────────────────────────────────────────────────────── #
def plot_gradient_norm(history):
    ep = _ep(history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, history['grad_norm'], color='#FF5722', lw=1.5)
    ax.fill_between(ep, history['grad_norm'], alpha=0.2, color='#FF5722')
    ax.set_title('Gradient Norm per Epoch', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Gradient Norm')
    ax.grid(alpha=STYLE['grid_alpha'])
    plt.tight_layout()
    return fig_save(fig, '08_gradient_norm_curve.png')


# ── 9. Train–Val Gap ──────────────────────────────────────────────────────── #
def plot_train_val_gap(history):
    ep   = _ep(history)
    keys = ['iou', 'dice', 'f1']
    labels = ['IoU', 'Dice', 'F1']
    colors = ['#2196F3', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in zip(keys, labels, colors):
        gap = [t - v for t, v in zip(history[f'train_{key}'], history[f'val_{key}'])]
        ax.plot(ep, gap, color=color, lw=2, label=label)

    ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
    ax.fill_between(ep, 0, [max(g, 0) for g in
                            [t - v for t, v in zip(history['train_iou'], history['val_iou'])]],
                    alpha=0.08, color='#2196F3')
    ax.set_title('Train–Val Gap (positive = overfitting)',
                 fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train − Val score')
    ax.legend(); ax.grid(alpha=STYLE['grid_alpha'])
    plt.tight_layout()
    return fig_save(fig, '09_train_val_gap.png')


# ── 10. Per-epoch Summary Table ────────────────────────────────────────────── #
def plot_per_epoch_table(history):
    ep = _ep(history)
    step = max(1, len(ep) // 20)
    rows = list(range(0, len(ep), step))
    if (len(ep) - 1) not in rows:
        rows.append(len(ep) - 1)

    col_labels = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                  'Train Dice', 'Val Dice', 'Val Prec', 'Val Recall', 'Val F1', 'LR']
    table_data = []
    for i in rows:
        table_data.append([
            str(i + 1),
            f"{history['train_loss'][i]:.4f}",
            f"{history['val_loss'][i]:.4f}",
            f"{history['train_iou'][i]:.4f}",
            f"{history['val_iou'][i]:.4f}",
            f"{history['train_dice'][i]:.4f}",
            f"{history['val_dice'][i]:.4f}",
            f"{history['val_precision'][i]:.4f}",
            f"{history['val_recall'][i]:.4f}",
            f"{history['val_f1'][i]:.4f}",
            f"{history['lr'][i]:.2e}",
        ])

    fig_h = max(4, len(table_data) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Highlight header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#37474F')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best val IoU row
    best_row_in_table = max(range(len(rows)), key=lambda x: history['val_iou'][rows[x]])
    for j in range(len(col_labels)):
        tbl[best_row_in_table + 1, j].set_facecolor('#C8E6C9')

    ax.set_title('Per-Epoch Metrics Summary (every Nth epoch shown; green = best val IoU)',
                 fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig_save(fig, '10_per_epoch_summary_table.png')


# ── All plots ──────────────────────────────────────────────────────────────── #
def generate_all_plots(history):
    print("\n  Generating training plots…")
    plots = [
        plot_loss_curves(history),
        plot_iou_dice_curves(history),
        plot_lr_schedule(history),
        plot_precision_recall_curves(history),
        plot_f1_specificity_curves(history),
        plot_metric_heatmap(history),
        plot_best_epoch_confusion_matrix(history),
        plot_gradient_norm(history),
        plot_train_val_gap(history),
        plot_per_epoch_table(history),
    ]
    for p in plots:
        print(f"    Saved: {p}")
    return plots


# ─────────────────────────── MARKDOWN REPORT ─────────────────────────────── #
def write_training_report(history, best_epoch, timestamp, elapsed_s):
    ep = _ep(history)
    best = best_epoch  # 0-indexed
    n_ep = len(ep)

    def fmt(key, i):
        return f"{history[key][i]:.4f}"

    lines = [
        "# Training Report",
        f"**Generated:** {timestamp}  ",
        f"**Model:** U-Net (ResNet-34 encoder, scSE decoder attention)  ",
        f"**Total epochs completed:** {n_ep}  ",
        f"**Total training time:** {elapsed_s / 60:.1f} min  ",
        "",
        "---",
        "",
        "## 1. Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Encoder | ResNet-34 (ImageNet pretrained) |",
        f"| Decoder attention | scSE |",
        f"| Input size | {IMG_SIZE} × {IMG_SIZE} |",
        f"| Batch size | {BATCH_SIZE} |",
        f"| Initial LR | {LR} |",
        f"| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |",
        f"| Early stopping | Disabled (full {EPOCHS} epochs) |",
        f"| Validation split | {int(VAL_SPLIT * 100)}% |",
        f"| Loss function | Dice Loss (binary) |",
        f"| Augmentations | HFlip, VFlip, Rotate90, ShiftScaleRotate, ImageNet Normalise |",
        f"| Random seed | {RANDOM_SEED} |",
        "",
        "---",
        "",
        "## 2. Best Epoch Results",
        "",
        f"**Best epoch (by val IoU):** Epoch {best + 1}  ",
        "",
        "| Metric | Train | Val |",
        "|--------|-------|-----|",
        f"| Loss      | {fmt('train_loss', best)} | {fmt('val_loss', best)} |",
        f"| IoU       | {fmt('train_iou', best)} | {fmt('val_iou', best)} |",
        f"| Dice      | {fmt('train_dice', best)} | {fmt('val_dice', best)} |",
        f"| Precision | {fmt('train_precision', best)} | {fmt('val_precision', best)} |",
        f"| Recall    | {fmt('train_recall', best)} | {fmt('val_recall', best)} |",
        f"| F1        | {fmt('train_f1', best)} | {fmt('val_f1', best)} |",
        f"| Specificity | {fmt('train_specificity', best)} | {fmt('val_specificity', best)} |",
        f"| Accuracy  | {fmt('train_accuracy', best)} | {fmt('val_accuracy', best)} |",
        "",
        "---",
        "",
        "## 3. Final Epoch Results",
        "",
        f"**Final epoch:** Epoch {n_ep}  ",
        "",
        "| Metric | Train | Val |",
        "|--------|-------|-----|",
        f"| Loss      | {fmt('train_loss', -1)} | {fmt('val_loss', -1)} |",
        f"| IoU       | {fmt('train_iou', -1)} | {fmt('val_iou', -1)} |",
        f"| Dice      | {fmt('train_dice', -1)} | {fmt('val_dice', -1)} |",
        f"| Precision | {fmt('train_precision', -1)} | {fmt('val_precision', -1)} |",
        f"| Recall    | {fmt('train_recall', -1)} | {fmt('val_recall', -1)} |",
        f"| F1        | {fmt('train_f1', -1)} | {fmt('val_f1', -1)} |",
        "",
        "---",
        "",
        "## 4. Full Epoch History",
        "",
        "| Epoch | Train Loss | Val Loss | Train IoU | Val IoU | "
        "Train Dice | Val Dice | Val F1 | LR |",
        "|-------|-----------|----------|-----------|---------|"
        "-----------|----------|--------|-----|",
    ]

    for i in range(n_ep):
        marker = " ← best" if i == best else ""
        lines.append(
            f"| {i + 1} | {history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
            f"{history['train_iou'][i]:.4f} | {history['val_iou'][i]:.4f} | "
            f"{history['train_dice'][i]:.4f} | {history['val_dice'][i]:.4f} | "
            f"{history['val_f1'][i]:.4f} | {history['lr'][i]:.2e}{marker} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Generated Plots",
        "",
        "| # | File | Description |",
        "|---|------|-------------|",
        "| 1 | `01_loss_curves.png` | Train and val Dice loss per epoch |",
        "| 2 | `02_iou_dice_curves.png` | Train/val IoU and Dice per epoch |",
        "| 3 | `03_lr_schedule.png` | Learning rate schedule |",
        "| 4 | `04_precision_recall_curves.png` | Train/val precision and recall per epoch |",
        "| 5 | `05_f1_specificity_curves.png` | Train/val F1 and specificity per epoch |",
        "| 6 | `06_metric_heatmap.png` | Heatmap of all val metrics across epochs |",
        "| 7 | `07_best_epoch_confusion_matrix.png` | Pixel confusion matrix at best val epoch |",
        "| 8 | `08_gradient_norm_curve.png` | Average gradient norm per epoch |",
        "| 9 | `09_train_val_gap.png` | Train−Val gap (overfitting monitor) |",
        "| 10 | `10_per_epoch_summary_table.png` | Summary table of key epochs |",
        "",
        "---",
        "",
        "*Auto-generated by `trainUsingMask.py`*",
    ]

    path = os.path.join(OUTPUT_DIR, 'training_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")
    return path


# ─────────────────────────── MAIN TRAINING LOOP ───────────────────────────── #
def train_model():
    timestamp   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time  = time.time()

    print("=" * 65)
    print("  U-Net Pipe Segmentation — Training")
    print(f"  {timestamp}  |  device: {DEVICE}")
    print("=" * 65)

    # ── Datasets ──
    print("\nLoading datasets…")
    full_train = PipeDataset(IMAGE_DIR, MASK_DIR, augmentation=get_augmentation(train=True))
    full_val   = PipeDataset(IMAGE_DIR, MASK_DIR, augmentation=get_augmentation(train=False))

    n_total = len(full_train)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=g)

    train_loader = DataLoader(
        torch.utils.data.Subset(full_train, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_val, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Train: {n_train} | Val: {n_val} images")

    # ── Model ──
    print("\nInitialising model…")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
        decoder_attention_type='scse',
    )
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    loss_fn   = smp.losses.DiceLoss(mode='binary', from_logits=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )

    # ── History ──
    metric_keys = ['loss', 'iou', 'dice', 'precision', 'recall',
                   'f1', 'specificity', 'accuracy']
    history = {f'train_{k}': [] for k in metric_keys}
    history.update({f'val_{k}': [] for k in metric_keys})
    history['lr']       = []
    history['grad_norm'] = []
    history['val_tp']   = []
    history['val_fp']   = []
    history['val_fn']   = []
    history['val_tn']   = []

    best_val_iou = -1.0
    best_epoch   = 0

    # ── Loop ──
    print(f"\nTraining for {EPOCHS} epochs…\n")
    for epoch in range(EPOCHS):
        ep_start = time.time()

        train_m = run_epoch(model, train_loader, optimizer, loss_fn,
                            train=True,  device=DEVICE)
        val_m   = run_epoch(model, val_loader,   optimizer, loss_fn,
                            train=False, device=DEVICE)

        scheduler.step(val_m['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        for k in metric_keys:
            history[f'train_{k}'].append(train_m[k])
            history[f'val_{k}'].append(val_m[k])
        history['lr'].append(current_lr)
        history['grad_norm'].append(train_m.get('grad_norm', 0.0))

        # Store pixel-level CM totals (already summed inside avg_metrics)
        history['val_tp'].append(val_m['tp'])
        history['val_fp'].append(val_m['fp'])
        history['val_fn'].append(val_m['fn'])
        history['val_tn'].append(val_m['tn'])

        ep_time  = time.time() - ep_start
        improved = val_m['iou'] > best_val_iou

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"TrainLoss={train_m['loss']:.4f} IoU={train_m['iou']:.4f} Dice={train_m['dice']:.4f} | "
            f"ValLoss={val_m['loss']:.4f} IoU={val_m['iou']:.4f} Dice={val_m['dice']:.4f} "
            f"F1={val_m['f1']:.4f} | LR={current_lr:.2e} | {ep_time:.1f}s"
            + (" ★" if improved else "")
        )

        if improved:
            best_val_iou = val_m['iou']
            best_epoch   = epoch
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'best_pipe_unet.pth'))

        # Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f'pipe_unet_epoch_{epoch + 1}.pth'))

    elapsed_s = time.time() - start_time
    print(f"\nTraining complete. Best val IoU={best_val_iou:.4f} at epoch {best_epoch + 1}.")
    print(f"Total time: {elapsed_s / 60:.1f} min")

    # ── Save history JSON ──
    json_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {json_path}")

    # ── Generate all plots ──
    generate_all_plots(history)

    # ── Write report ──
    print("\nWriting training report…")
    write_training_report(history, best_epoch, timestamp, elapsed_s)

    print("\n" + "=" * 65)
    print(f"  Results → {OUTPUT_DIR}/")
    print(f"  Best model → {CHECKPOINT_DIR}/best_pipe_unet.pth")
    print("=" * 65)


if __name__ == '__main__':
    train_model()
