"""
Comprehensive Model Evaluation Script
U-Net Pipe Segmentation — Post-Training Evaluation Only

Generates:
  - Per-image metric distributions (IoU, Dice, Precision, Recall, F1, Specificity)
  - Global Precision–Recall curve + AP
  - ROC curve + AUC
  - Pixel-level confusion matrix (raw + normalised)
  - IoU vs Dice scatter coloured by precision
  - Summary bar chart (mean ± std)
  - Failure case analysis with TP/FP/FN error maps
  - Success case analysis with prediction overlays
  - Raw vs Ground Truth vs Prediction vs Heatmap comparison grid
  - Threshold sweep analysis
  - Per-image IoU ranking
  - Pipe mask coverage analysis
  - validation_report.md
  - per_image_metrics.json
"""

import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, ConfusionMatrixDisplay,
)

warnings.filterwarnings('ignore')

# ─────────────────────────── CONFIGURATION ────────────────────────────────── #
MODEL_PATH   = 'models-new-final/best_pipe_unet.pth'
IMAGE_DIR    = 'merged_dataset/images/Train'
MASK_DIR     = 'merged_dataset/masks/Train'
OUTPUT_DIR   = 'evaluation_results'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE     = 512
THRESHOLD    = 0.5
RANDOM_SEED  = 42
N_FAILURE    = 8
N_COMPARISON = 6

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE = {
    'train_color': '#2196F3',
    'val_color':   '#F44336',
    'iou_color':   '#4CAF50',
    'dice_color':  '#FF9800',
    'grid_alpha':  0.25,
    'fig_dpi':     150,
    'title_size':  14,
    'label_size':  11,
}


# ─────────────────────────── MODEL LOADER ─────────────────────────────────── #
def load_model(path=MODEL_PATH):
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        classes=1,
        activation='sigmoid',
        decoder_attention_type='scse',
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ──────────────────────── IMAGE PREPROCESSING ─────────────────────────────── #
def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return img, x.to(DEVICE)


def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    return (mask > 128).astype(np.float32)


# ─────────────────────── FULL DATASET INFERENCE ───────────────────────────── #
def run_full_inference(model):
    all_images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    results = []
    print(f"Running inference on {len(all_images)} images…")
    for name in all_images:
        img_path  = os.path.join(IMAGE_DIR, name)
        mask_path = os.path.join(MASK_DIR, Path(name).stem + '.png')
        if not os.path.exists(mask_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        true_mask = load_mask(mask_path)
        if true_mask is None:
            continue
        img_rgb, x = preprocess(img_bgr)
        with torch.no_grad():
            prob = model(x).squeeze().cpu().numpy()
        results.append({
            'name':      name,
            'img_rgb':   img_rgb,
            'true_mask': true_mask,
            'pred_prob': prob,
            'pred_bin':  (prob > THRESHOLD).astype(np.float32),
        })
    print(f"  Processed {len(results)} valid image-mask pairs.")
    return results


# ──────────────────────── PER-IMAGE METRICS ───────────────────────────────── #
def compute_per_image_metrics(results):
    metrics = []
    for r in results:
        gt = r['true_mask'].flatten()
        pb = r['pred_bin'].flatten()
        pp = r['pred_prob'].flatten()

        tp = np.sum(gt * pb)
        fp = np.sum((1 - gt) * pb)
        fn = np.sum(gt * (1 - pb))
        tn = np.sum((1 - gt) * (1 - pb))

        iou         = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        dice        = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        precision   = (tp + 1e-6) / (tp + fp + 1e-6)
        recall      = (tp + 1e-6) / (tp + fn + 1e-6)
        f1          = (2 * precision * recall) / (precision + recall + 1e-6)
        specificity = (tn + 1e-6) / (tn + fp + 1e-6)
        accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-6)

        try:
            ap = average_precision_score(gt.astype(int), pp)
        except Exception:
            ap = 0.0

        metrics.append({
            'name':        r['name'],
            'iou':         float(iou),
            'dice':        float(dice),
            'precision':   float(precision),
            'recall':      float(recall),
            'f1':          float(f1),
            'specificity': float(specificity),
            'accuracy':    float(accuracy),
            'ap':          float(ap),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    return metrics


def aggregate_metrics(metrics):
    keys = ['iou', 'dice', 'precision', 'recall', 'f1', 'specificity', 'accuracy', 'ap']
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics]
        agg[k] = {
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals)),
            'min':    float(np.min(vals)),
            'max':    float(np.max(vals)),
            'median': float(np.median(vals)),
        }
    agg['global_cm'] = {
        'tp': sum(m['tp'] for m in metrics),
        'fp': sum(m['fp'] for m in metrics),
        'fn': sum(m['fn'] for m in metrics),
        'tn': sum(m['tn'] for m in metrics),
    }
    return agg


# ═══════════════════════════ PLOTTING FUNCTIONS ═══════════════════════════════ #
def fig_save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=STYLE['fig_dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── 1. Per-image Metric Distributions ─────────────────────────────────────── #
def plot_metric_distributions(metrics):
    keys   = ['iou', 'dice', 'precision', 'recall', 'f1', 'specificity']
    titles = ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Specificity']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Per-Image Metric Distributions', fontsize=16, fontweight='bold')

    for ax, key, title, color in zip(axes.flat, keys, titles, colors):
        vals     = [m[key] for m in metrics]
        mean_v   = np.mean(vals)
        median_v = np.median(vals)
        ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor='white', lw=0.5)
        ax.axvline(mean_v,   color='black',  ls='--', lw=2, label=f'Mean={mean_v:.3f}')
        ax.axvline(median_v, color='orange', ls=':',  lw=2, label=f'Median={median_v:.3f}')
        ax.set_title(title, fontsize=STYLE['title_size'])
        ax.set_xlabel('Score'); ax.set_ylabel('Count')
        ax.legend(fontsize=9); ax.grid(alpha=STYLE['grid_alpha'])
        ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig_save(fig, '01_metric_distributions.png')


# ── 2. Precision–Recall Curve ─────────────────────────────────────────────── #
def plot_precision_recall_curve(results):
    all_gt   = np.concatenate([r['true_mask'].flatten() for r in results])
    all_prob = np.concatenate([r['pred_prob'].flatten()  for r in results])

    if len(all_gt) > 2_000_000:
        idx = np.random.choice(len(all_gt), 2_000_000, replace=False)
        all_gt   = all_gt[idx]
        all_prob = all_prob[idx]

    precision, recall, thresholds = precision_recall_curve(all_gt.astype(int), all_prob)
    ap = average_precision_score(all_gt.astype(int), all_prob)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = np.argmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else THRESHOLD

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#E91E63', lw=2.5, label=f'PR Curve (AP={ap:.3f})')
    ax.scatter(recall[best_idx], precision[best_idx], s=120, color='black', zorder=5,
               label=f'Best F1={f1_scores[best_idx]:.3f} @ thr={best_thr:.2f}')
    ax.fill_between(recall, precision, alpha=0.15, color='#E91E63')
    ax.set_title('Precision–Recall Curve', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Recall',    fontsize=STYLE['label_size'])
    ax.set_ylabel('Precision', fontsize=STYLE['label_size'])
    ax.legend(fontsize=10); ax.grid(alpha=STYLE['grid_alpha'])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig_save(fig, '02_precision_recall_curve.png'), ap


# ── 3. ROC Curve ──────────────────────────────────────────────────────────── #
def plot_roc_curve(results):
    all_gt   = np.concatenate([r['true_mask'].flatten() for r in results])
    all_prob = np.concatenate([r['pred_prob'].flatten()  for r in results])

    if len(all_gt) > 2_000_000:
        idx = np.random.choice(len(all_gt), 2_000_000, replace=False)
        all_gt   = all_gt[idx]
        all_prob = all_prob[idx]

    fpr, tpr, _ = roc_curve(all_gt.astype(int), all_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'ROC Curve (AUC={roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    ax.set_title('ROC Curve', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=STYLE['label_size'])
    ax.set_ylabel('True Positive Rate',  fontsize=STYLE['label_size'])
    ax.legend(fontsize=10); ax.grid(alpha=STYLE['grid_alpha'])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig_save(fig, '03_roc_curve.png'), roc_auc


# ── 4. Confusion Matrix ────────────────────────────────────────────────────── #
def plot_confusion_matrix(agg):
    cm_data = agg['global_cm']
    tp, fp, fn, tn = cm_data['tp'], cm_data['fp'], cm_data['fn'], cm_data['tn']
    cm = np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')

    ConfusionMatrixDisplay(cm, display_labels=['Background', 'Pipe']).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Pixel-level Confusion Matrix (counts)', fontsize=12)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    ConfusionMatrixDisplay(cm_norm, display_labels=['Background', 'Pipe']).plot(
        ax=axes[1], colorbar=False, cmap='Blues', values_format='.3f')
    axes[1].set_title('Normalised Confusion Matrix', fontsize=12)

    total = tp + fp + fn + tn
    prec  = tp / (tp + fp + 1e-6)
    rec   = tp / (tp + fn + 1e-6)
    f1    = 2 * prec * rec / (prec + rec + 1e-6)
    acc   = (tp + tn) / (total + 1e-6)
    spec  = tn / (tn + fp + 1e-6)

    stats = (f"Total pixels: {total:,}\n"
             f"TP={tp:,}  FP={fp:,}\n"
             f"FN={fn:,}  TN={tn:,}\n\n"
             f"Accuracy:    {acc:.4f}\n"
             f"Precision:   {prec:.4f}\n"
             f"Recall:      {rec:.4f}\n"
             f"Specificity: {spec:.4f}\n"
             f"F1:          {f1:.4f}")
    fig.text(0.98, 0.5, stats, ha='right', va='center', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.80, 1])
    return fig_save(fig, '04_confusion_matrix.png')


# ── 5. IoU vs Dice Scatter ─────────────────────────────────────────────────── #
def plot_iou_dice_scatter(metrics):
    ious  = [m['iou']       for m in metrics]
    dices = [m['dice']      for m in metrics]
    precs = [m['precision'] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(ious, dices, c=precs, cmap='RdYlGn', s=50, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Precision')

    iou_line  = np.linspace(0, 1, 200)
    dice_line = 2 * iou_line / (1 + iou_line)
    ax.plot(iou_line, dice_line, 'k--', lw=1.5, alpha=0.5, label='Dice = 2·IoU/(1+IoU)')

    ax.set_title('IoU vs Dice per Image', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('IoU', fontsize=STYLE['label_size'])
    ax.set_ylabel('Dice', fontsize=STYLE['label_size'])
    ax.legend(fontsize=9); ax.grid(alpha=STYLE['grid_alpha'])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig_save(fig, '05_iou_dice_scatter.png')


# ── 6. Summary Bar Chart ──────────────────────────────────────────────────── #
def plot_summary_bars(agg):
    keys   = ['iou', 'dice', 'precision', 'recall', 'f1', 'specificity', 'accuracy', 'ap']
    labels = ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'Specificity', 'Accuracy', 'AP']
    means  = [agg[k]['mean'] for k in keys]
    stds   = [agg[k]['std']  for k in keys]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63',
              '#9C27B0', '#00BCD4', '#FF5722', '#795548']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='white', lw=0.5)
    for bar, v, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Summary Performance Metrics (mean ± std across all images)',
                 fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_ylabel('Score', fontsize=STYLE['label_size'])
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
    ax.grid(alpha=STYLE['grid_alpha'], axis='y')
    plt.tight_layout()
    return fig_save(fig, '06_summary_bars.png')


# ── 7. Failure Cases ──────────────────────────────────────────────────────── #
def plot_failure_cases(results, metrics, n=N_FAILURE):
    worst_names = {m['name'] for m in sorted(metrics, key=lambda x: x['iou'])[:n]}
    worst_results = sorted(
        [r for r in results if r['name'] in worst_names],
        key=lambda r: next(m['iou'] for m in metrics if m['name'] == r['name'])
    )

    rows, cols = len(worst_results), 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle(f'Failure Case Analysis — {n} Worst IoU Images',
                 fontsize=15, fontweight='bold')
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(worst_results):
        m = next(m for m in metrics if m['name'] == r['name'])
        gt, pred = r['true_mask'], r['pred_bin']

        err = np.zeros((*gt.shape, 3), dtype=np.float32)
        err[..., 1] = ((gt == 1) & (pred == 1)).astype(float)
        err[..., 0] = ((gt == 0) & (pred == 1)).astype(float)
        err[..., 2] = ((gt == 1) & (pred == 0)).astype(float)

        name_short = Path(r['name']).stem[:28]
        axes[i, 0].imshow(r['img_rgb']);  axes[i, 0].axis('off')
        axes[i, 0].set_title(f'{name_short}\nIoU={m["iou"]:.3f}', fontsize=8)
        axes[i, 1].imshow(gt,   cmap='gray'); axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth', fontsize=8)
        axes[i, 2].imshow(pred, cmap='gray'); axes[i, 2].axis('off')
        axes[i, 2].set_title(f'Prediction\nDice={m["dice"]:.3f}', fontsize=8)
        axes[i, 3].imshow(err);              axes[i, 3].axis('off')
        axes[i, 3].legend(handles=[
            Patch(color='green', label='TP'),
            Patch(color='red',   label='FP'),
            Patch(color='blue',  label='FN'),
        ], fontsize=7, loc='lower right')
        axes[i, 3].set_title('Error Map', fontsize=8)

    plt.tight_layout()
    return fig_save(fig, '07_failure_cases.png')


# ── 8. Success Cases ──────────────────────────────────────────────────────── #
def plot_success_cases(results, metrics, n=6):
    best_names = {m['name'] for m in sorted(metrics, key=lambda x: x['iou'], reverse=True)[:n]}
    best_results = sorted(
        [r for r in results if r['name'] in best_names],
        key=lambda r: next(m['iou'] for m in metrics if m['name'] == r['name']),
        reverse=True,
    )

    rows, cols = len(best_results), 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle(f'Success Cases — {n} Best IoU Images',
                 fontsize=15, fontweight='bold')
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(best_results):
        m = next(m for m in metrics if m['name'] == r['name'])
        overlay = r['img_rgb'].copy()
        contours, _ = cv2.findContours(
            r['pred_bin'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        axes[i, 0].imshow(r['img_rgb']);         axes[i, 0].axis('off')
        axes[i, 0].set_title(f'{Path(r["name"]).stem[:28]}\nIoU={m["iou"]:.3f}', fontsize=8)
        axes[i, 1].imshow(r['true_mask'], cmap='gray'); axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth', fontsize=8)
        axes[i, 2].imshow(overlay);              axes[i, 2].axis('off')
        axes[i, 2].set_title(f'Pred Overlay\nDice={m["dice"]:.3f}', fontsize=8)

    plt.tight_layout()
    return fig_save(fig, '08_success_cases.png')


# ── 9. Raw vs Prediction Comparison Grid ─────────────────────────────────── #
def plot_raw_vs_prediction(results, metrics, n=N_COMPARISON):
    selected = [results[i] for i in random.sample(range(len(results)), min(n, len(results)))]
    met_dict = {m['name']: m for m in metrics}

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, len(selected) * 4))
    fig.suptitle('Raw Image | Ground Truth | Prediction | Probability Heatmap',
                 fontsize=15, fontweight='bold')
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(selected):
        m = met_dict.get(r['name'], {})
        axes[i, 0].imshow(r['img_rgb']);              axes[i, 0].axis('off')
        axes[i, 0].set_title(Path(r['name']).stem[:24], fontsize=8)
        axes[i, 1].imshow(r['true_mask'], cmap='gray'); axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth', fontsize=8)
        axes[i, 2].imshow(r['pred_bin'],  cmap='gray'); axes[i, 2].axis('off')
        axes[i, 2].set_title(f'Prediction (thr={THRESHOLD})\nIoU={m.get("iou", 0):.3f}', fontsize=8)
        hm = axes[i, 3].imshow(r['pred_prob'], cmap='plasma', vmin=0, vmax=1)
        axes[i, 3].axis('off')
        axes[i, 3].set_title('Probability Heatmap', fontsize=8)
        plt.colorbar(hm, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig_save(fig, '09_raw_vs_prediction_comparison.png')


# ── 10. Threshold Sweep ────────────────────────────────────────────────────── #
def plot_threshold_analysis(results):
    thresholds = np.linspace(0.1, 0.95, 40)
    all_gt   = np.concatenate([r['true_mask'].flatten() for r in results])
    all_prob = np.concatenate([r['pred_prob'].flatten()  for r in results])

    if len(all_gt) > 1_000_000:
        idx = np.random.choice(len(all_gt), 1_000_000, replace=False)
        all_gt   = all_gt[idx]
        all_prob = all_prob[idx]

    thr_metrics = {k: [] for k in ['iou', 'dice', 'precision', 'recall', 'f1']}
    for t in thresholds:
        pred = (all_prob > t).astype(float)
        tp = np.sum(all_gt * pred)
        fp = np.sum((1 - all_gt) * pred)
        fn = np.sum(all_gt * (1 - pred))
        thr_metrics['iou'].append((tp + 1e-6) / (tp + fp + fn + 1e-6))
        thr_metrics['dice'].append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        rec  = (tp + 1e-6) / (tp + fn + 1e-6)
        thr_metrics['precision'].append(prec)
        thr_metrics['recall'].append(rec)
        thr_metrics['f1'].append(2 * prec * rec / (prec + rec + 1e-6))

    best_thr = thresholds[np.argmax(thr_metrics['f1'])]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, thr_metrics['iou'],       label='IoU',       lw=2, color='#2196F3')
    ax.plot(thresholds, thr_metrics['dice'],      label='Dice',      lw=2, color='#FF9800')
    ax.plot(thresholds, thr_metrics['precision'], label='Precision', lw=2, color='#4CAF50')
    ax.plot(thresholds, thr_metrics['recall'],    label='Recall',    lw=2, color='#E91E63')
    ax.plot(thresholds, thr_metrics['f1'],        label='F1',        lw=2.5, color='#9C27B0')
    ax.axvline(best_thr, color='black', ls='--', lw=1.5, label=f'Best F1 thr={best_thr:.2f}')
    ax.axvline(THRESHOLD, color='gray', ls=':', lw=1.5, alpha=0.7, label=f'Default thr={THRESHOLD}')
    ax.set_title('Metrics vs Decision Threshold', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=STYLE['label_size'])
    ax.set_ylabel('Score',     fontsize=STYLE['label_size'])
    ax.legend(fontsize=9); ax.grid(alpha=STYLE['grid_alpha'])
    ax.set_xlim(0.1, 0.95); ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig_save(fig, '10_threshold_analysis.png'), best_thr


# ── 11. Per-image IoU Ranking ─────────────────────────────────────────────── #
def plot_iou_ranking(metrics):
    sorted_m = sorted(metrics, key=lambda x: x['iou'])
    ious   = [m['iou'] for m in sorted_m]
    colors = ['#F44336' if v < 0.4 else '#FF9800' if v < 0.7 else '#4CAF50' for v in ious]

    fig, ax = plt.subplots(figsize=(max(10, len(ious) * 0.12), 5))
    ax.bar(range(len(ious)), ious, color=colors, width=1.0, edgecolor='none')
    ax.axhline(np.mean(ious), color='black', ls='--', lw=2,
               label=f'Mean IoU = {np.mean(ious):.3f}')
    ax.axhline(0.5, color='orange', ls=':', lw=1.5, alpha=0.7)
    ax.legend(handles=[
        Patch(color='#4CAF50', label='Good (IoU≥0.7)'),
        Patch(color='#FF9800', label='Fair (0.4–0.7)'),
        Patch(color='#F44336', label='Poor (IoU<0.4)'),
        plt.Line2D([0], [0], color='black', ls='--', lw=2, label=f'Mean={np.mean(ious):.3f}'),
    ], fontsize=9)
    ax.set_title('Per-Image IoU Ranking (sorted)', fontsize=STYLE['title_size'], fontweight='bold')
    ax.set_xlabel('Image Index (sorted by IoU)'); ax.set_ylabel('IoU')
    ax.set_xlim(-1, len(ious)); ax.set_ylim(0, 1)
    ax.grid(alpha=STYLE['grid_alpha'], axis='y')
    plt.tight_layout()
    return fig_save(fig, '11_iou_ranking.png')


# ── 12. Mask Coverage Analysis ────────────────────────────────────────────── #
def plot_mask_coverage(results, metrics):
    coverage = [r['true_mask'].sum() / r['true_mask'].size for r in results]
    met_dict = {m['name']: m for m in metrics}
    ious     = [met_dict[r['name']]['iou'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Pipe Mask Coverage Analysis', fontsize=15, fontweight='bold')

    axes[0].hist(coverage, bins=30, color='#00BCD4', alpha=0.8, edgecolor='white')
    axes[0].set_title('Distribution of Pipe Coverage per Image')
    axes[0].set_xlabel('Fraction of Pixels that are Pipe')
    axes[0].set_ylabel('Count')
    axes[0].grid(alpha=STYLE['grid_alpha'])

    sc = axes[1].scatter(coverage, ious, alpha=0.6, c=ious,
                         cmap='RdYlGn', s=40, edgecolors='none')
    plt.colorbar(sc, ax=axes[1], label='IoU')
    axes[1].set_title('IoU vs Pipe Coverage')
    axes[1].set_xlabel('Pipe Coverage Fraction')
    axes[1].set_ylabel('IoU')
    axes[1].grid(alpha=STYLE['grid_alpha'])

    plt.tight_layout()
    return fig_save(fig, '12_mask_coverage_analysis.png')


# ─────────────────────────── MARKDOWN REPORT ─────────────────────────────── #
def write_markdown_report(agg, ap, roc_auc, best_thr, metrics, timestamp):
    n = len(metrics)

    def grade(v):
        if v >= 0.85: return "Excellent"
        if v >= 0.70: return "Good"
        if v >= 0.55: return "Fair"
        return "Needs Improvement"

    cm  = agg['global_cm']
    total = cm['tp'] + cm['fp'] + cm['fn'] + cm['tn']
    prec  = cm['tp'] / (cm['tp'] + cm['fp'] + 1e-6)
    rec   = cm['tp'] / (cm['tp'] + cm['fn'] + 1e-6)
    f1_cm = 2 * prec * rec / (prec + rec + 1e-6)
    acc   = (cm['tp'] + cm['tn']) / (total + 1e-6)
    spec  = cm['tn'] / (cm['tn'] + cm['fp'] + 1e-6)

    lines = [
        "# Quantitative Validation Report",
        f"**Generated:** {timestamp}  ",
        "**Model:** U-Net (ResNet-34 encoder, scSE decoder attention)  ",
        f"**Dataset:** {n} images — `merged_dataset/images/Train`  ",
        f"**Default threshold:** {THRESHOLD}  |  "
        f"**Optimal F1 threshold:** {best_thr:.3f}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Mean | Std | Min | Max | Median | Grade |",
        "|--------|------|-----|-----|-----|--------|-------|",
    ]

    for k, label in [('iou','IoU'), ('dice','Dice'), ('precision','Precision'),
                     ('recall','Recall'), ('f1','F1'), ('specificity','Specificity'),
                     ('accuracy','Accuracy'), ('ap','Avg Precision')]:
        s = agg[k]
        lines.append(
            f"| {label} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['median']:.4f} | "
            f"{grade(s['mean'])} |"
        )

    lines += [
        "",
        f"**ROC AUC:** {roc_auc:.4f}  ",
        f"**Global Average Precision (PR curve):** {ap:.4f}  ",
        "",
        "---",
        "",
        "## 2. Confusion Matrix (pixel-level)",
        "",
        "|  | Predicted Background | Predicted Pipe |",
        "|--|----------------------|----------------|",
        f"| **Actual Background** | TN = {cm['tn']:,} | FP = {cm['fp']:,} |",
        f"| **Actual Pipe**       | FN = {cm['fn']:,} | TP = {cm['tp']:,} |",
        "",
        f"- **Total pixels evaluated:** {total:,}",
        f"- **Global Accuracy:**    {acc:.4f}",
        f"- **Global Precision:**   {prec:.4f}",
        f"- **Global Recall:**      {rec:.4f}",
        f"- **Global Specificity:** {spec:.4f}",
        f"- **Global F1:**          {f1_cm:.4f}",
        "",
        "---",
        "",
        "## 3. Failure Case Analysis",
        "",
        "Images with the lowest IoU (model struggled most):",
        "",
        "| Rank | Image | IoU | Dice | Precision | Recall |",
        "|------|-------|-----|------|-----------|--------|",
    ]

    for i, m in enumerate(sorted(metrics, key=lambda x: x['iou'])[:10], 1):
        lines.append(
            f"| {i} | `{Path(m['name']).stem}` | {m['iou']:.4f} | "
            f"{m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Success Case Analysis",
        "",
        "Images with the highest IoU (best predictions):",
        "",
        "| Rank | Image | IoU | Dice | Precision | Recall |",
        "|------|-------|-----|------|-----------|--------|",
    ]

    for i, m in enumerate(sorted(metrics, key=lambda x: x['iou'], reverse=True)[:10], 1):
        lines.append(
            f"| {i} | `{Path(m['name']).stem}` | {m['iou']:.4f} | "
            f"{m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Model Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Architecture | U-Net |",
        "| Encoder | ResNet-34 (ImageNet pretrained) |",
        "| Decoder attention | scSE |",
        "| Input size | 512 × 512 |",
        "| Parameters | 31,043,521 |",
        "| Loss function | Dice Loss (binary) |",
        "| Augmentations | HFlip, VFlip, Rotate90, ShiftScaleRotate, ImageNet Normalise |",
        "",
        "---",
        "",
        "## 6. Generated Plots",
        "",
        "| # | File | Description |",
        "|---|------|-------------|",
        "| 1 | `01_metric_distributions.png` | Per-image IoU/Dice/P/R/F1/Specificity histograms |",
        "| 2 | `02_precision_recall_curve.png` | Global PR curve with AP and best-F1 point |",
        "| 3 | `03_roc_curve.png` | ROC curve with AUC |",
        "| 4 | `04_confusion_matrix.png` | Pixel confusion matrix (counts + normalised) |",
        "| 5 | `05_iou_dice_scatter.png` | IoU vs Dice scatter coloured by precision |",
        "| 6 | `06_summary_bars.png` | Mean ± std bar chart for all metrics |",
        "| 7 | `07_failure_cases.png` | Worst-IoU images with TP/FP/FN error maps |",
        "| 8 | `08_success_cases.png` | Best-IoU images with prediction overlays |",
        "| 9 | `09_raw_vs_prediction_comparison.png` | Raw / GT / Prediction / Heatmap grid |",
        "| 10 | `10_threshold_analysis.png` | All metrics swept across thresholds 0.1–0.95 |",
        "| 11 | `11_iou_ranking.png` | All images ranked by IoU |",
        "| 12 | `12_mask_coverage_analysis.png` | Pipe coverage fraction vs IoU |",
        "",
        "---",
        "",
        "*Auto-generated by `evaluate_model.py`*",
    ]

    path = os.path.join(OUTPUT_DIR, 'validation_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")
    return path


# ─────────────────────────── MAIN ─────────────────────────────────────────── #
def main():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 65)
    print("  Comprehensive U-Net Evaluation")
    print(f"  {timestamp}")
    print("=" * 65)

    print("\n[1/6] Loading model…")
    model = load_model(MODEL_PATH)
    print(f"  Loaded: {MODEL_PATH}  (device: {DEVICE})")

    print("\n[2/6] Running full dataset inference…")
    results = run_full_inference(model)

    print("\n[3/6] Computing per-image metrics…")
    metrics = compute_per_image_metrics(results)
    agg     = aggregate_metrics(metrics)
    print(f"  Mean IoU  = {agg['iou']['mean']:.4f}")
    print(f"  Mean Dice = {agg['dice']['mean']:.4f}")
    print(f"  Mean F1   = {agg['f1']['mean']:.4f}")

    print("\n[4/6] Generating plots…")
    plot_metric_distributions(metrics)
    _, ap      = plot_precision_recall_curve(results)
    _, roc_auc = plot_roc_curve(results)
    plot_confusion_matrix(agg)
    plot_iou_dice_scatter(metrics)
    plot_summary_bars(agg)
    plot_failure_cases(results, metrics)
    plot_success_cases(results, metrics)
    plot_raw_vs_prediction(results, metrics)
    _, best_thr = plot_threshold_analysis(results)
    plot_iou_ranking(metrics)
    plot_mask_coverage(results, metrics)

    print("\n[5/6] Writing validation report…")
    write_markdown_report(agg, ap, roc_auc, best_thr, metrics, timestamp)

    print("\n[6/6] Saving raw metrics JSON…")
    json_path = os.path.join(OUTPUT_DIR, 'per_image_metrics.json')
    with open(json_path, 'w') as f:
        json.dump({'aggregate': agg, 'per_image': metrics}, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print(f"  Results → {OUTPUT_DIR}/")
    print("=" * 65)
    print(f"\n  Mean IoU       : {agg['iou']['mean']:.4f} ± {agg['iou']['std']:.4f}")
    print(f"  Mean Dice      : {agg['dice']['mean']:.4f} ± {agg['dice']['std']:.4f}")
    print(f"  Mean Precision : {agg['precision']['mean']:.4f}")
    print(f"  Mean Recall    : {agg['recall']['mean']:.4f}")
    print(f"  Mean F1        : {agg['f1']['mean']:.4f}")
    print(f"  ROC AUC        : {roc_auc:.4f}")
    print(f"  Global AP      : {ap:.4f}")
    print(f"  Best threshold : {best_thr:.3f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
