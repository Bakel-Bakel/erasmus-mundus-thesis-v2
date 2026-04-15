# Model Architecture — U-Net with ResNet-34 Encoder and scSE Attention

## 1. Overview

| Property | Value |
|----------|-------|
| Framework | `segmentation_models_pytorch` (SMP) |
| Architecture | U-Net |
| Encoder | ResNet-34 |
| Encoder pre-training | ImageNet |
| Decoder attention | scSE (Spatial & Channel Squeeze-Excitation) |
| Input channels | 3 (RGB) |
| Output classes | 1 (binary: pipe / background) |
| Output activation | Sigmoid |
| Input resolution | 512 × 512 px |
| **Total parameters** | **24,550,360** |
| Trainable parameters | 24,550,360 (100 %) |

---

## 2. High-level Architecture

```
Input (3 × 512 × 512)
        │
┌───────▼────────────────────────────────┐
│           ENCODER  (ResNet-34)         │
│                                        │
│  stem   →  64ch  @ 256×256            │
│  layer1 →  64ch  @ 256×256  (×3 blks) │
│  layer2 → 128ch  @ 128×128  (×4 blks) │
│  layer3 → 256ch  @  64×64   (×6 blks) │
│  layer4 → 512ch  @  32×32   (×3 blks) │
└───────────────────────────────────────-┘
        │  skip connections (×5)
┌───────▼────────────────────────────────┐
│           DECODER  (5 blocks)          │
│  Each block:                           │
│    Upsample → concat skip → Conv×2     │
│    + scSE attention after each Conv    │
│                                        │
│  block 0: 512+256 → 256ch @ 64×64      │
│  block 1: 256+128 → 128ch @ 128×128   │
│  block 2: 128+ 64 →  64ch @ 256×256   │
│  block 3:  64+ 64 →  32ch @ 512×512   │
│  block 4:   3+ 32 →  16ch @ 512×512   │
└───────────────────────────────────────-┘
        │
┌───────▼──────────────────┐
│  Segmentation Head       │
│  Conv2d(16→1, 3×3)       │
│  Sigmoid                 │
└──────────────────────────┘
        │
Output (1 × 512 × 512)  — probability map [0, 1]
Binary mask via threshold (default 0.5)
```

---

## 3. Encoder — ResNet-34

ResNet-34 uses **Basic Blocks** (two 3×3 convolutions with a residual shortcut).  
It does **not** use bottleneck blocks (those are ResNet-50+).

### Stem

| Layer | Type | Kernel | In → Out | Stride | Params |
|-------|------|--------|----------|--------|--------|
| conv1 | Conv2d | 7×7 | 3 → 64 | 2 | 9,408 |
| bn1   | BatchNorm2d | — | 64 | — | 128 |
| maxpool | MaxPool2d | 3×3 | — | 2 | 0 |

Output: **64 ch @ 128×128**  *(with 512×512 input)*

### Layer 1 — 3 Basic Blocks

| Block | Layers | In → Out ch | Spatial | Params/block |
|-------|--------|-------------|---------|-------------|
| 1.0 | conv1(3×3) + bn + conv2(3×3) + bn | 64 → 64 | 128×128 | 73,984 |
| 1.1 | same | 64 → 64 | 128×128 | 73,984 |
| 1.2 | same | 64 → 64 | 128×128 | 73,984 |

**Layer 1 total: 221,952 params**  
Output: **64 ch @ 128×128**

### Layer 2 — 4 Basic Blocks (stride 2 at block 0)

| Block | In → Out ch | Spatial | Notes |
|-------|-------------|---------|-------|
| 2.0 | 64 → 128 | 64×64 | 1×1 downsample shortcut (8,192 + 384 params) |
| 2.1 | 128 → 128 | 64×64 | — |
| 2.2 | 128 → 128 | 64×64 | — |
| 2.3 | 128 → 128 | 64×64 | — |

**Layer 2 total: 1,117,184 params**  
Output: **128 ch @ 64×64**

### Layer 3 — 6 Basic Blocks (stride 2 at block 0)

| Block | In → Out ch | Spatial | Notes |
|-------|-------------|---------|-------|
| 3.0 | 128 → 256 | 32×32 | 1×1 downsample shortcut (32,768 + 768 params) |
| 3.1–3.5 | 256 → 256 | 32×32 | — |

**Layer 3 total: 6,819,840 params**  
Output: **256 ch @ 32×32**

### Layer 4 — 3 Basic Blocks (stride 2 at block 0)

| Block | In → Out ch | Spatial | Notes |
|-------|-------------|---------|-------|
| 4.0 | 256 → 512 | 16×16 | 1×1 downsample shortcut (131,072 + 2,048 params) |
| 4.1 | 512 → 512 | 16×16 | — |
| 4.2 | 512 → 512 | 16×16 | — |

**Layer 4 total: 14,964,736 params**  
Output: **512 ch @ 16×16**

### Encoder parameter summary

| Section | Params | % of encoder |
|---------|--------|-------------|
| Stem (conv1 + bn1) | 9,536 | 0.04 % |
| Layer 1 | 221,952 | 1.0 % |
| Layer 2 | 1,117,184 | 5.0 % |
| Layer 3 | 6,819,840 | 30.6 % |
| Layer 4 | 14,964,736 | 67.1 % |
| **Encoder total** | **~22,133,248** | **~90.1 %** |

---

## 4. Decoder — U-Net with scSE Attention

Each decoder block:
1. **Bilinear upsample** ×2
2. **Concatenate** with encoder skip connection
3. **Conv block 1**: Conv2d(3×3) → BatchNorm → ReLU → **scSE attention**
4. **Conv block 2**: Conv2d(3×3) → BatchNorm → ReLU → **scSE attention**

### scSE — Squeeze-Excitation

Each scSE module has two parallel branches:

- **cSE (Channel Squeeze-Excitation)**: GlobalAvgPool → FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid → channel-wise scale
- **sSE (Spatial Squeeze-Excitation)**: Conv2d(C→1, 1×1) → Sigmoid → spatial-wise scale

Both outputs are element-wise added and applied to the feature map.

### Decoder blocks

| Block | Skip source | Input ch (up+skip) | Output ch | Spatial | Params |
|-------|------------|---------------------|-----------|---------|--------|
| 0 | encoder.layer4 | 512 + 256 = 768 | 256 | 32×32 | ~2,479,297 |
| 1 | encoder.layer3 | 256 + 128 = 384 | 128 | 64×64 | ~611,097 |
| 2 | encoder.layer2 | 128 + 64  = 192 | 64  | 128×128 | ~152,901 |
| 3 | encoder.layer1 | 64  + 64  = 128 | 32  | 256×256 | ~48,453 |
| 4 | encoder.stem   | 32  + 3   = 35  | 16  | 512×512 | ~7,202 |

> Note: block 4 skip is from the raw input (3 ch), as SMP U-Net includes the original image as the shallowest skip.

**Decoder total: ~3,298,950 params (~13.4 %)**

---

## 5. Segmentation Head

| Layer | Type | Kernel | In → Out | Params |
|-------|------|--------|----------|--------|
| conv | Conv2d | 3×3 | 16 → 1 | 145 |
| activation | Sigmoid | — | — | 0 |

Output: **1 × 512 × 512** probability map in [0, 1].  
Binary mask = `output > 0.5`.

---

## 6. Complete Parameter Count by Section

| Section | Parameters | Percentage |
|---------|-----------|------------|
| Encoder (ResNet-34) | 21,797,672 | 88.8 % |
| Decoder (5 blocks + scSE) | 2,752,543 | 11.2 % |
| Segmentation head | 145 | <0.01 % |
| **Total** | **24,550,360** | **100 %** |

---

## 7. Feature Map Dimensions (512×512 input)

| Stage | Channels | Spatial | Scale vs input |
|-------|---------|---------|----------------|
| Input | 3 | 512×512 | 1× |
| After stem (conv1 + pool) | 64 | 128×128 | 1/4 |
| After layer1 | 64 | 128×128 | 1/4 |
| After layer2 | 128 | 64×64 | 1/8 |
| After layer3 | 256 | 32×32 | 1/16 |
| After layer4 (bottleneck) | 512 | 16×16 | 1/32 |
| After decoder block 0 | 256 | 32×32 | 1/16 |
| After decoder block 1 | 128 | 64×64 | 1/8 |
| After decoder block 2 | 64 | 128×128 | 1/4 |
| After decoder block 3 | 32 | 256×256 | 1/2 |
| After decoder block 4 | 16 | 512×512 | 1× |
| Output (segmentation head) | 1 | 512×512 | 1× |

---

## 8. Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss function | Dice Loss (binary, `from_logits=False`) |
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |
| Batch size | 8 |
| Epochs | 100 (full, no early stopping) |
| Validation split | 20 % |
| Input normalisation | ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) |
| Augmentations | HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate |
| Random seed | 42 |
| Device | CUDA (NVIDIA RTX 5080 Laptop GPU, 15.48 GB) |
| Best val IoU achieved | 0.9350 (epoch 96) |

---

## 9. Design Rationale

**ResNet-34 encoder**  
Chosen over lighter encoders (MobileNet, EfficientNet-B0) for its strong feature extraction capability on high-resolution underwater imagery. The residual connections prevent gradient vanishing across 34 layers.

**scSE decoder attention**  
Standard U-Net skip connections pass all spatial features equally. The scSE modules allow the decoder to:
- Suppress uninformative channels (cSE branch)
- Focus on spatially relevant regions such as thin pipe structures (sSE branch)  
This is particularly valuable for pipe segmentation where the target object can be narrow and partially occluded by marine debris or turbidity.

**Dice Loss**  
Binary cross-entropy treats all pixels equally, which is problematic when pipe pixels are a small minority of the image (class imbalance). Dice Loss directly optimises the overlap coefficient, making it insensitive to class imbalance and better suited for thin-structure segmentation.

**512×512 resolution**  
Balances model capacity (the encoder bottleneck at 16×16 still retains sufficient spatial resolution for thin pipe localisation) against GPU memory constraints at batch size 8.

---

*Generated from live model introspection — `segmentation_models_pytorch` v. installed.*
