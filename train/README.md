   # U-Net Pipe Segmentation Training

This directory contains the training code for a U-Net-based semantic segmentation model designed to detect and segment underwater pipes in images.

## Overview

The training notebook (`train-u-net.ipynb`) implements a U-Net architecture for binary semantic segmentation, where the model learns to predict pixel-wise masks indicating the presence of pipes in underwater images.

## Results 

![alt text](<docs/Screenshot from 2026-01-27 13-15-09.png>)

![alt text](<docs/Screenshot from 2026-01-27 13-13-31.png>)
![alt text](<docs/Screenshot from 2026-01-27 13-12-23.png>)

![alt text](<docs/Screenshot from 2026-01-27 13-11-01.png>)

![alt text](<docs/Screenshot from 2026-01-27 13-10-48.png>)

## Dataset

### Format
- **Dataset Format**: COCO JSON annotation format
- **Dataset Location**: `data/lhsealine_pipe_seg_train_coco/`
- **Structure**:
  ```
  data/lhsealine_pipe_seg_train_coco/
  ├── images/
  │   └── Train/          # Training images
  └── annotations/
      └── instances_Train.json  # COCO format annotations
  ```

### Dataset Details
- **Total Images**: 100 annotated images
- **Annotation Format**: COCO JSON with polygon/mask annotations
- **Image Format**: RGB images (converted to tensors, normalized [0,1])
- **Mask Format**: Binary masks (0 = background, 1 = pipe)

### Data Loading
The `PipeSegmentationDataset` class:
- Loads images from the COCO dataset
- Converts COCO annotations to binary segmentation masks
- Handles multiple annotations per image by accumulating masks
- Returns image-mask pairs as PyTorch tensors

## Model Architecture

### U-Net Structure

The U-Net architecture consists of:

1. **Encoder Path (Downsampling)**:
   - 3 downsampling blocks with feature channels: [32, 64, 128]
   - Each block: `DoubleConv` → `MaxPool2d`
   - Progressively reduces spatial dimensions while increasing channels

2. **Bottleneck**:
   - `DoubleConv` block with 256 channels
   - Processes the most compressed representation

3. **Decoder Path (Upsampling)**:
   - 3 upsampling blocks mirroring the encoder
   - Each block: `ConvTranspose2d` → concatenate skip connection → `DoubleConv`
   - Progressively increases spatial dimensions while decreasing channels

4. **Output Layer**:
   - Final 1×1 convolution producing single-channel output (binary mask)

### DoubleConv Block

Each `DoubleConv` block consists of:
```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU
```

This double convolution pattern helps the model learn richer feature representations at each level.

### Skip Connections

The U-Net uses skip connections between encoder and decoder at corresponding levels:
- Preserves fine-grained spatial information
- Helps with precise boundary localization
- Enables the model to combine high-level semantic features with low-level details

## Training Configuration

### Hyperparameters
- **Learning Rate**: 1e-4 (0.0001)
- **Optimizer**: Adam
- **Batch Size**: 1 (to avoid CUDA out-of-memory errors)
- **Number of Epochs**: 10
- **Device**: CUDA (GPU) if available, else CPU

### Loss Function

The training uses a combined loss function:

1. **BCE Loss** (`BCEWithLogitsLoss`):
   - Binary cross-entropy loss for pixel-wise classification
   - Handles the binary nature of the segmentation task

2. **Dice Loss**:
   - Measures overlap between predicted and ground truth masks
   - Particularly effective for imbalanced segmentation tasks
   - Formula: `Dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)`
   - Loss = `1 - Dice`

**Combined Loss**: `BCE Loss + Dice Loss`

This combination leverages the strengths of both losses:
- BCE provides stable gradients
- Dice focuses on overlap and handles class imbalance

### Training Process

1. **Data Loading**:
   - Images and masks are loaded in batches
   - DataLoader with `shuffle=True` for randomization

2. **Forward Pass**:
   - Images pass through the U-Net encoder-decoder
   - Outputs are logits (before sigmoid)

3. **Loss Calculation**:
   - Combined BCE + Dice loss computed
   - Loss averaged over batch

4. **Backward Pass**:
   - Gradients computed via backpropagation
   - Optimizer updates model weights

5. **Memory Management**:
   - Explicit cleanup after each batch (`del` variables)
   - `torch.cuda.empty_cache()` to free GPU memory
   - Critical for training with limited GPU memory

## Usage

### Prerequisites

Install required dependencies:
```bash
pip install torch torchvision pycocotools matplotlib tqdm
```

### Running the Training

1. **Open the notebook**:
   ```bash
   jupyter notebook train-u-net.ipynb
   ```

2. **Execute cells in order**:
   - Cell 0: Install dependencies and load COCO dataset
   - Cell 1: Example mask creation (optional, for understanding)
   - Cell 2: Define dataset class and create DataLoader
   - Cell 3: Define U-Net model architecture
   - Cell 4: Define loss functions and optimizer
   - Cell 5: Check GPU availability and memory
   - Cell 6: Training loop

3. **Training Output**:
   - Progress bar showing epoch and batch progress
   - Average loss printed after each epoch

4. **Save Model** (Cell 7):
   ```python
   torch.save(model.state_dict(), "unet_pipe_segmentation.pth")
   ```

5. **Load Model** (Cell 8):
   ```python
   model.load_state_dict(torch.load("unet_pipe_segmentation.pth"))
   model.eval()
   ```

6. **Inference and Visualization** (Cell 9):
   - Visualize predictions on test images
   - Compare input image, true mask, and predicted mask

7. **Evaluation Metrics** (Cell 10):
   - Dice coefficient function for quantitative evaluation

## Model Checkpoints

- **Saved Model**: `unet_pipe_segmentation.pth`
- **Format**: PyTorch state dictionary (`.pth`)
- **Contents**: Model weights and parameters

To resume training or use for inference:
```python
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("unet_pipe_segmentation.pth"))
model.eval()  # Set to evaluation mode
```

## Technical Details

### Architecture Fixes

The notebook includes a critical fix for the decoder channel mismatch:
- **Issue**: Decoder layers expected incorrect input channel counts
- **Solution**: Properly track channel dimensions through bottleneck and decoder
- **Key Fix**: `DoubleConv(ch * 2, ch)` in decoder to handle concatenated skip connections

### Memory Optimization

Several strategies are used to manage GPU memory:
- **Batch size = 1**: Minimizes memory footprint
- **Explicit cleanup**: Delete tensors after use
- **Cache clearing**: `torch.cuda.empty_cache()` after each batch
- **No data workers**: `num_workers=0` to avoid memory duplication

### Image Processing

- Images are converted to PyTorch tensors with shape `[C, H, W]`
- Normalized to range [0, 1] via `ToTensor()`
- Masks are binary (0 or 1) with shape `[1, H, W]`
- Spatial dimensions preserved (no resizing in current implementation)

## Evaluation

### Dice Coefficient

The Dice coefficient measures segmentation overlap:
```python
Dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
```

- Range: [0, 1] where 1 = perfect overlap
- Useful for imbalanced datasets (small pipe regions vs large background)

### Visualization

The notebook includes visualization code to:
- Display input images
- Show ground truth masks
- Display predicted masks
- Compare predictions with ground truth

## Future Improvements

Potential enhancements:
- Data augmentation (rotation, flipping, color jitter)
- Learning rate scheduling
- Validation split for monitoring overfitting
- Early stopping based on validation metrics
- Multi-scale training
- Test-time augmentation
- Additional evaluation metrics (IoU, precision, recall)

## Notes

- The model is trained on a relatively small dataset (100 images)
- Consider data augmentation to improve generalization
- Monitor for overfitting with such a small dataset
- Batch size of 1 is used due to memory constraints but slows training
- The model architecture can be adjusted via the `features` parameter in UNet initialization

## References

- **U-Net Paper**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **COCO Dataset Format**: https://cocodataset.org/#format-data
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
