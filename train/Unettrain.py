import os
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
import logging
from datetime import datetime
import time

# ==== Dataset Loader ====
class CocoSegmentationDataset(Dataset):
    def __init__(self, images_dir, ann_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask |= self.coco.annToMask(ann)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask.float()

# ==== Transforms ====
def get_transforms(augment=False):
    def _transform(image, mask):
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
        
        return image, mask
    return _transform

# ==== U-Net ====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Deeper UNet with 4 levels for better feature extraction on high-res images
        self.downs = nn.ModuleList([
            DoubleConv(3, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2)
        ])
        self.bottleneck = DoubleConv(512, 1024)

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ])
        self.up_convs = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64),
        ])
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final(x)

# ==== Loss ====
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)  # pt = p if target==1 else 1-p
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def loss_fn(pred, target):
    # Weight dice loss more heavily (2x) as it's better for segmentation
    # Add focal loss to handle class imbalance
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return bce_loss + 2.0 * dice + 0.5 * focal

# ==== Metrics ====
def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2. * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)

# ==== Training ====
def train(model, loader, optimizer, device, gradient_accumulation_steps=1, epoch=0, logger=None):
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    epoch_dice = 0
    batch_losses = []
    batch_ious = []
    batch_dices = []
    gradient_norms = []
    
    optimizer.zero_grad()
    batch_start_time = time.time()
    batch_idx = -1  # Initialize to handle edge case of empty loader
    
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training")):
        batch_iter_start = time.time()
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Scale loss by accumulation steps
        loss_scaled = loss / gradient_accumulation_steps
        loss_scaled.backward()
        
        batch_loss = loss.item() * gradient_accumulation_steps  # Unscale for logging
        epoch_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Calculate metrics
        with torch.no_grad():
            batch_iou = calculate_iou(outputs, masks).item()
            batch_dice = calculate_dice(outputs, masks).item()
            epoch_iou += batch_iou
            epoch_dice += batch_dice
            batch_ious.append(batch_iou)
            batch_dices.append(batch_dice)
        
        # Update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_norms.append(total_norm.item())
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Log batch update
            batch_time = time.time() - batch_iter_start
            if logger:
                logger.info(f"  [TRAIN] Batch {batch_idx+1}/{len(loader)} | "
                          f"Loss: {batch_loss:.6f} | IoU: {batch_iou:.6f} | "
                          f"Dice: {batch_dice:.6f} | GradNorm: {total_norm.item():.6f} | "
                          f"Time: {batch_time:.3f}s")
    
    # Handle remaining gradients if batch count is not divisible by accumulation steps
    if batch_idx >= 0 and (batch_idx + 1) % gradient_accumulation_steps != 0:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        gradient_norms.append(total_norm.item())
        optimizer.step()
        optimizer.zero_grad()
        if logger:
            logger.info(f"  [TRAIN] Final batch update | GradNorm: {total_norm.item():.6f}")
    
    avg_loss = epoch_loss / len(loader)
    avg_iou = epoch_iou / len(loader)
    avg_dice = epoch_dice / len(loader)
    avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
    
    if logger:
        min_loss = min(batch_losses) if batch_losses else 0.0
        max_loss = max(batch_losses) if batch_losses else 0.0
        min_iou = min(batch_ious) if batch_ious else 0.0
        max_iou = max(batch_ious) if batch_ious else 0.0
        logger.info(f"  [TRAIN EPOCH SUMMARY] Avg Loss: {avg_loss:.6f} | "
                   f"Avg IoU: {avg_iou:.6f} | Avg Dice: {avg_dice:.6f} | "
                   f"Avg GradNorm: {avg_grad_norm:.6f} | "
                   f"Min Loss: {min_loss:.6f} | Max Loss: {max_loss:.6f} | "
                   f"Min IoU: {min_iou:.6f} | Max IoU: {max_iou:.6f}")
    
    return avg_loss, avg_iou, avg_dice

# ==== Validation ====
def validate(model, loader, device, epoch=0, logger=None):
    model.eval()
    epoch_loss = 0
    epoch_iou = 0
    epoch_dice = 0
    batch_losses = []
    batch_ious = []
    batch_dices = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validating")):
            batch_start = time.time()
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Calculate metrics
            batch_iou = calculate_iou(outputs, masks).item()
            batch_dice = calculate_dice(outputs, masks).item()
            epoch_iou += batch_iou
            epoch_dice += batch_dice
            batch_ious.append(batch_iou)
            batch_dices.append(batch_dice)
            
            batch_time = time.time() - batch_start
            if logger:
                logger.info(f"  [VAL] Batch {batch_idx+1}/{len(loader)} | "
                          f"Loss: {batch_loss:.6f} | IoU: {batch_iou:.6f} | "
                          f"Dice: {batch_dice:.6f} | Time: {batch_time:.3f}s")
    
    avg_loss = epoch_loss / len(loader)
    avg_iou = epoch_iou / len(loader)
    avg_dice = epoch_dice / len(loader)
    
    if logger:
        min_loss = min(batch_losses) if batch_losses else 0.0
        max_loss = max(batch_losses) if batch_losses else 0.0
        min_iou = min(batch_ious) if batch_ious else 0.0
        max_iou = max(batch_ious) if batch_ious else 0.0
        logger.info(f"  [VAL EPOCH SUMMARY] Avg Loss: {avg_loss:.6f} | "
                   f"Avg IoU: {avg_iou:.6f} | Avg Dice: {avg_dice:.6f} | "
                   f"Min Loss: {min_loss:.6f} | Max Loss: {max_loss:.6f} | "
                   f"Min IoU: {min_iou:.6f} | Max IoU: {max_iou:.6f}")
    
    return avg_loss, avg_iou, avg_dice

# ==== Setup ====
if __name__ == "__main__":
    # Setup logging
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    
    # Configuration
    data_dir = "merged_dataset"
    images_dir = os.path.join(data_dir, "images", "Train")
    ann_file = os.path.join(data_dir, "annotations/instances_Train_fixed.json")
    
    # Training hyperparameters
    BATCH_SIZE = 1  # Reduced due to large image size (1920x1080) and GPU memory constraints
    NUM_EPOCHS = 100  # More epochs for deeper model
    LEARNING_RATE = 3e-4  # Slightly higher LR for deeper model
    VAL_SPLIT = 0.2  # 20% for validation
    PATIENCE = 15  # More patience for deeper model
    MIN_DELTA = 0.0005  # Smaller delta for better convergence
    GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate batch_size=4

    logger.info("="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Annotation file: {ann_file}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Number of epochs: {NUM_EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Validation split: {VAL_SPLIT}")
    logger.info(f"Early stopping patience: {PATIENCE}")
    logger.info(f"Minimum delta: {MIN_DELTA}")
    logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("Running on CPU")
    
    # Load full dataset to get size
    logger.info("="*80)
    logger.info("LOADING DATASET")
    logger.info("="*80)
    logger.info(f"Loading dataset from: {ann_file}")
    temp_dataset = CocoSegmentationDataset(images_dir, ann_file, transforms=None)
    dataset_size = len(temp_dataset)
    logger.info(f"Total dataset size: {dataset_size} samples")
    
    # Split into train and validation indices
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    logger.info(f"Random seed: 42 (for reproducibility)")
    
    # Create separate train and validation datasets with different transforms
    logger.info("Creating train dataset with augmentation...")
    train_dataset = CocoSegmentationDataset(images_dir, ann_file, transforms=get_transforms(augment=True))
    logger.info("Creating validation dataset without augmentation...")
    val_dataset = CocoSegmentationDataset(images_dir, ann_file, transforms=get_transforms(augment=False))
    
    # Create subsets with the split indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    # Use num_workers=0 if on Windows or if there are issues, otherwise use 2
    num_workers = 2 if torch.cuda.is_available() else 0
    logger.info(f"Creating data loaders with num_workers={num_workers}, pin_memory={torch.cuda.is_available()}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    logger.info("="*80)
    logger.info("INITIALIZING MODEL")
    logger.info("="*80)
    logger.info("Creating UNet model...")
    model = UNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Optimizer: Adam with lr={LEARNING_RATE}")
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    logger.info("Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)")

    # ==== Main Training Loop ====
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    previous_lr = LEARNING_RATE
    training_start_time = time.time()
    
    logger.info("="*80)
    logger.info("STARTING TRAINING LOOP")
    logger.info("="*80)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        logger.info("")
        logger.info("="*80)
        logger.info(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        logger.info("="*80)
        
        # Log GPU memory before epoch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        # Training
        logger.info(f"[EPOCH {epoch+1}] Starting training phase...")
        train_loss, train_iou, train_dice = train(model, train_loader, optimizer, device, 
                                                   GRADIENT_ACCUMULATION_STEPS, epoch, logger)
        
        # Validation
        logger.info(f"[EPOCH {epoch+1}] Starting validation phase...")
        val_loss, val_iou, val_dice = validate(model, val_loader, device, epoch, logger)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        logger.info("")
        logger.info("-"*80)
        logger.info(f"[EPOCH {epoch+1} SUMMARY]")
        logger.info("-"*80)
        logger.info(f"  Train - Loss: {train_loss:.6f}, IoU: {train_iou:.6f}, Dice: {train_dice:.6f}")
        logger.info(f"  Val   - Loss: {val_loss:.6f}, IoU: {val_iou:.6f}, Dice: {val_dice:.6f}")
        logger.info(f"  Learning Rate: {current_lr:.8f}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
        
        # Notify if learning rate was reduced
        if current_lr < previous_lr:
            logger.warning(f"  ⚠ LEARNING RATE REDUCED: {previous_lr:.8f} -> {current_lr:.8f}")
        previous_lr = current_lr
        
        # Save best model based on validation loss
        improvement = best_val_loss - val_loss
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_val_iou = val_iou
            patience_counter = 0
            model_path = "best_unet.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ✓ NEW BEST MODEL SAVED!")
            logger.info(f"    Model path: {model_path}")
            logger.info(f"    Validation Loss: {val_loss:.6f} (improvement: {improvement:.6f})")
            logger.info(f"    Validation IoU: {val_iou:.6f}")
            logger.info(f"    Validation Dice: {val_dice:.6f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement (patience: {patience_counter}/{PATIENCE})")
            logger.info(f"    Current Val Loss: {val_loss:.6f}")
            logger.info(f"    Best Val Loss: {best_val_loss:.6f}")
            logger.info(f"    Required improvement: {MIN_DELTA:.6f}, Actual: {improvement:.6f}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            total_time = time.time() - training_start_time
            logger.warning("")
            logger.warning("="*80)
            logger.warning("EARLY STOPPING TRIGGERED")
            logger.warning("="*80)
            logger.warning(f"Stopped after {epoch+1} epochs (patience limit: {PATIENCE})")
            logger.warning(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.warning(f"Best validation loss: {best_val_loss:.6f}")
            logger.warning(f"Best validation IoU: {best_val_iou:.6f}")
            break
    
    total_time = time.time() - training_start_time
    logger.info("")
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total epochs completed: {epoch+1}/{NUM_EPOCHS}")
    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per epoch: {total_time/(epoch+1):.2f}s")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best validation IoU: {best_val_iou:.6f}")
    logger.info(f"Best model saved to: best_unet.pth")
    logger.info("="*80)
