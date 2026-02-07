import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
ENCODER = 'resnet34'        # Stronger feature extractor than vanilla U-Net
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Pipe']          # Your specific class
ACTIVATION = 'sigmoid'      # 'sigmoid' for binary segmentation (pipe vs water)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.0001
EPOCHS = 100
IMG_SIZE = 512              # U-Net is heavy; 512x512 is standard (vs 1280 for YOLO)

# --- 1. THE DATASET CLASS ---
class PipeDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None):
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Filter to only include images that have corresponding mask files
        self.images_fps = []
        self.masks_fps = []
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            # Convert image extension to .png for mask
            base, ext = os.path.splitext(image_file)
            mask_file = base + ".png"
            mask_path = os.path.join(masks_dir, mask_file)
            
            # Only include if both image and mask exist
            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.images_fps.append(image_path)
                self.masks_fps.append(mask_path)
            else:
                print(f"Warning: Skipping {image_file} - mask file not found: {mask_path}")
        
        self.augmentation = augmentation
        print(f"Dataset initialized with {len(self.images_fps)} valid image-mask pairs")

    def __getitem__(self, i):
        # Read image and mask
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise ValueError(f"Failed to read image: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks_fps[i], 0)  # Read as grayscale
        if mask is None:
            raise ValueError(f"Failed to read mask: {self.masks_fps[i]}")
        
        # Binary mask: Pipe pixels = 1, Background = 0
        mask = np.where(mask > 128, 1.0, 0.0).astype(np.float32)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # After ToTensorV2, mask should be a tensor
            # Add channel dimension if it's 2D (H, W) -> (1, H, W)
            if isinstance(mask, torch.Tensor) and len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # If no augmentation, convert to tensor and add channel dim
            mask = torch.from_numpy(mask).unsqueeze(0)
            
        return image, mask

    def __len__(self):
        return len(self.images_fps)

# --- 2. AUGMENTATION (Protecting your CLAHE colors) ---
def get_training_augmentation():
    train_transform = [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

# --- 3. TRAINING LOOP ---
def train_model():
    # A. Model: Use Attention U-Net or standard U-Net with ResNet backbone
    # 'UnetPlusPlus' or 'Unet' with attention_type='scse' is SOTA for this
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
        decoder_attention_type='scse' # Spatial & Channel "Attention" to ignore marine snow
    )
    
    # B. Loss Function: DiceLoss is MANDATORY for thin pipes
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # C. Data Loaders (Replace paths with your actual folders)
    train_dataset = PipeDataset('merged_dataset/images/Train', 'merged_dataset/masks/Train', get_training_augmentation())
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    print(f"Training on {DEVICE} with {len(train_dataset)} images...")
    model.to(DEVICE)

    # D. Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate Dice Loss
            loss = loss_fn(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch: {epoch+1}, Dice Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save best model
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'./best_pipe_unet_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_model()