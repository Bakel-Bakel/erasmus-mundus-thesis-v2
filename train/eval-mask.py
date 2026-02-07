import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
import random

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, AttributeError) as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib not available ({e}). Using cv2 for visualization instead.")

# --- CONFIGURATION ---
MODEL_PATH = 'best_pipe_unet_epoch_100.pth' # Replace with your actual.pth file name
IMAGE_DIR = 'merged_dataset/images/Train'             # Path to your images
MASK_DIR = 'merged_dataset/masks/Train'               # Path to your masks
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_dataset_iou(model):
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_iou = 0
    count = 0
    
    print(f"Calculating IoU for {len(all_images)} images...")
    
    for img_name in all_images:
        # Load and preprocess image
        img_path = os.path.join(IMAGE_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, img_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path):
            continue
        
        # Read and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (512, 512))
        
        # Convert to tensor and normalize with ImageNet stats (matching training)
        x = torch.from_numpy(img_input).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # CHW format, Batch dim
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.to(DEVICE)
        
        # Load ground truth mask
        true_mask = cv2.imread(mask_path, 0)
        if true_mask is None:
            continue
        true_mask = cv2.resize(true_mask, (512, 512))
        true_mask = (true_mask > 128).astype(np.float32)  # Binary mask
        
        # Run inference
        with torch.no_grad():
            pred = (model(x) > 0.5).float().squeeze().cpu().numpy()
        
        # Calculate IoU
        intersection = np.logical_and(true_mask, pred)
        union = np.logical_or(true_mask, pred)
        
        if np.sum(union) > 0:
            iou = np.sum(intersection) / np.sum(union)
            total_iou += iou
            count += 1
    
    if count > 0:
        avg_iou = total_iou / count
        print(f"Average Dataset IoU: {avg_iou:.4f} ({count} images processed)")
        return avg_iou
    else:
        print("No valid image-mask pairs found!")
        return 0.0

def load_model():
    # Must match the architecture used in training
    model = smp.Unet(
        encoder_name='resnet34', 
        encoder_weights=None, # We are loading our own weights
        classes=1, 
        activation='sigmoid',
        decoder_attention_type='scse'
    )
    # Load the weights you trained
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def visualize_prediction(model, image_name):
    # 1. Load and Preprocess Image
    img_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, image_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
    
    # Read Image
    original_img = cv2.imread(img_path)
    if original_img is None: return
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize to 512x512 (Model Requirement)
    img_input = cv2.resize(original_img, (512, 512))
    
    # Convert to Tensor and normalize with ImageNet stats (matching training)
    x = torch.from_numpy(img_input).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # CHW format, Batch dim
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(DEVICE)

    # 2. Run Inference
    with torch.no_grad():
        pred_mask = model(x)
        # Convert sigmoid output to Binary (0 or 1)
        pred_mask = (pred_mask > 0.5).float()
        
    # 3. Process Result for Display
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Load Ground Truth (If it exists)
    if os.path.exists(mask_path):
        true_mask = cv2.imread(mask_path, 0)
        true_mask = cv2.resize(true_mask, (512, 512))
    else:
        true_mask = np.zeros((512, 512)) # Black mask if no annotation

    # 4. Visualize
    if HAS_MATPLOTLIB:
        # Use matplotlib if available
        plt.figure(figsize=(15, 5))
        
        # Original
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_input)
        plt.axis('off')

        # Ground Truth
        plt.subplot(1, 3, 2)
        plt.title("Your Annotation (Ground Truth)")
        plt.imshow(true_mask, cmap='gray')
        plt.axis('off')

        # Prediction
        plt.subplot(1, 3, 3)
        plt.title("AI Prediction")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Fallback to cv2 visualization
        # Convert masks to 0-255 range for display
        true_mask_display = (true_mask > 128).astype(np.uint8) * 255
        pred_mask_display = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Create side-by-side visualization
        vis_img = np.hstack([
            img_input,
            cv2.cvtColor(true_mask_display, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(pred_mask_display, cv2.COLOR_GRAY2RGB)
        ])
        
        # Add text labels
        cv2.putText(vis_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, "Ground Truth", (img_input.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, "Prediction", (img_input.shape[1] * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize if too large for display
        if vis_img.shape[1] > 1920:
            scale = 1920 / vis_img.shape[1]
            new_h = int(vis_img.shape[0] * scale)
            vis_img = cv2.resize(vis_img, (1920, new_h))
        
        cv2.imshow("Evaluation: Original | Ground Truth | Prediction", vis_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model()
    print("Model loaded! Displaying random examples...")
    calculate_dataset_iou(model)
    # Get list of images
    all_images = os.listdir(IMAGE_DIR)
    
    # Show 5 random examples
    for _ in range(5):
        random_img = random.choice(all_images)
        print(f"Verifying: {random_img}")
        visualize_prediction(model, random_img)

    