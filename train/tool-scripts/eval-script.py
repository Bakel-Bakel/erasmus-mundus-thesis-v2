import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Unettrain import CocoSegmentationDataset  # or your custom dataset class
from Unettrain import UNet  # import your UNet definition
from Unettrain import get_transforms  # import the transform function

from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm

# --------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_unet.pth"
TEST_IMAGES_DIR = "./1/images/Train"  # adjust as needed
TEST_ANN_PATH = "./1/annotations/instances_Train.json"  # adjust
BATCH_SIZE = 1  # Evaluation usually done per-image
THRESHOLD = 0.5  # For binary segmentation output
# ---------------------------

# Dice Score
def dice_score(y_true, y_pred):
    smooth = 1e-6
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

# Load dataset
test_dataset = CocoSegmentationDataset(TEST_IMAGES_DIR, TEST_ANN_PATH, transforms=get_transforms())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = UNet(in_channels=3, out_channels=1)  # adjust if multiclass
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluate
ious = []
dices = []

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        preds = (torch.sigmoid(outputs) > THRESHOLD).float()

        for pred, mask in zip(preds, masks):
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)

            iou = jaccard_score(mask_np.flatten(), pred_np.flatten(), zero_division=1)
            dice = dice_score(mask_np, pred_np)

            ious.append(iou)
            dices.append(dice)

print(f"Avg IoU: {np.mean(ious) * 100:.2f}%")
print(f"Avg Dice Score: {np.mean(dices) * 100:.2f}%")
