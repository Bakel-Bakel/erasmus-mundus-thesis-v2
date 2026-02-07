import json
import numpy as np
import cv2
import os
from pycocotools.coco import COCO

# --- CONFIGURATION ---
ANNOTATION_FILE = 'merged_dataset/annotations/instances_Train_fixed.json' # e.g., 'result.json' or 'instances_default.json'
IMAGE_DIR = 'merged_dataset/images/Train'                 # Where your original.jpg images are
OUTPUT_MASK_DIR = 'merged_dataset/masks/Train'              # Where the U-Net needs the masks
TARGET_CATEGORY = 'Pipe'                          # The exact name of your class in the JSON

def create_masks():
    if not os.path.exists(OUTPUT_MASK_DIR):
        os.makedirs(OUTPUT_MASK_DIR)

    # Initialize COCO api
    coco = COCO(ANNOTATION_FILE)
    
    # Get ID for the target category
    catIds = coco.getCatIds(catNms=[TARGET_CATEGORY])
    if not catIds:
        print(f"Error: Category '{TARGET_CATEGORY}' not found in JSON.")
        return
        
    # Get all image IDs containing the pipe
    imgIds = coco.getImgIds(catIds=catIds)

    print(f"Found {len(imgIds)} images with {TARGET_CATEGORY}. Generating masks...")

    for imgId in imgIds:
        # coco.loadImgs returns a list; take the first (and only) element
        img_info = coco.loadImgs(imgId)[0]
        filename = img_info['file_name']
        
        # 1. Create a black canvas (Same size as original image)
        # We need the height/width from the JSON to create the empty mask
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)

        # 2. Get annotations for this image
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # 3. Draw the annotations onto the mask
        for ann in anns:
            # COCO provides a method to turn polygons into a binary mask
            pixel_mask = coco.annToMask(ann)
            
            # Combine this object with the main mask (logical OR)
            # Use 255 for White (Pipe), 0 for Black (Background)
            mask = np.maximum(mask, pixel_mask * 255)

        # 4. Save the mask
        # Ensure the mask filename matches the image filename EXACTLY (but with .png extension)
        # The U-Net code expects: image="img1.jpg" -> mask="img1.png"
        base, _ = os.path.splitext(filename)
        mask_filename = base + ".png"
        save_path = os.path.join(OUTPUT_MASK_DIR, mask_filename)
        
        cv2.imwrite(save_path, mask)
        
    print(f"Done! Saved {len(imgIds)} masks to {OUTPUT_MASK_DIR}")

if __name__ == '__main__':
    # You might need to install this library first:
    # pip install pycocotools
    create_masks()