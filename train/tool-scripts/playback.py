import os
import cv2
import json
import time
from pycocotools.coco import COCO
import numpy as np

# === Edit these ===
json_path = "merged_dataset/annotations/instances_Train.json"
images_dir = "merged_dataset/images/Train"
frame_delay = 50  # milliseconds between frames

# === Load COCO annotations ===
coco = COCO(json_path)
img_ids = coco.getImgIds()
cat_ids = coco.getCatIds()

# Pick colors per category
colors = {}
for cid in cat_ids:
    colors[cid] = np.random.randint(0, 255, size=3).tolist()

# === Play through images like a video ===
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(images_dir, img_info["file_name"])
    image = cv2.imread(img_path)

    if image is None:
        print(f"Missing image: {img_info['file_name']}")
        continue

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if 'segmentation' in ann:
            pts = np.array(ann['segmentation'][0]).reshape(-1, 2).astype(np.int32)
            color = colors[ann['category_id']]
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(image, str(ann['category_id']), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Playback with Annotations", image)
    key = cv2.waitKey(frame_delay)
    if key == 27:  # ESC to stop
        break

cv2.destroyAllWindows()
