import cv2, numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

img_dir = "lhsealine_pipe_seg_train_coco/images/Train"
ann_file = "lhsealine_pipe_seg_train_coco/annotations/instances_Train.json"

coco = COCO(ann_file)

img_ids = coco.getImgIds()
img_id = img_ids[50]  # 10th (0-based index)
img_info = coco.loadImgs(img_id)[0]

img_path = f"{img_dir}/{img_info['file_name']}"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

mask_all = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

for ann in anns:
    seg = ann["segmentation"]

    # polygons -> RLE(s)
    rles = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
    rle = maskUtils.merge(rles) if isinstance(rles, list) else rles

    m = maskUtils.decode(rle)  # can be (H,W) or (H,W,1)
    if m.ndim == 3:
        m = m[:, :, 0]

    mask_all = np.maximum(mask_all, (m > 0).astype(np.uint8))

# overlay yellow where mask is 1
img[mask_all == 1] = (0, 255, 255)

print("Showing:", img_info["file_name"], "id:", img_id)
cv2.imshow("mask check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
