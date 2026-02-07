import os
import json
import shutil
from collections import defaultdict

# === EDIT THESE ===
input_dirs = ['1', '2','3','4']  # your dataset folders
output_dir = 'merged_dataset'
output_json = os.path.join(output_dir, 'annotations/instances_Train.json')
output_images_dir = os.path.join(output_dir,  'images', 'Train')
os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# === ID Offsets ===
img_id_offset = 0
ann_id_offset = 0

merged = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

used_image_filenames = set()
category_map = {}  # maps category names to new ids
next_category_id = 1

for d in input_dirs:
    json_path = os.path.join(d, 'annotations', 'instances_Train.json')
    with open(json_path) as f:
        data = json.load(f)

    # Use first dataset's info and licenses
    if not merged["info"]:
        merged["info"] = data.get("info", {})
    if not merged["licenses"]:
        merged["licenses"] = data.get("licenses", [])

    # === Merge categories ===
    for cat in data["categories"]:
        key = (cat["name"], cat.get("supercategory", ""))
        if key not in category_map:
            cat_id = next_category_id
            category_map[key] = cat_id
            next_category_id += 1
            merged["categories"].append({
                "id": cat_id,
                "name": cat["name"],
                "supercategory": cat.get("supercategory", "")
            })

    # === Merge images ===
    img_id_map = {}
    for img in data["images"]:
        old_id = img["id"]
        file_name = img["file_name"]
        new_id = img_id_offset + old_id

        # Prevent duplicate filenames
        if file_name in used_image_filenames:
            base, ext = os.path.splitext(file_name)
            file_name = f"{base}_{new_id}{ext}"
        used_image_filenames.add(file_name)

        new_img = img.copy()
        new_img["id"] = new_id
        new_img["file_name"] = file_name
        merged["images"].append(new_img)
        img_id_map[old_id] = new_id

        # Copy image file
        src = os.path.join(d, 'images', 'Train', img["file_name"])
        dst = os.path.join(output_images_dir, file_name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    # === Merge annotations ===
    for ann in data["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = ann_id_offset + ann["id"]
        new_ann["image_id"] = img_id_map[ann["image_id"]]

        # remap category id
        cat_name = next((c["name"] for c in data["categories"] if c["id"] == ann["category_id"]), None)
        cat_super = next((c["supercategory"] for c in data["categories"] if c["id"] == ann["category_id"]), "")
        new_ann["category_id"] = category_map[(cat_name, cat_super)]

        merged["annotations"].append(new_ann)

    img_id_offset = max(img["id"] for img in merged["images"]) + 1
    ann_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

# === Save merged JSON ===
with open(output_json, 'w') as f:
    json.dump(merged, f, indent=2)

print(f"\n Merged dataset saved to: {output_json}")
print(f" Images copied to: {output_images_dir}")
