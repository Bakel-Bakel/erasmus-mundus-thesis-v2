import json

# Load your file
with open('merged_dataset/annotations/instances_Train.json', 'r') as f:
    data = json.load(f)

# 1. Move all annotations from ID 2 to ID 1
count = 0
for ann in data['annotations']:
    if ann['category_id'] == 2:
        ann['category_id'] = 1
        count += 1
print(f"Fixed {count} annotations.")

# 2. Now it is safe to delete Category ID 2 because nothing points to it
data['categories'] = [cat for cat in data['categories'] if cat['id']!= 2]

# Save the fixed file
with open('merged_dataset/annotations/instances_Train_fixed.json', 'w') as f:
    json.dump(data, f)