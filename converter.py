import os
import json

# Set your input and output paths
json_folder = 'dataset/CV_Train/Labels'
output_folder = 'dataset/CV_Train/YOLO_Labels'  # YOLO format labels will go here

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define class mapping
class_map = {
    "standing": 0,
    "sitting": 1,
    "lying": 2,
    "throwing": 3
}

# Loop through all JSON files
for filename in os.listdir(json_folder):
    if not filename.endswith('.json'):
        continue

    json_path = os.path.join(json_folder, filename)
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_w, img_h = data['imageWidth'], data['imageHeight']
    label_lines = []

    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            continue  # skip unknown labels

        class_id = class_map[label]
        (x1, y1), (x2, y2) = shape['points']
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Convert to YOLO format
        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        label_lines.append(label_line)

    # Save the YOLO .txt label file
    txt_filename = filename.replace('.json', '.txt')
    txt_path = os.path.join(output_folder, txt_filename)
    with open(txt_path, 'w') as txt_file:
        txt_file.write('\n'.join(label_lines))