import os
import shutil
import random

# Set paths
train_img_dir = 'datasets/CV/images/train'
val_img_dir = 'datasets/CV/images/val'
train_lbl_dir = 'datasets/CV/labels/train'
val_lbl_dir = 'datasets/CV/labels/val'

# Move validation images to train
for file_name in os.listdir(val_img_dir):
    src = os.path.join(val_img_dir, file_name)
    dst = os.path.join(train_img_dir, file_name)
    shutil.move(src, dst)

# Move validation labels to train
for file_name in os.listdir(val_lbl_dir):
    src = os.path.join(val_lbl_dir, file_name)
    dst = os.path.join(train_lbl_dir, file_name)
    shutil.move(src, dst)

# Get list of image files
image_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.jpg') or f.endswith('.png')])

# Randomly select 20 for validation
val_files = set(random.sample(image_files, 20))

# Move files
for val_file in val_files:
    base_name = os.path.splitext(val_file)[0]
    label_file = base_name + '.txt'

    shutil.move(os.path.join(train_img_dir, val_file), os.path.join(val_img_dir, val_file))
    shutil.move(os.path.join(train_lbl_dir, label_file), os.path.join(val_lbl_dir, label_file))
