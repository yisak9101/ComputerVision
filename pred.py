from ultralytics import YOLO
import os
import shutil
import random

model = YOLO("runs/detect/train3/weights/best.pt")
image_dir = "datasets/CV/images/test"

image_files = sorted([f"{image_dir}/{f}" for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
test_files = list(random.sample(image_files, 20))

results = model.predict(source=test_files, save=True)