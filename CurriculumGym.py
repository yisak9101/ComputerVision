import os
import shutil
import random
import os
import shutil

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou


class Transition:
    def __init__(self, state, next_state, done):
        self.state = state
        self.next_state = next_state
        self.action = None
        self.reward = None
        self.done = done



class CurriculumGym:
    def __init__(self):
        self.vision_model = YOLO("yolo11n.pt")
        self.val_size = 20
        self.img_size = 640
        self.n_actions = 3
        self.input_h = 224
        self.input_w = 224
        base_dir = os.getcwd()
        self.img_dir = os.path.join(base_dir, 'datasets', 'CV', 'images')
        self.lbl_dir = os.path.join(base_dir, 'datasets', 'CV', 'labels')
        self.curriculums = [
            {"data": os.path.join(base_dir, 'datasets', 'CV', '0.yaml'), "epochs": 5},
            {"data": os.path.join(base_dir, 'datasets', 'CV', '1.yaml'), "epochs": 5},
            {"data": os.path.join(base_dir, 'datasets', 'CV', '2.yaml'), "epochs": 5},
        ]
        self.best_checkpoint_path = "runs/detect/train/weights/best.pt"

        self.original_img_dir = os.path.join(self.img_dir, 'augmented')
        self.original_lbl_dir = os.path.join(self.lbl_dir, 'augmented')
        self.train_img_dir = os.path.join(self.img_dir, 'train')
        self.val_img_dir = os.path.join(self.img_dir, 'val')
        self.train_lbl_dir = os.path.join(self.lbl_dir, 'train')
        self.val_lbl_dir = os.path.join(self.lbl_dir, 'val')

        self.original_img_files = sorted([
            f for f in os.listdir(self.original_img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL image to torch.Tensor
            transforms.Resize((self.input_h, self.input_w))  # Resize to a standard size if needed
        ])
        for directory in [self.val_img_dir, self.val_lbl_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        self.train_img_files = []
        self.transitions = {}
        self.val_imgs = random.sample(self.original_img_files, self.val_size)
        for i, img in enumerate(self.original_img_files):
            if img in self.val_imgs:
                lbl = os.path.splitext(img)[0] + ".txt"
                shutil.copy2(os.path.join(self.original_img_dir, img), os.path.join(self.val_img_dir, img))
                shutil.copy2(os.path.join(self.original_lbl_dir, lbl), os.path.join(self.val_lbl_dir, lbl))
            else:
                self.train_img_files.append(img)
        for i, img in enumerate(self.train_img_files):
            state = self.rgb(os.path.join(self.original_img_dir, img))
            if i != len(self.train_img_files) - 1:
                next_state = self.rgb(os.path.join(self.original_img_dir, self.train_img_files[i + 1]))
                self.transitions[img] = Transition(state, next_state, False)
            else:
                self.transitions[img] = Transition(state, None, True)

        self.dirs = [
            os.path.join(self.img_dir, '0'),
            os.path.join(self.img_dir, '1'),
            os.path.join(self.img_dir, '2'),
            os.path.join(self.lbl_dir, '0'),
            os.path.join(self.lbl_dir, '1'),
            os.path.join(self.lbl_dir, '2'),
        ]

    def rgb(self, img_path):
        return self.transform(Image.open(img_path).convert('RGB'))

    def rgb_batch(self, base_dir, img_files):
        imgs = []
        for file in img_files:
            img_path = os.path.join(base_dir, file)
            image = Image.open(img_path).convert('RGB')
            imgs.append(self.transform(image))

        return torch.stack(imgs).to('cuda')

    def reset(self):
        self.vision_model = YOLO("yolo11n.pt")
        if os.path.exists("runs"):
            shutil.rmtree("runs")
        for directory in self.dirs:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        for transition in self.transitions.values():
            transition.action = None
            transition.reward = None

        return

    def rollout(self, model, eps, test_mode=False):
        states = self.rgb_batch(self.original_img_dir, self.train_img_files)
        actions = model.select_actions(states, eps, test_mode)  # Assume model returns a list or tensor of actions

        for i, action in enumerate(actions):
            transition = self.transitions[self.train_img_files[i]]
            transition.action = action
            img_src_path = os.path.join(self.original_img_dir, self.train_img_files[i])
            lbl_src_path = os.path.join(self.original_lbl_dir, self.train_img_files[i].replace("png", "txt"))
            level = int(action)
            for n in range(level, self.n_actions):  # Copy to all levels <= assigned level
                img_dst_path = os.path.join(self.img_dir, str(n), self.train_img_files[i])
                lbl_dst_path = os.path.join(self.lbl_dir, str(n), self.train_img_files[i].replace("png", "txt"))
                shutil.copy2(img_src_path, img_dst_path)
                shutil.copy2(lbl_src_path, lbl_dst_path)

        result = self.vision_model.train(data=self.curriculums[0]["data"], epochs=self.curriculums[0]["epochs"], imgsz=self.img_size)
        self.vision_model = YOLO(self.last_path(result))

        result = self.vision_model.train(data=self.curriculums[1]["data"], epochs=self.curriculums[1]["epochs"], imgsz=self.img_size)
        self.vision_model = YOLO(self.last_path(result))

        for i in range(self.curriculums[2]["epochs"]):
            result = self.vision_model.train(data=self.curriculums[2]["data"], epochs=1, imgsz=self.img_size)
            self.vision_model = YOLO(self.last_path(result))

            preds = self.vision_model.predict(source=os.path.join(self.img_dir, '2'), save=False, save_txt=False, stream=True)
            for pred in preds:
                img = os.path.basename(pred.path)

                if self.transitions[img].reward is not None:
                    continue

                pred_boxes = pred.boxes.xyxy.cpu().numpy()
                pred_classes = pred.boxes.cls.cpu().numpy()

                # Load GT
                label_path = os.path.join(os.path.join(self.lbl_dir, '2'), img.replace("png", "txt"))
                gt = np.loadtxt(label_path).reshape(-1, 5)
                gt_classes = gt[:, 0]
                gt_boxes = self.convert_yolo_to_xyxy(gt[:, 1:], img_shape=pred.orig_shape)  # H, W

                if self.match_predictions(pred_boxes, pred_classes, gt_boxes, gt_classes):
                    self.transitions[img].reward = -i

        for transition in self.transitions.values():
            if transition.reward is None:
                transition.reward = -10

        return self.transitions

    def best_path(self, result):
        return os.path.join(str(result.save_dir.resolve()), "weights", "best.pt")

    def last_path(self, result):
        return os.path.join(str(result.save_dir.resolve()), "weights", "last.pt")

    def convert_yolo_to_xyxy(self, yolo_boxes, img_shape):
        """
        Converts YOLO box format (x_center, y_center, width, height) to xyxy (x1, y1, x2, y2).
        """
        h, w = img_shape
        x_c, y_c, bw, bh = yolo_boxes[:, 0], yolo_boxes[:, 1], yolo_boxes[:, 2], yolo_boxes[:, 3]
        x1 = (x_c - bw / 2) * w
        y1 = (y_c - bh / 2) * h
        x2 = (x_c + bw / 2) * w
        y2 = (y_c + bh / 2) * h
        return np.stack([x1, y1, x2, y2], axis=1)

    def match_predictions(self, pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
        """
        Matches predicted boxes to ground-truth boxes using IoU and class matching.
        Returns number of correct predictions.
        """
        correct = 0
        matched_gt = set()
        for pb, pc in zip(pred_boxes, pred_classes):
            for i, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                if i in matched_gt:
                    continue
                iou = bbox_iou(torch.tensor(pb).unsqueeze(0), torch.tensor(gb).unsqueeze(0), xywh=False)[0]
                if iou >= iou_threshold and pc == gc:
                    correct += 1
                    matched_gt.add(i)
                    break
        return correct == len(gt_classes)
