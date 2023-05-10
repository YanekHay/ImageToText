import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from custom_functions.yolo_data import rescale


root = Path("../TRAIN_DATA/Letters/train")

image_paths = glob(os.path.join(root, "images/*"))
label_paths = [str(i).replace("images", "labels").replace("jpeg", "txt") for i in image_paths]


def load_yolo_labels(label_path, img_shape):
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        lines[i] = np.array([float(j) for j in lines[i].replace("\n","").split()[1:]]).reshape((-1,2))
        lines[i][:, 0] *= img_shape[1]
        lines[i][:, 1] *= img_shape[0]
        lines[i] = lines[i].astype(np.int32)
    return lines


def draw_polygon(img, polygon, fill_color=(200, 0, 0), outline_color=(200, 200, 200)):
    overlay = np.zeros_like(img)
    outlines = np.zeros_like(img)
    cv2.fillPoly(overlay, [polygon], fill_color)
    cv2.polylines(outlines, [polygon], True, outline_color, 1)

    img = cv2.addWeighted(img, 1, overlay, 0.5, 0)
    img = cv2.addWeighted(img, 1, outlines, 1, 0)
        
    return img


for i in range(len(image_paths)):
    img = plt.imread(image_paths[i])
    polygons = load_yolo_labels(label_paths[i], img.shape)
    for i in range(len(polygons)):
        img_c = draw_polygon(img, polygons[i], fill_color=(0,200,0))
        for j,polygon in enumerate(polygons):
            if j==i:
                continue
            img_c = draw_polygon(img_c, polygon)
        plt.imshow(img_c)
        plt.show()

