import os
import json
import shutil
import threading
import numpy as np
from glob import glob
from pathlib import Path
from  matplotlib.path import  Path as  mplPath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from tkinter import filedialog
from ultralytics import YOLO
from time import time
# plt.rcParams['keymap.save'].remove('s')
# plt.rcParams['keymap.save'].remove('ctrl+s')

# plt.rcParams['keymap.quit'].remove('q')
plt.rcParams['keymap.quit'].remove('ctrl+w')

# plt.rcParams['keymap.pan'].remove('p')
# plt.rcParams['keymap.zoom'].remove('o')
plt.rcParams['keymap.fullscreen'].remove('f')
plt.rcParams['keymap.xscale'].remove('k')
plt.rcParams['keymap.yscale'].remove('l')
plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.grid_minor'].remove('G')

from custom_functions.yolo_data import  load_yolo_labels, draw_polygons
import cv2


class YoloCaller:
    def __init__(self,
                 model_path:str,
                 ):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.polygons = []
        self.boxes = []

    
    def detect_single(self, img, imgsz=640, conf=0.5, iou=0.7, device="cpu"):
        # Detect objects
        preds = self.model.predict(img, imgsz=imgsz, conf=conf, iou=iou)[0].to(device=device)
        polygons = [ (seg*np.array(preds.orig_shape[::-1])).astype(int) for seg in preds.masks.xyn]
        
            
        return preds.boxes, polygons

    def convert_result(self, yolo_result):
        for res in yolo_result:
            pass
                    
            
class App:
    def __init__(self,
                 image_paths:list,
                 label_paths:list,
                 table_detector:str,
                 word_detector:str
                 ):
        
        
        self.image_paths = image_paths
        self.label_paths = label_paths
        print("Loading the models")
        st = time()
        self.table_detector = YoloCaller(table_detector)
        self.word_detector = YoloCaller(word_detector)
        print(f"Models loaded in {(time()-st)*1000:.2f} ms")

        self.table_polygons = []
        self.word_polygons = []
        self.selected_polygon = None
        self.prev_selected_polygon = None
        self.current_img_index = 0
        
        # The labels variable is for storing the labels give n by user
        # It is a list of `Labels`
        self.labels = {}

        
        # Connect the key press and mouse motion event handlers
        self.fig, self.ax = plt.subplots()
        
        # Bind the key press event handler and mouse move
        # self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.update_image()
    
    # def set_poly_mask(self, image, polygon, value):
    #     path = mplPath(polygon)
    #     x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    #     points = np.column_stack((x.ravel(), y.ravel()))
    #     mask = path.contains_points(points).reshape(image.shape[0], image.shape[1])
    #     image[mask] = value
    #     print(value)
    #     return image
    def set_poly_mask(self, image, polygon, value):
        cv2.fillPoly(image, np.int32([polygon]), value)
        return image
    
        
    def update_image(self):
        # Clear the cache
        # plt.cla()
        self.ax.clear()
        self.ax.patches.clear()
        
        img = mpimg.imread(self.image_paths[self.current_img_index])
        # raw_polygons =  load_yolo_labels(self.label_paths[self.current_img_index], img.shape)
        raw_polygons =  self.word_detector.detect_single(img, imgsz=640, conf=0.2, iou=0.7, device="cpu")[1]
        
        self.ax.imshow(img)
        self.ax.text(0.05, 1.08, f'Image: {self.current_img_index+1}/{len(self.image_paths)}',
                     transform=self.ax.transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.5),
                     fontdict={'fontsize': 15})
        
        self.selection_label  = self.ax.text(0.95, 0.45, f'Label',
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5),
                        fontdict={'fontsize': 40})
        # Exit if the image has no labels
        if len(raw_polygons)==0:
            return
        

        if self.current_img_index not in self.labels:
            self.labels[self.current_img_index] = []
        
        self.polygons_cache = []
        self.polygons = []
        self.poly_image_mask = np.zeros(img.shape[:2], dtype=int)
        for i, poly in enumerate(raw_polygons):
            # Here need to have duplicate of polygons for correctly checking if the mouse is hovered
            self.polygons_cache.append(Polygon(poly, closed=True))
            self.polygons.append(Polygon(poly, closed=True, facecolor=((1,0,0,0.5)), edgecolor='white'))
            self.poly_image_mask = self.set_poly_mask(self.poly_image_mask, poly, i+1)
            self.ax.add_patch(self.polygons[-1])
        self.fig.canvas.draw()



    # # Function to handle mouse motion events
    def on_mouse_move(self, event):
        # Check if mouse is over any polygon
        if (event.xdata is not None) and (event.ydata is not None):
            if self.prev_selected_polygon!=self.selected_polygon:
                self.prev_selected_polygon = self.selected_polygon
            
            
            self.selected_polygon = self.poly_image_mask[int(event.ydata), int(event.xdata)]

            if self.prev_selected_polygon:
                self.ax.patches[self.prev_selected_polygon-1].set_facecolor((1,0,0,0.5))

            if self.selected_polygon:
                # print(self.selected_polygon, self.prev_selected_polygon)
                self.ax.patches[self.selected_polygon-1].set_facecolor((0,1,0,0.5))
                self.selection_label.set_text(f'{self.selected_polygon}')

            self.fig.canvas.draw()
            
            # if self.selected_polygon==0:
            #     polygon.set_facecolor((0,1,0,0.5))
            # else:
            #     polygon = self.ax.patches[]
            #     polygon.set_facecolor((1,0,0,0.5))

            # self.fig.canvas.draw()
    # # # Function to handle mouse motion events
    # def on_mouse_move(self, event):
    #     # Check if mouse is over any polygon
    #     if (event.xdata is not None) and (event.ydata is not None):
    #         for i, polygon in enumerate(self.ax.patches):
    #             if self.polygons_cache[i].contains_point((event.xdata, event.ydata)):
    #                 self.selected_polygon = i
    #                 polygon.set_facecolor((0,1,0,0.5))
    #                 # self.selection_label.set_text(f'{self.labels[self.current_img_index][i].label}')
    #             else:
    #                 polygon.set_facecolor((1,0,0,0.5))
                    
    #         self.fig.canvas.draw()


if __name__=="__main__":
    root = filedialog.askdirectory()
    root = Path(root)
    image_paths = glob(os.path.join(root, "images/*"))
    label_paths = [str(i).replace("images", "labels").replace("jpeg", "txt") for i in image_paths]
    
    app = App(image_paths, label_paths, table_detector="./weights/Table_det.pt", word_detector="./weights/Word_det.pt")
    plt.show()