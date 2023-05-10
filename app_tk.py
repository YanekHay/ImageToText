import os
import cv2
import torch
import numpy as np
from glob import glob
from time import time
from tkinter import *
from pathlib import Path
from ultralytics import YOLO

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog


class GLOBAL:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YoloCaller:
    def __init__(self,
                 model_path:str,
                 device:torch.device
                 ):
        self.device = device
        
        self.model_path = model_path
        self.model = YOLO(self.model_path)

        self.polygons = []
        self.boxes = []

    
    def detect_single(self, img, imgsz=640, conf=0.5, iou=0.7, return_device="cpu"):
        # Detect objects
        preds = self.model.predict(img, imgsz=imgsz, conf=conf, iou=iou)[0].to(device=return_device)

        return preds.boxes, preds.masks.xyn

    def convert_result(self, yolo_result):
        for res in yolo_result:
            pass
                    
class App(tk.Tk):
    def __init__(self,
                 image_paths:list,
                 device:torch.device
                 ):
        self.current_image_index = 0
        self.image_paths = image_paths
        st = time()
        self.table_detector = YoloCaller("./weights/Table_det.pt", device=device)
        self.word_detector = YoloCaller("./weights/Word_det.pt", device=device)
        self.char_detector = YoloCaller("./weights/Char_det.pt", device=device)
        self.word_polygons = []
        self.table_polygons = []
        
        print(f"Yolo init time: {time()-st}")
        super().__init__()
        self.__init_window()
        self.__bind_keys()

    def on_key_press(self, event):
        pass
    
    def __bind_keys(self):
        self.bind_all("<Key>", self.on_key_press)
        self.bind('<Control-d>', self.next)
        self.bind('<Control-a>', self.prev) 
         
    
    def next(self, event):
        self.current_image_index += 1
        if self.current_image_index == len(self.image_paths):
            self.current_image_index = 0
            
        self.update_image()
        
    def prev(self, event):
        self.current_image_index -= 1
        if self.current_image_index == -1:
            self.current_image_index = len(self.image_paths) - 1
            
        self.update_image()
        
    def _delete_polygons(self):
        for poly in self.word_polygons:
            self.canvas.delete(poly)
        for poly in self.table_polygons:
            self.canvas.delete(poly)   
            
        self.word_polygons.clear()

            
                     
    def update_image(self):
        self._delete_polygons()
        
        self.img = Image.open(self.image_paths[self.current_image_index])
        img_width, img_height = self.img.size
        scale = min(self.width / img_width, self.height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        self.resized_image = self.img.resize((new_width, new_height))
        
        self.canvas_image = ImageTk.PhotoImage(self.resized_image,  master=self.canvas)
        self.canvas.pack(fill="both", expand=True)
        self._canvas_img_id = self.canvas.create_image(self.width/2, self.height/2, image=self.canvas_image, anchor="center")
        # Get the bounding box of the image
        self._canvas_img_box = self.canvas.bbox(self._canvas_img_id)

        
        table_boxes, _table_polygons = self.table_detector.detect_single(self.img, conf=0.5, iou=0.7)
        self.create_polygons(_table_polygons, self.table_polygons)
        
        word_boxes, _word_polygons  = self.word_detector.detect_single(self.img, imgsz=640, conf=0.2, iou=0.3)
        self.create_polygons(_word_polygons, self.word_polygons)
        
    def create_polygons(self, polygons, poly_list):
        self.table_polygons.clear()
        for polygon in polygons:
            polygon *= np.array(self.resized_image.size) # Rescale the polygon to the image size
            polygon += np.array(self._canvas_img_box[:2]).astype(int) # Move the polygon to the center of the canvas
            
            tk_poly = self.canvas.create_polygon(polygon.tolist(), outline='', fill='', width=3)
            
            poly_list.append(tk_poly)
            self.canvas.tag_bind(tk_poly, "<Enter>", lambda event, polygon=tk_poly: self.on_enter(event, polygon))
            self.canvas.tag_bind(tk_poly, "<Leave>", lambda event, polygon=tk_poly: self.on_leave(event, polygon))


    def __init_window(self):
        self.state("zoomed")
        self.width = int(self.winfo_screenwidth()*0.9)
        self.height = int(self.winfo_screenheight()*0.9)
        self.geometry(f"{self.width}x{self.height}")
        self.canvas = tk.Canvas(self)
        self.update_image()

    def on_enter(self, event, polygon):
        self.canvas.itemconfig(polygon, outline="#000000")

    def on_leave(self, event, polygon):
        self.canvas.itemconfig(polygon, outline="")

if __name__=="__main__":
    root = filedialog.askdirectory()
    root = Path(root)
    image_paths = glob(os.path.join(root, "images/*"))
    app = App(image_paths=image_paths, device=GLOBAL.device)
    app.mainloop()
    