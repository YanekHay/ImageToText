import os
import json
import shutil
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from tkinter import filedialog

plt.rcParams['keymap.save'].remove('s')
plt.rcParams['keymap.save'].remove('ctrl+s')

plt.rcParams['keymap.quit'].remove('q')
plt.rcParams['keymap.quit'].remove('ctrl+w')

plt.rcParams['keymap.pan'].remove('p')
plt.rcParams['keymap.zoom'].remove('o')
plt.rcParams['keymap.fullscreen'].remove('f')
plt.rcParams['keymap.xscale'].remove('k')
plt.rcParams['keymap.yscale'].remove('l')
plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.grid_minor'].remove('G')

import sys
sys.path.append("../")
from custom_functions.yolo_data import  load_yolo_labels


all_classes = json.load(open("all_classes.json"))

class Label:
    def __init__(self, img_id, id, label, img_path, polygon):
        self.img_id = img_id
        self.id = id
        self.label = label
        self.img_path = Path(img_path)
        self.polygon = polygon # [[x1,y1],[x2,y2],...,[xn,yn]] | Shape = (N x 2)
    
    @property
    def _class(self):
       all_classes[self.label]
       
    def copy_image(self, location):
        shutil.copy2(str(self.img_path), location)
    
    def __str__(self) -> str:
        res = f"{self._class}"
        flat_poly = self.normalize(self.polygon).flatten()
        for i in range(len(flat_poly)):
            res += f" {flat_poly[i]}"
        return res
    
    def normalize(self, polygon):
        img_shape = mpimg.imread(self.img_path).shape
        polygon = polygon / np.array(img_shape[:2])
          
        return polygon
        
        
class LetterLabeler:
    def __init__(self,
                 image_paths:list,
                 label_paths:list,
                 ):
        
        
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.polygons = []
        self.polygons_cache = []
        self.selected_polygon = None
        self.current_img_index = 0
        
        # The labels variable is for storing the labels give n by user
        # It is a list of `Labels`
        self.labels = {}

        
        # Connect the key press and mouse motion event handlers
        self.fig, self.ax = plt.subplots()
        
        # Bind the key press event handler and mouse move
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.update_image()
    
    def update_image(self):
        # Clear the cache
        # plt.cla()
        self.ax.clear()
        self.ax.patches.clear()
        
        img = mpimg.imread(self.image_paths[self.current_img_index])
        raw_polygons =  load_yolo_labels(self.label_paths[self.current_img_index], img.shape)
        self.ax.imshow(img)
        self.ax.text(0.05, 1.08, f'Image: {self.current_img_index+1}/{len(self.image_paths)}',
                     transform=self.ax.transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.5),
                     fontdict={'fontsize': 15})
        self.selection_label  = self.ax.text(0.95, 0.45, f'?',
                        transform=self.ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5),
                        fontdict={'fontsize': 40})
        # Exit if the image has no labels
        if len(raw_polygons)==0:
            return
        

        self.polygons = []
        self.polygons_cache = []
        
        if self.current_img_index not in self.labels:
            self.labels[self.current_img_index] = []
            
        for i, poly in enumerate(raw_polygons):
            # Here need to have duplicate of polygons for correctly checking if the mouse is hovered
            self.polygons_cache.append(Polygon(poly, closed=True))
            self.polygons.append(Polygon(poly, closed=True, facecolor=((1,0,0,0.5)), edgecolor='white'))
            
            if len(self.labels[self.current_img_index])!=len(raw_polygons):
                # Add label if it does not already exist
                label = Label(self.current_img_index, i, '?', self.image_paths[self.current_img_index], poly)
                self.labels[self.current_img_index].append(label)
                
            self.ax.add_patch(self.polygons[-1])
        self.fig.canvas.draw()
        
    # Function to handle key press events
    def on_key(self, event):
        # Show next image and polygons
        if event.key == 'ctrl+a':
            self.current_img_index -= 1
            if self.current_img_index < 0:
                self.current_img_index = len(self.image_paths)-1
                
            self.update_image()

        # Show previous image and polygon
        elif event.key == 'ctrl+d':
            self.current_img_index += 1
            if  self.current_img_index == len(self.image_paths):
                self.current_img_index = 0
                
            self.update_image()
        elif event.key == 'ctrl+s':
            self.save_labels("./TEST")
            
        elif len(event.key)==1:
            # Add label to the selected polygon
            if self.selected_polygon is not None:
                self.labels[self.current_img_index][self.selected_polygon].label = event.key
                self.labels[self.current_img_index][self.selected_polygon].polygon = self.polygons[self.selected_polygon].get_xy()
                self.selection_label.set_text(event.key)
                self.ax.patches[self.selected_polygon].set_facecolor((1,1,0,0.5))
                self.fig.canvas.draw()
                
    # Function to handle mouse motion events
    def on_mouse_move(self, event):
        # Check if mouse is over any polygon
        if (event.xdata is not None) and (event.ydata is not None):
            for i, polygon in enumerate(self.ax.patches):
                if self.polygons_cache[i].contains_point((event.xdata, event.ydata)):
                    self.selected_polygon = i
                    polygon.set_facecolor((0,1,0,0.5))
                    self.selection_label.set_text(f'{self.labels[self.current_img_index][i].label}')
                else:
                    polygon.set_facecolor((1,0,0,0.5))
                    
            self.fig.canvas.draw()

    def save_labels(self, location):
        # Save the labels
        os.makedirs(os.path.join(location, "images"), exist_ok=True)
        os.makedirs(os.path.join(location, "labels"), exist_ok=True)
        q = 0
        for i, label in enumerate(self.labels.values()):
            shutil.copy2(str(self.image_paths[i]), os.path.join(location, "images"))
            if len(label) == 0:
                continue
            with open(os.path.join(location, f"labels/{label[0].img_path.stem}.txt"), "w") as f:
                for poly in label:
                    f.write(str(poly))
                    f.write("\n")
                q+=1
        
        print(f"Saved {q} Labels and Images")

if __name__=="__main__":
    root = filedialog.askdirectory()
    root = Path(root)
    image_paths = glob(os.path.join(root, "images/*"))
    label_paths = [str(i).replace("images", "labels").replace("jpeg", "txt") for i in image_paths]
    
    labeler = LetterLabeler(image_paths, label_paths)
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Exiting...")
        labeler.save_labels("./TEST")
        