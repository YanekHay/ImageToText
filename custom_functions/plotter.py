import torchvision
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from time import time
import cv2

from sklearn.cluster import AgglomerativeClustering as AGC


def plot_features(feature_list:torch.Tensor, feature, feature_id, save=False):
    '''
    A function for plotting given list of features as plt.subplots
    '''
    print(feature_list.shape)
    col_count = int(feature_list.shape[0]**0.5)
    row_count = ceil(feature_list.shape[0] / col_count)
    
    fig, axes = plt.subplots(row_count, col_count, figsize=(50, 50))
    plt.xticks
    plt.suptitle(str(feature))
    
    ind = 0
    for i in range(row_count):
        for j in range(col_count):
            if ind >= feature_list.shape[0]:
                break
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            
            axes[i,j].set_title(i*col_count+j)
            axes[i,j].imshow(feature_list[i*col_count+j])
            ind += 1
    plt.tight_layout()
    if save:
        plt.savefig(f"./Features/{feature_id}_{str(feature).split('(')[0]}.png")
    else:
        plt.show()
        
        
def plot_islands(*labeled_imgs):
    '''
    A function for finding the separated lines in the binary image and plotting each group in different color
    '''
    for labeled_img in labeled_imgs:
        # Applying cv2.connectedComponents() 
        _labeled_img = labeled_img.copy()
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*_labeled_img/np.max(_labeled_img))
        blank_ch = 255*np.ones_like(label_hue)
        _labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        _labeled_img = cv2.cvtColor(_labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        _labeled_img[label_hue==0] = 0

        #Showing Image after Component Labeling
        plt.imshow(cv2.cvtColor(_labeled_img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
        plt.title("Image after Component Labeling")
        plt.show()
    
    return _labeled_img


def draw_hv_lines(image, lines , vertical:bool, color=(255,255,255), thickness=10):
    '''
    Draw either vertical or horizontal lines on the given image
    '''
    img = image.copy()
    for line in lines:
        lineq_ = line.get_equation(vertical)
        
        pt1 = (lineq_(0), 0)
        pt2 = (lineq_(image.shape[0]), image.shape[0])
        
        if  vertical:
            pt1 = pt1[::-1]
            pt2 = (image.shape[1], lineq_(image.shape[1]))
        img = cv2.line(img, pt1, pt2, color=color, thickness=thickness)
            
    return img


def draw_curve_on_image(image, points, thickness=10, color=(255,0,0)):
    '''
    Draw a curve on the copy of given image
    '''
    # points is a list of tuples containing coordinates of points (x,y)
    # read image
    img = image.copy()
    # convert points to numpy array
    pts = np.array(points, np.int32)
    
    # reshape points
    pts = pts.reshape((-1,1,2))

    # draw curve on image
    img = cv2.polylines(img, [pts], False, color, thickness=thickness)
    
    return img
