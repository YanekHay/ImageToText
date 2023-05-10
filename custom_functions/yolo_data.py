import numpy as np
import cv2
import os   

def load_yolo_labels(label_path, img_shape):
    """
    Load YOLO labels from a file and convert them to pixel coordinates.

    Inputs:
        - `label_path`: Path to the label file | type: str
        - `img_shape`: Shape of the image (height, width) | type: tuple
    Output(s):
        - List of labels in pixel coordinates | type: list
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        # Convert string to numpy array and reshape
        lines[i] = np.array([float(j) for j in lines[i].replace("\n","").split()[1:]]).reshape((-1,2))
        # Convert normalized coordinates to pixel coordinates
        lines[i][:, 0] *= img_shape[1]
        lines[i][:, 1] *= img_shape[0]
        # Convert to integer
        lines[i] = lines[i].astype(np.int32)
    return lines

def draw_polygons(img, polygons, fill_color=(200, 0, 0), outline_color=(200, 200, 200)):
    """
    Draw a polygon on an image.

    Inputs:
        - `img`: Image to draw the polygon on | type: numpy.ndarray
        - `polygons`: Coordinates of the polygon vertices | type: numpy.ndarray
        - `fill_color`: Color to fill the polygon with (BGR) | type: tuple
        - `outline_color`: Color of the polygon outline (BGR) | type: tuple
    Output(s):
        - Image with the polygon drawn on it | type: numpy.ndarray
    """
    # Create overlay and outlines images
    overlay = np.zeros_like(img)
    outlines = np.zeros_like(img)
    if not isinstance(polygons, list):
        polygons = [polygons]
    # Draw filled polygon on overlay image
    cv2.fillPoly(overlay, polygons, fill_color)
    
    # Draw polygon outline on outlines image
    cv2.polylines(outlines, polygons, True, outline_color, 1)

    # Add overlay and outlines to the original image
    img = cv2.addWeighted(img, 1, overlay, 0.5, 0)
    img = cv2.addWeighted(img, 1, outlines, 1, 0)
        
    return img
