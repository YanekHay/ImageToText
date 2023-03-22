import cv2
import torch
import torchvision
import numpy as np

from .utils import get_n_percent_val, fit_curve
from .plotter import draw_curve_on_image
from .Line import Line

def extend_lines(feature, orientation, dilation_count=1, erode_count=1, line_min_width=50, dilation_line_min_width=150):
    if orientation.lower() not in ["vertical","horizontal"]:
        raise ValueError('`orientation` should be one of the following  ["vertical","horizontal"]')
        
    kernel_erosion = np.ones((1,line_min_width), np.uint8)
    kernel_dilation = np.ones((1,dilation_line_min_width), np.uint8)
    
    if orientation.lower()=="vertical":
        kernel_erosion = kernel_erosion.T
        kernel_dilation = kernel_dilation.T
    
    ## Erode for removing extra small lines
    img_bin = cv2.erode(feature.numpy().astype(np.uint8), kernel_erosion, erode_count) 
    ## Dilate for getting better expressed lines
    img_bin = cv2.dilate(img_bin, kernel_dilation, dilation_count)
    
    return img_bin


def get_line_indices(group_areas, diff_threshold=3000):
    '''
    A function for getting the line indices of pixels that represent a line
    '''
    sort_mask = np.argsort(group_areas)
    # Finding the places where th group area is less than threshold
    break_indices = np.where(np.diff(group_areas[sort_mask])<diff_threshold)[0]
    if break_indices.shape[0]>0:
        return sort_mask[:break_indices[-1]], sort_mask[break_indices[-1]:]
    else:
        return []
    
    
def filter_out_small_lines(dilated_binary_image, difference_threshold):
    '''
    A function to zero down the pixels that are in one group(an island) that have small area
    Returns:
        - The cleaned image with big lines on it and the indices of the groups with big enough area
    '''
    num_groups, grouped = cv2.connectedComponents(dilated_binary_image, connectivity=8)
    group_areas = np.array([((grouped==i).sum()) for i in range(num_groups)])
    small_lines, big_lines = get_line_indices(group_areas, difference_threshold)
    for i in small_lines:
        grouped[grouped==i] = 0
    
    return grouped, big_lines


def get_clean_lines(dilated_binary_image:np.ndarray, difference_threshold:float, vertical:bool):
    '''
    A function for cleaning up the given image from small noisy lines and draw 
    approximate lines according to the remaining ones
    
    
    Returns 1. the grouped image of connected pixels, 
            2. average lines Line(pt1, pt2) for connected pixels,
            
    Use `draw_hv_lines` function from plotter to draw the returned lines on any image
    '''
    lines = []
    
    clean_grouped, big_lines = filter_out_small_lines(dilated_binary_image, difference_threshold) # Get cleaner image with remainig line indices on it
    for i in big_lines:
        if i==0:
            continue
        group_indices = np.where(clean_grouped==i)
        # Get **[y]** coordinates if the vertical lines are required, **[x]** coordinates otherwise
        if vertical:
            unique_xes = np.unique(group_indices[1])
            x1 = int(get_n_percent_val(unique_xes, 0.2))
            x2 = int(get_n_percent_val(unique_xes, 1))
            y1 = int(np.mean(group_indices[0][group_indices[1]==x1])) # Get the mean of y coords where x1 is selected
            y2 = int(np.mean(group_indices[0][group_indices[1]==x2])) # Get the mean of y coords where x2 is selected
        else:
            unique_yes = np.unique(group_indices[0])
            y1 = int(get_n_percent_val(unique_yes, 0.2))
            y2 = int(get_n_percent_val(unique_yes, 1))
            x1 = int(np.mean(group_indices[1][group_indices[0]==y1])) # Get the mean of x coords where y1 is selected
            x2 = int(np.mean(group_indices[1][group_indices[0]==y2])) # Get the mean of x coords where y2 is selected
        
        if x1==x2 and y1==y2:
            continue
            
        _line = Line((x1, y1),(x2,y2))
        if _line.length > 100:
            lines.append(_line)

        
        
    return clean_grouped, lines


def get_clean_curves(dilated_binary_image:np.ndarray, difference_threshold:float, vertical:bool, resolution=5):
    '''
    Get a cleaner image of the given dilated binary image. Clean small lines, get approximate points of curves for the remaining big lines on the image
    
    Returns:
        1. Cleaned image with only big lines on it
        2. Points of the curves passing through those lines
        3. A clean image containing only the curves on it
    '''
    num_groups, grouped = cv2.connectedComponents(dilated_binary_image, connectivity=8)
    group_areas = np.array([((grouped==i).sum()) for i in range(num_groups)])
    
    curve_step = 1/resolution
    
    lines = []
    
    clean_grouped, big_lines = filter_out_small_lines(dilated_binary_image, difference_threshold)
    img_with_curves = np.zeros_like(clean_grouped)
    
    # If vertical lines are needed, for correct curve extraction need to Transpose the image, draw the curves on it and then rotate back
    if vertical:
        img_with_curves = img_with_curves.T
    
    for i in big_lines:
        lines.append([])
        
        if i==0:
            continue
            
        group_indices = np.where(clean_grouped==i)
#         print(group_indices)
        x_list = []
        y_list = []
        # Get **[y]** coordinates if the vertical lines are required, **[x]** coordinates otherwise
        percentages = np.linspace(0,1,resolution)
        
        
        uniques = np.unique(group_indices[vertical])
        
        for j in range(resolution+1):
            y_list.append(int(get_n_percent_val(uniques, j*curve_step))) # Get coordinates of lines for constructing a curve
            x_list.append(int(np.mean(group_indices[not vertical][group_indices[vertical]==y_list[-1]])))
        
        # Appending starting and ending points to make the curves finished
        y_list.append(y_list[0])
        y_list.append(y_list[-2])

        x_list.append(0)
        x_list.append(dilated_binary_image.shape[not vertical])
        
        # Stack the x,y points on the curves
        points = np.stack((x_list, y_list))
        points = np.unique(points, axis=1)
        

        if points.shape[1] != (resolution+3):
            continue
        
        # Check if the distance between the start and the end of the curve is enough
        
        _line = Line(tuple(points[:,1]), tuple(points[:,-2]))
        if _line.length < 100:
            continue
            
        # Get the curve
        points = fit_curve(points)
        img_with_curves = (draw_curve_on_image(img_with_curves, points))
        lines[-1].append(points)
        
    if vertical:
        img_with_curves=img_with_curves.T
    
    return clean_grouped, np.array(lines), img_with_curves