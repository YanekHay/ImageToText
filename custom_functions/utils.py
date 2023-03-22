import matplotlib.pyplot as plt
import numpy as np
import torch

def conv1d(array, kernel=None, kernel_size=5, kernel_fill="gauss", seed=None, conv_mode='valid', pad_mode="mean"):
    '''
    A function for passing an array through 1D convolution\n
    Parameters:
        - `array`: A 1D array `list` or `numpy.ndarray`
        - `kernel`: If specified, the kernel of convolution will be set to this, `kernel_size` parameter will be ignored
        - `kernel_size`: If the kernel is not specified, generates the kernel with this size
        - `kernel_fill`: The way of filling the kernel
        Accepts one of the following values:
            - `gauss`: Fills with exp(-1/max(1-(2*i/kernel_size)**2,1e-1) function
            - `random`: Fills with random values
            - `mean`: Fills with `(1/element_count)`
            - `any number` of type `int | float`: Fills with given constant value
            - `all the methods available for 
        - `seed`: If kernel_fill is set to `random`, a random seed can be given.
        - `conv_mode`: The mode of convolution (look numpy.convolve)
        - `pad_mode`: The way of padding the array for getting the expected results | see `numpy.pad modes`
    '''
    if kernel is None:
        if seed is not None:
            np.random.seed(seed)
        if kernel_fill.lower() == "random":
            kernel = np.random.rand(kernel_size,)
        elif kernel_fill.lower() == "mean":
            kernel = [1/kernel_size]*kernel_size
        elif kernel_fill.lower() == "gauss":
            kernel = [np.exp(-1/max(1-(2*i/kernel_size)**2,1e-1)) for i in range(-kernel_size//2,kernel_size//2+1 )]
            S = sum(kernel)
            kernel = [i/S for i in kernel]
        elif isinstance(kernel_fill, (int,float)):
            kernel = [kernel_fill]*kernel_size

                        
    elif not isinstance(kernel, (list,np.ndarray)):
        raise TypeError(f"Expected 'kernel' to be an instance of (list | np.ndarray), but got {type(kernel)}")
    elif isinstance(kernel[0],(list,tuple,np.ndarray)):
        raise TypeError(f"The 'kernel should be a 1D arraylike'")
    else:
        kernel_size = len(kernel)
        
    # Pad with given method
    array = np.pad(array, kernel_size//2, mode=pad_mode)
    
    # Pass through the convolution
    return np.convolve(array, kernel, mode=conv_mode)

def change_range(x,old_xmin,old_xmax,new_xmin,new_xmax):
    '''
    A function for mapping a value from one range to another,
    e.g. -> `150` from range `[0,255]` to range `[0,1]` will be \n
    `change_range(150, 0, 255, 0, 1)`
    '''
    return new_xmin + ((new_xmax - new_xmin) / (old_xmax - old_xmin)) * (x - old_xmin)


def get_n_percent_val(inp, n=0.9):
    '''
    Sort the input and get the n% th value of the array,matrix or tensor
    '''
    if isinstance(inp, np.ndarray):
        inp = torch.from_numpy(inp)
    if n>1 or n<0:
        print("`n` should be a number in [0,1] range")
    values, indices = torch.sort(inp.flatten())
    return values[int(n*(len(values)-1))]


def fit_curve(points, degree=3): # degree of polynomial
    '''
    get smoother points of a curve based on the given points
    '''
    # points is a list of tuples containing coordinates of points (x,y)
    x = points[0]
    y = points[1]
    
    # fit curve
    coefficients = np.polyfit(x,y,degree)
    
    # create polynomial function
    poly = np.poly1d(coefficients)
    
    # plot data
#     plt.scatter(x,y)
    
    # plot curve
    x_new = np.linspace(min(x),max(x), 10)
    y_new = poly(x_new)
    
    return np.array((x_new,y_new)).T
