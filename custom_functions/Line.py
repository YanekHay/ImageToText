import numpy as np
from math import ceil

class Line:
    def __init__(self, pt1, pt2):
        if not isinstance(pt1, (tuple,list,np.ndarray)) or \
           not isinstance(pt2, (tuple,list,np.ndarray)) or \
        len(pt1)!=2 or len(pt2)!=2 or not isinstance(pt2, type(pt1)):
            
            raise ValueError("pt1 and pt2 should be arraylike of the same class containing exactly 2 integers")
        
        if pt1==pt2:
            raise ValueError("The two points of a line should vary from each other")
            
        self.pt1 = pt1
        self.pt2 = pt2
        
        self.x1, self.y1 = pt1
        self.x2, self.y2 = pt2
    
    @property
    def y_intercept(self):
        return self.y1 - self.slope * self.x1
    
    @property
    def slope(self):
        return self.dy/self.dx
    
    @property
    def dx(self):
        return self.x2 - self.x1
    
    @property
    def dy(self):
        return self.y2 - self.y1
    
    @property
    def angle(self):
        '''
        The angle of the line from horizontal axis (x axis).
        '''
        return np.degrees(np.arctan2(self.dy, self.dx))
    
    @property
    def length(self):
        '''
        Get the length of the line.
        '''
        return np.sqrt(self.dx**2 + self.dy**2)
    
    @property
    def only_coords(self):
        return (self.pt1,self.pt2)
        
    def get_equation(self, vertical:bool=False):
        '''
        return the equation of a line(pt1,pt2)
        if vertical is true returns equation for computing y for given x and the opposite otherwise
        '''
        if vertical:
            return lambda x: int(self.slope * x + self.y_intercept)
        else:
            return lambda y: int((y - self.y_intercept)/self.slope)
        
            
    def __str__(self):
        return f"Line:\n\t{self.pt1} -> {self.pt2}\n\tlength =  {self.length:.2f} px\n\tangle  =  {self.angle:.2f} °\n"