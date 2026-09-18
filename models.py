"""
This defines the models used for the different tests

They are all functions taking as inputs two numpy arrays x, y and outputs scalar functions

the functions are given the following attributes that allow nice usage 
- x_min, x_max, y_min, y_max : defines a suitable window for analysis 

They are very simply wrapped in the Model class that is a callable in addition to having those fields 
"""
import numpy as np 

class Model:
    def __init__(self, func, bounds=None, name=None):

        ## default parameters 
        if bounds is None:
            bounds = [(-1, 1), (-1, 1)]

        self._func = func
        (self.x_min, self.x_max), (self.y_min, self.y_max) = bounds
        self.name = name 

    def __call__(self, x, y):
        return self._func(x, y)

def _branin(x, y):
    a = 1 
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi 
    r = 6 
    s = 10 
    t = 1 / (8 * np.pi)
    return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s 

branin = Model(func=_branin, bounds=[(-5, 10), (0, 15)], name="2D Branin Function")

def _goldstein(x, y):
    return (
        1 + (x + y + 1)**2 *
            (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    ) * (
        30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    )

goldstein = Model(func=_goldstein, bounds=[(-2, 2), (-2, 2)], name="Goldstein price function")

def _rosenbrock(x, y):
    a, b = 1, 100 
    return (a - x) ** 2 + b * (y - x**2) ** 2

rosenbrock = Model(func=_rosenbrock, bounds=[(-2, 2), (-1, 3)], name="2D Rosenbrock function")

def _custom_sine(x, y):
    return np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + 0.3*np.sin(6*np.pi*x + 2*y)

custom_sine = Model(func=_custom_sine, bounds=[(-1, 1), (-1, 1)], name="Custom wave surface")
