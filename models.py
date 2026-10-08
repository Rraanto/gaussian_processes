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

# https://www.sfu.ca/~ssurjano/ackley.html
def _ackley(x, y):
    a, b, c = 20, 0.2, 2*np.pi
    return (
        -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
        - np.exp((np.cos(c*x) + np.cos(c*y)) / 2) + a + np.e
    )

ackley = Model(func=_ackley, bounds=[(-32.768, 32.768), (-32.768, 32.768)], name="2D Ackley function")

# https://www.sfu.ca/~ssurjano/rastr.html
def _rastrigin(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

rastrigin = Model(func=_rastrigin, bounds=[(-5.12, 5.12), (-5.12, 5.12)], name="2D Rastrigin function")

# https://www.sfu.ca/~ssurjano/beale.html
def _beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

beale = Model(func=_beale, bounds=[(-4.5, 4.5), (-4.5, 4.5)], name="Beale function")

# https://www.sfu.ca/~ssurjano/booth.html
def _booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

booth = Model(func=_booth, bounds=[(-10, 10), (-10, 10)], name="Booth function")

# https://www.sfu.ca/~ssurjano/matya.html
def _matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48*x*y

matyas = Model(func=_matyas, bounds=[(-10, 10), (-10, 10)], name="Matyas function")

# https://www.sfu.ca/~ssurjano/camel3.html
def _three_hump_camel(x, y):
    return 2*x**2 - 1.05*x**4 + x**6 / 6 + x*y + y**2

three_hump_camel = Model(func=_three_hump_camel, bounds=[(-5, 5), (-5, 5)], name="Three-hump Camel function")

# https://www.sfu.ca/~ssurjano/camel6.html
def _six_hump_camel(x, y):
    return (4 - 2.1*x**2 + x**4 / 3) * x**2 + x*y + (-4 + 4*y**2) * y**2

six_hump_camel = Model(func=_six_hump_camel, bounds=[(-3, 3), (-2, 2)], name="Six-hump Camel function")

# https://www.sfu.ca/~ssurjano/easom.html
def _easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi)**2 - (y - np.pi)**2)

easom = Model(func=_easom, bounds=[(-100, 100), (-100, 100)], name="Easom function")

# https://www.sfu.ca/~ssurjano/drop.html
def _drop_wave(x, y):
    return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

drop_wave = Model(func=_drop_wave, bounds=[(-5.12, 5.12), (-5.12, 5.12)], name="Drop-Wave function")

# https://www.sfu.ca/~ssurjano/levy13.html
def _levy13(x, y):
    return (
        np.sin(3*np.pi*x)**2
        + (x - 1)**2 * (1 + np.sin(3*np.pi*y)**2)
        + (y - 1)**2 * (1 + np.sin(2*np.pi*y)**2)
    )

levy13 = Model(func=_levy13, bounds=[(-10, 10), (-10, 10)], name="Levy function N. 13")
