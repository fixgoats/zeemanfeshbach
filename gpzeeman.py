import numpy as np
from helperfunctions import *
from numba import njit
from scipy.integrate import RK45


R = 0.012
gamma = 0.2
Gamma = 0.05
W = Gamma / 2
Gammas = Gamma / 5
EXY = 0.0004
P = 10
xi = 20

B = 0


def f(t, y):
    return np.array(
            


y0 = np.array([0.01j, 0.01j, 0, 0]

gpsolver = RK45(f, 0, y0)
