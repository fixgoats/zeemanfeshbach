import numpy as np
from numba import njit
from scipy.integrate import RK45


Ep = 1
Em = 2
R = 0.0016
gamma = 0.2
alphap = 0.0004
alpham = 0.0004
Gp = 0.04
Gm = 0.04
W = 1
EXY = 0.0006
Gamma = 0.4
P = 10


def eta(B):
    return B


def Gammasp(B):
    return Gammas + eta(B)


def Gammasm(B):
    return Gammas - eta(B)



@njit
def normSqr(x):
    return x.real**2 + x.imag**2

def f(t, y):
    return (
            -1j*(E
