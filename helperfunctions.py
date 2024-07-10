import numpy as np
from numba import njit

hbar = 6.582e-1  # meV ps
lightspeed = 2.9979e2  # µm/ps
echarge = 1e3  # meV ps / (T µm^2)
wavelength = 0.7588  # µm
Eph = 2 * np.pi * hbar * lightspeed / wavelength
a0 = 0.01  # µm
eta1 = 0.0002  # ps^-1 T^-1
gX = -0.364
muB = 5.788e-2  # meV T^-1
gammadia = 0.117
Omega0 = 4
uX = 0.01
detuning0 = 1.9  # meV


@njit
def rabi(B):
    return (Omega0 / np.sqrt(2)) * np.sqrt(
        np.sqrt(1 + 1.5 * ((echarge**2 * a0**4 * B**2) / hbar**2)) + 1
    )


@njit
def detuningp(B):
    return detuning0 + gX * muB * B + gammadia * B**2


@njit
def detuningm(B):
    return detuning0 - gX * muB * B + gammadia * B**2


@njit
def hopfieldp(B):
    D = detuningp(B)
    return 0.5 * (1 + (D / np.sqrt(D**2 + (2 * rabi(B)) ** 2)))


@njit
def hopfieldm(B):
    D = detuningm(B)
    return 0.5 * (1 + (D / np.sqrt(D**2 + (2 * rabi(B)) ** 2)))


@njit
def Ep(B):
    D = detuningp(B)
    return Eph - D / 2 - 0.5 * np.sqrt((2 * rabi(B)) ** 2 + D**2)


@njit
def Em(B):
    D = detuningm(B)
    return Eph - D / 2 - 0.5 * np.sqrt((2 * rabi(B)) ** 2 + D**2)


@njit
def Gp(B):
    return 2 * uX * hopfieldp(B)


@njit
def Gm(B):
    return 2 * uX * hopfieldm(B)


@njit
def alpham(B):
    return 2 * uX * hopfieldm(B) ** 2


@njit
def alphap(B):
    return 2 * uX * hopfieldp(B) ** 2


@njit
def eta(B):
    return eta1 * B


@njit
def Gammasp(B):
    return Gammas + eta(B)


@njit
def Gammasm(B):
    return Gammas - eta(B)


@njit
def normSqr(x):
    return x.real**2 + x.imag**2
