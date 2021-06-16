# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:17:25 2019

@author: von.gostev
"""
import numpy as np
from scipy.special import binom
from scipy.special import gamma as Γ

from scipy.stats import poisson, nbinom

from qutip import basis
from qutip.operators import displace, squeeze

from fpdet import normalize


def ppoisson(mean, N, norm=True):
    P = poisson.pmf(np.arange(N), mean)
    return normalize(P) if norm else P


def pthermal(mean, N, norm=True):
    P = pthermal_polarized(mean, 1, N)
    return normalize(P) if norm else P


def pfock(mean, N, norm=True):
    if np.floor(mean) != mean:
        raise ValueError(f'Fock state energy must be int, not {mean}')
    P = np.zeros(N)
    P[mean] = 1
    return normalize(P) if norm else P


def pthermal_photonsub(mean, photonsub, N, norm=True):
    """
    Barnett, Stephen M., et al. 
    "Statistics of photon-subtracted and photon-added states." 
    Physical Review A 98.1 (2018): 013809.

    Parameters
    ----------
    mean : float
        mean of distribution.
    photonsub : int
        count of substracted photons.
    N : int
        maximal photon number.

    Returns
    -------
    float
        Probability for the given photon number.

    """
    n = np.arange(N)
    P = mean ** n / (1 + mean) ** (n + photonsub + 1) * \
        binom(n + photonsub, photonsub)
    return normalize(P) if norm else P


def pthermal_photonadd(mean, photonadd, N, norm=True):
    """
    Barnett, Stephen M., et al.
    "Statistics of photon-subtracted and photon-added states."
    Physical Review A 98.1 (2018): 013809.

    Parameters
    ----------
    mean : float
        mean of distribution.
    photonadd : int
        count of added photons.
    N : int
        maximal photon number.

    Returns
    -------
    float
        Probability for the given photon number.

    """
    n = np.arange(N)
    P = mean ** (n - photonadd) / (1 + mean) ** (n + 1) * binom(n, photonadd)
    P[:photonadd] = 0
    return normalize(P) if norm else P


def phyper_poisson(lam, beta, N, norm=True):
    """
    Bardwell, G. E., & Crow, E. L. (1964).
    A two-parameter family of hyper-Poisson distributions.
    Journal of the American Statistical Association, 59(305), 133-141.
    Formula (6)
    On the Hyper-Poisson Distribution and its
    Generalization with Applications
    Bayo H. Lawal
    Formulas (2.1, 2.2)    

    Parameters
    ----------
    lam : float
        Parameter 1.
    beta : float
        Parameter 2.
    N : int
        maximal photon number.

    Returns
    -------
    float
        Probability for given m.

    """

    def phi_function(beta, lam, N=100):
        k = np.arange(N)
        return np.sum(Γ(beta) / Γ(beta + k) * lam ** k)

    n = np.arange(N)
    phi = phi_function(beta, lam)
    P = Γ(beta) / Γ(beta + n) * lam ** n / phi
    return normalize(P) if norm else P


def psqueezed_coherent1(ampl, sq_coeff, N, norm=True):
    vac = basis(N, 0)
    d = displace(N, ampl)
    s = squeeze(N, sq_coeff)
    print('Squeeze', np.exp(- 2 * np.abs(sq_coeff)) / 4)
    P = d * s * vac
    return normalize(P) if norm else P


def psqueezed_vacuumM(r, theta, M, N, norm=True):
    """
    M-mode squeezed vacuum state

    Parameters
    ---------
    mean : complex
        Pump parameter [0, 1]
    phase : float
        relative phase shift of two modes one by one
    M : int
        number of modes
    N : int
        maximal photon number.

    Returns
    -------
        M-mode squeezed vacuum photon-number distribution
    """
    n = np.arange(N)
    distribution = np.tanh(r) ** n / np.cosh(r) * (1 - n % 2)
    P = distribution ** 2
    return normalize(P) if norm else P


def pthermal_polarized(mean, dof, N, norm=True):
    p = 1 - mean / (dof + mean)
    P = nbinom.pmf(np.arange(N), dof, p)
    return normalize(P) if norm else P
