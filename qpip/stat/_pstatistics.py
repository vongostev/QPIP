# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:17:25 2019

@author: von.gostev
"""
import numpy as np
from scipy.special import gamma, binom

from qutip import thermal_dm, fock_dm, coherent_dm, basis
from qutip.operators import displace, squeeze

from .._numpy_core import normalize


def correct_distrib(p, non_ideal=False, dtype='default'):
    if non_ideal:
        p[p < 1e-5] = 0
    if dtype == 'default':
        return normalize(p)
    else:
        p = np.array([dtype(x) for x in p])
        return normalize(p)


def ppoisson(mean, N, non_ideal=False, dtype='default'):
    p = coherent_dm(N, np.sqrt(mean)).diag()
    return correct_distrib(p, non_ideal, dtype)


def pthermal(mean, N, non_ideal=False, dtype='default'):
    p = thermal_dm(N, float(mean)).diag()
    return correct_distrib(p, non_ideal, dtype)


def pfock(mean, N, non_ideal=False, dtype='default'):
    p = fock_dm(N, int(mean)).diag()
    return correct_distrib(p, non_ideal, dtype)


@np.vectorize
def pthermal_photonsub(N, mean, photonsub):
    """
    Barnett, Stephen M., et al. 
    "Statistics of photon-subtracted and photon-added states." 
    Physical Review A 98.1 (2018): 013809.

    Parameters
    ----------
    N : int
        maximal photon number.
    mean : float
        mean of distribution.
    photonsub : int
        count of substracted photons.

    Returns
    -------
    float
        Probability for the given photon number.

    """
    return np.array([
        mean ** n / (1 + mean) ** (n + photonsub + 1) *
        binom(n + photonsub, photonsub)
        for n in range(N)])


def pthermal_photonadd(N, mean, photonadd):
    """
    Barnett, Stephen M., et al.
    "Statistics of photon-subtracted and photon-added states."
    Physical Review A 98.1 (2018): 013809.

    Parameters
    ----------
    n : int
        maximal photon number.
    mean : float
        mean of distribution.
    photonadd : int
        count of added photons.

    Returns
    -------
    float
        Probability for the given photon number.

    """
    P = np.array([mean ** (n - photonadd) / (1 + mean) ** (n + 1)
                  * binom(n, photonadd) for n in range(N)])
    P[:photonadd] = 0
    return P


@np.vectorize
def phyper_poisson(m, lam, beta):
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
    m : int
        discrete variable.
    lam : float
        Parameter 1.
    beta : float
        Parameter 2.

    Returns
    -------
    float
        Probability for given m.

    """

    def phi_function(beta, lam, N=1000):
        return sum(gamma(beta) / gamma(beta + k) * lam ** k
                   for k in range(N))

    phi = phi_function(beta, lam)
    return gamma(beta) / gamma(beta + m) * lam ** m / phi


def psqueezed_coherent1(N, ampl, sq_coeff):
    vac = basis(N, 0)
    d = displace(N, ampl)
    s = squeeze(N, sq_coeff)
    print('Squeeze', np.exp(- 2 * np.abs(sq_coeff)) / 4)
    return (d * s * vac).unit()


def psqueezed_vacuumM(N, M, r, theta):
    """
    M-mode squeezed vacuum state
    ------------------------------
    Parameters:
        N : int
            number of fock levels in the generated state
        M : int
            number of modes
        mean : complex
            Pump parameter [0, 1]
        phase : float
            relative phase shift of two modes one by one

    Returns:
        A Qobj instance that represents
        M-mode squeezed vacuum state as ket vector
    """
    distribution = np.array(
        [(np.tanh(r) ** n / np.cosh(r) if n % 2 == 0 else 0) for n in range(N)])
    return normalize(distribution ** 2)
