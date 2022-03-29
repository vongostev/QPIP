# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:17:25 2019

@author: von.gostev
"""
import numpy as np
from scipy.special import binom
from scipy.special import gamma as Γ
from scipy.special import beta
from mpmath import hermite


from scipy.stats import poisson, nbinom

from qutip import basis, ket2dm
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
    The photon-number distribution

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
    The photon-number distribution

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
    The photon-number distribution

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
    state = d * s * vac
    rho = ket2dm(state)
    P = rho.diag()
    return normalize(P) if norm else P


def psqueezed_vacuum(r, theta, N, norm=True):
    """
    2-mode squeezed vacuum state

    Parameters
    ---------
    r : complex
        Pump parameter [0, 1]
    theta : float
        relative phase shift of two modes one by one
    N : int
        maximal photon number.

    Returns
    -------
        2-mode squeezed vacuum photon-number distribution
    """
    n = np.arange(N)
    distribution = np.tanh(r) ** n / np.cosh(r) * (1 - n % 2)
    P = distribution ** 2
    return normalize(P) if norm else P


def pthermal_polarized(mean, dof, N, norm=True):
    p = 1 - mean / (dof + mean)
    P = nbinom.pmf(np.arange(N), dof, p)
    return normalize(P) if norm else P


def pcompound_poisson(mu: float, a: float, N: int, norm=True):
    """
    Bogdanov, Y. I., Bogdanova, N. A., Katamadze, K. G., Avosopyants,
    G. V., & Lukichev, V. F. (2016).
    Study of photon statistics using a compound Poisson distribution
    and quadrature measurements.
    Optoelectronics, Instrumentation and Data Processing, 52(5), 475-485.

    Formula (10)

    Parameters
    ----------
    mu : float
        mean value.
    a : float
        a parameter.
    N : int
        maximal photon number.
    norm : bool, optional
        Flag to normalization. The default is True.

    Returns
    -------
    The photon-number distribution

    """
    n = np.arange(N)
    if a > 0:
        P = (mu / a) ** n * Γ(a + n) / Γ(a) / \
            Γ(n + 1) / (1 + mu / a) ** (n + a)
    elif a < 0:
        if int(a) == a and mu == -a:
            P = pfock(-a, N)
        else:
            P = (mu / a) ** n / (beta(a - 1, n + 1) * (a - 1)) / \
                (1 + mu / a) ** (n + a)
    return normalize(P) if norm else P


def psqueezed_displaced(x0: float, p0: float, u: float, N: int, norm=True):
    """
    Dutta B. et al.
    Squeezed states, photon-number distributions, and U (1) invariance
    JOSA B. – 1993. – Т. 10. – №. 2. – С. 253-264.

    Parameters
    ----------
    x0 : float
        DESCRIPTION.
    p0 : float
        DESCRIPTION.
    u : float
        DESCRIPTION.
    N : int
        DESCRIPTION.
    norm : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    numpy.ndarray
        The photon-number distribution.

    """
    assert (u > 0) and (u <= 1)
    n = np.arange(N)
    s1 = 1 / Γ(n + 1) / 2. ** (n - 1)
    s2 = np.sqrt(u) * (1 - u) ** n / (1 + u) ** (n + 1)
    ksi = (x0 + 1j * u * p0) / (1 - u ** 2) ** 0.5
    s3 = np.exp(- u * (x0 ** 2 + p0 ** 2) + (1 - u ** 2) * np.abs(ksi) ** 2)

    P = s1 * s2 * s3 * np.abs(np.vectorize(hermite)(n, ksi)) ** 2
    P = P.astype(np.float64)
    return normalize(P) if norm else P
