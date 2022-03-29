#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:59:54 2020

@author: vong
"""
import numpy as np
from scipy.special import gamma as Γ
from fpdet import normalize

# ========================= PHOTOCOUNTING STATISTICS ==========================


def qthermal_unpolarized(mean, dof, N, norm=True):
    """
    Дж. Гудмен, Статистическая оптика, ф-ла 9.2.29 при P = 0

    Parameters
    ----------
    mean : float
        Mean value of the distribution.
    dof : int
        Ration of measurement time to coherence time.
    N : int
        Maximal photocounts number.

    Returns
    -------
    np.ndarray
        Photounts distribution.

    """
    @np.vectorize
    def fsum(_m, dof):
        k = np.arange(_m + 1)
        return np.sum(Γ(_m - k + dof) / (Γ(_m - k + 1) * Γ(dof)) *
                      Γ(k + dof) / (Γ(k + 1) * Γ(dof)))

    m = np.arange(N)
    P = fsum(m, dof) * (1 + 2 * dof / mean) ** (- m) * \
        (1 + mean / 2 / dof) ** (- 2 * dof)
    return normalize(P) if norm else P


def qthermal_polarized(mean, dof, N, norm=True):
    """
    Дж. Гудмен, Статистическая оптика, ф-ла 9.2.24 

    Parameters
    ----------
    mean : float
        Mean value of the distribution.
    dof : int
        Ration of measurement time to coherence time.
    N : int
        Maximal photocounts number.

    Returns
    -------
    np.ndarray
        Photounts distribution.

    """
    m = np.arange(N)
    P = Γ(m + dof) / (Γ(m + 1) * Γ(dof)) * (1 + dof /
                                            mean) ** (- m) * (1 + mean / dof) ** (- dof)
    return normalize(P) if norm else P
