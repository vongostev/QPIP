#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:59:54 2020

@author: vong
"""
import numpy as np
from scipy.special import gamma


# ========================= PHOTOCOUNTING STATISTICS ==========================
def qthermal_unpolarized(m, mean, M):
    """ Дж. Гудмен, Статистическая оптика, ф-ла 9.2.29 при P = 0 """
    return sum(gamma(m - k + M) / (gamma(m - k + 1) * gamma(M)) *
               gamma(k + M) / (gamma(k + 1) * gamma(M))
               for k in np.arange(m + 1)) * \
        (1 + 2 * M / mean) ** (- m) * (1 + mean / 2 / M) ** (- 2 * M)


@np.vectorize
def qthermal_polarized(m, mean, M):
    """ Дж. Гудмен, Статистическая оптика, ф-ла 9.2.24 """
    return gamma(m + M) / (gamma(m + 1) * gamma(M)) * \
        (1 + M / mean) ** (- m) * (1 + mean / M) ** (- M)
