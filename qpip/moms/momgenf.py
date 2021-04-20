# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:47:27 2021

@author: vonGostev

Herzog, U. (1996). 
Loss-error compensation in quantum-state measurements and the solution of
the time-reversed damping equation. 
Physical Review A, 53(3), 1245.
"""

import numpy as np
from .._numpy_core import fact, lrange
from joblib import Parallel, delayed


def Cj(eta, n, j, K, L):
    return (1 - 1 / (2 ** K * eta)) ** j * fact(n + j + sum(L)) / fact(j)


@np.vectorize
def Cl(eta, k, l):
    return (- 2 ** (k + 1) * eta) ** (- l) / fact(l)


def get_pn(n, Q, eta, K):
    L = np.array(np.meshgrid(*[lrange(Q)
                               for i in range(K + 1)])).T.reshape(-1, K + 1)
    L = list(filter(lambda x: np.sum(x) < len(Q) - n, L))

    def pmember(li):
        ls, j = li[:-1], li[-1]
        C1 = Cl(eta, lrange(ls), ls) if len(ls) else []
        C2 = Cj(eta, n, j, K, ls)
        return np.prod(C1) * C2 * Q[n + sum(li)]

    return np.sum(np.apply_along_axis(pmember, 1, L)) / eta ** n / fact(n)


def get_pn0(n, Q, eta, K):
    coeffs = []
    for j in range(len(Q) - n):
        C2 = Cj(eta, n, j, K, [])
        coeffs.append(C2)
    return sum(coeffs[j] * Q[n + j] for j in range(len(Q) - n)) / eta ** n / fact(n)


def Q2PGF(Q, eta, N, K=2):
    return Parallel(n_jobs=-1)(delayed(get_pn)(n, Q, eta, K) for n in range(N))
