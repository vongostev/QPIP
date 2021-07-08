# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:47:27 2021

@author: vonGostev

Herzog, U. (1996).
Loss-error compensation in quantum-state measurements and the solution of
the time-reversed damping equation.
Physical Review A, 53(3), 1245.
"""
import numba as nb
import numpy as np
from joblib import Parallel, delayed


@nb.njit(nogil=True, fastmath=True)
def nbfact(n: int) -> int:
    return np.prod(np.arange(1, n + 1, 1))


@nb.njit(nogil=True, fastmath=True)
def Cj(eta: float, n: int, j: int, K: int, L: int) -> float:
    return (1 - 1 / (2 ** K * eta)) ** j * nbfact(n + j + np.sum(L)) / nbfact(j)


@nb.njit(nogil=True, fastmath=True)
def Cl(eta: float, k: int, lelem: int) -> float:
    return (- 2 ** (k + 1) * eta) ** (- lelem) / nbfact(lelem)


@nb.njit(nogil=True, fastmath=True)
def get_pn_elems_nb(L: np.ndarray, llen: int, n: int, Q: np.ndarray,
                    eta: float, K: int) -> float:
    elems = np.zeros(llen)
    for i in nb.prange(llen):
        li = L[i]
        ls, j = li[:-1], int(li[-1])

        C1elems = np.zeros(K * K)
        for l in nb.prange(K):
            for k in nb.prange(K):
                C1elems[l * K + k] = Cl(eta, k, ls[l])

        C2 = Cj(eta, n, j, K, ls)
        elems[i] = np.prod(C1elems) * C2 * Q[n + np.sum(li)]
    return np.sum(elems)


def get_pn(n, Q, eta, K):
    M = len(Q)
    L = np.array(np.meshgrid(*[range(M)] * (K + 1))).T
    L = L.reshape(-1, K + 1)
    L = L[np.sum(L, axis=1) < M - n]

    llen = len(L)
    esum = get_pn_elems_nb(L, llen, n, Q, eta, K)

    return esum / eta ** n / nbfact(n)


def Q2PGF(Q, eta, N, K=2):
    if K < 3:
        return [get_pn(n, Q, eta, K) for n in range(N)]
    return Parallel(n_jobs=-2)(
        [delayed(get_pn)(n, Q, eta, K) for n in np.arange(N)])
