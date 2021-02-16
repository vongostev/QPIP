# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 02:16:22 2020

@author: von.gostev
"""

import numpy as np
from sympy.functions.combinatorial.numbers import stirling
# from sympy.functions.combinatorial.factorials import factorial
from scipy.special import binom
from scipy.linalg import pinv, lstsq
from scipy.optimize import minimize_scalar

from .pymaxent import reconstruct
from .._numpy_core import fact, DPREC, normalize, lrange, g2, mean

# pinv = np.linalg.pinv
vec = np.vectorize


def precarray(arr):
    X = np.array(arr, dtype=DPREC)
    for i, x in np.ndenumerate(X):
        try:
            X[i] = DPREC(x)
        except:
            X[i] = DPREC(np.float64(x))
            np.warnings.warn_explicit("Conversion to DPREC %s through np.float64" % DPREC,
                                      RuntimeWarning, __file__, 27)
    return X


# =============== CENTRAL MOMENTS ======================
def cmoms_matrix(max_order, C):
    max_order = range(max_order)
    return np.array([[
        DPREC(binom(n, k) * (-1)**(n-k)) * C**(n-k) for k in max_order]
        for n in max_order])


def central_moments(moms, C=None):
    """
    https://mathworld.wolfram.com/CentralMoment.html

    Parameters
    ----------
    moms : array
        array of initial moments.
    C : number
        Coordinate of the centralization.
        If C is None, moments are centralized around mean (moms[1]).

    Returns
    -------
    cmoms : array
        array of central moments.

    """
    if C is None:
        C = moms[1]
    cm_matrix = cmoms_matrix(len(moms), C)
    return cm_matrix.dot(moms)


# =================== VANDERMONDE MATRICES =======================
def vandermonde(nmax, max_order):
    return np.array([[DPREC(n**k) for n in range(nmax)] for k in range(max_order)])


def bvandermonde(nmax, max_order):
    return np.array([[binom(n, k) for n in range(nmax)] for k in range(max_order)])


def convandermonde(nmax, z, qe, max_order):
    return np.array([[
        np.power(1 - qe + qe * z, n - k) * binom(n, k) if n >= k else 0 for n in range(nmax)]
        for k in range(max_order)])


# ================== MATRICES TO COMPUTE P =======================
def mrec_matrices(qe, mmax, nmax, max_order):
    F = np.array([[
        DPREC(qe ** -s * fact(i) / fact(i - s) if i >= s else 0)
        for i in range(mmax)] for s in range(max_order)], dtype=DPREC)
    S = np.array([[
        DPREC(int(stirling(k, s, kind=2))) for s in range(max_order)]
        for k in range(max_order)], dtype=DPREC)
    W = vandermonde(nmax, max_order)
    return W, S, F


def bmrec_matrices(qe, mmax, nmax, max_order):
    B = np.array([[
        DPREC(qe) ** -s * binom(i, s) if i >= s else 0 for i in range(mmax)]
        for s in range(max_order)], dtype=DPREC)
    W = bvandermonde(nmax, max_order)
    return W, B


# ================= COMPUTE MOMENTS =======================
def convmoms(Q, qe, z, max_order):
    B = np.array([[
        DPREC(qe ** -s * z ** (i - s) * binom(i, s)) if i >= s else 0 for i in lrange(Q)]
        for s in range(max_order)], dtype=DPREC)
    return B.dot(Q)


def bmoms(Q, qe, max_order):
    B = np.array([[
        DPREC(qe) ** -s * binom(i, s) if i >= s else 0 for i in lrange(Q)]
        for s in range(max_order)], dtype=DPREC)
    return B.dot(Q)


def imoms(Q, qe, max_order):
    F = np.array([[
        DPREC(qe ** -s * fact(i) / fact(i - s) if i >= s else 0)
        for i in lrange(Q)] for s in range(max_order)], dtype=DPREC)
    S = np.array([[
        DPREC(int(stirling(k, s, kind=2))) for s in range(max_order)]
        for k in range(max_order)], dtype=DPREC)
    return S.dot(F).dot(Q)


# ================= MOMENTS UTILS =========================
def pn_moms(Q, qe, max_order):
    mmax = len(Q)
    W, S, F = mrec_matrices(qe, mmax, 2, max_order)
    return S.dot(F).dot(Q)


def precond_moms(W, moms, Q=None, qe=None):
    wmax = np.max(W, axis=1)
    W = np.array([w / np.max(W, axis=1) for w in W.T]).T
    moms = moms / wmax
    moms = np.append(moms, 1)
    W = np.vstack((W, np.ones(W.shape[1])))
    if Q is not None:
        moms = np.append(moms, g2(Q))
        W = np.vstack((W, [((n - 1) * n if n >= 1 else 0) / mean(Q) ** 2 * qe ** 2 for n in range(W.shape[1])]))
    return W, moms


def mrec_cond(mmax, nmax, qe, max_order=2):
    W, S, F = mrec_matrices(qe, mmax, nmax, max_order)
    S1 = pinv(W).dot(S).dot(F)
    return np.linalg.norm(S1, 2) * np.linalg.norm(pinv(S1), 2)


def fmatrix(qe, z, nmax, max_order):
    F = np.zeros((max_order, max_order))
    nk = nmax - max_order
    for i in range(max_order):
        for n in range(max_order):
            F[i, n] = sum((-1) ** n * (1 - qe + qe * z) ** (nk + i - k) *
                          binom(k, n) * binom(nk + i, k) for k in range(max_order, nmax))
    return F

# ====================== SOLVERS ==========================


def mrec_pn(Q, qe, nmax=0, max_order=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W = vandermonde(nmax, max_order)
    moms = imoms(Q, qe, max_order)
    W, moms = precond_moms(W, moms)
    P = lstsq(W, moms)[0]
    return normalize(P)


def bmrec_pn(Q, qe, nmax=0, max_order=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W = bvandermonde(nmax, max_order)
    moms = bmoms(Q, qe, max_order)
    W, moms = precond_moms(W, moms)
    P = lstsq(W, moms)[0]
    return normalize(P)


def convmrec_pn(Q, qe, z, nmax=0, max_order=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W = convandermonde(nmax, z, qe, max_order)
    moms = convmoms(Q, qe, z, max_order)
    W, moms = precond_moms(W, moms)
    return lstsq(W, moms)[0]


def Q2PCM(Q, qe, nmax=0, max_order=2):
    res = minimize_scalar(
        lambda z: -sum(x for x in convmrec_pn(Q, qe,
                                              z, nmax, max_order) if x < 0),
        bounds=(0, 1), method='Bounded')
    print(res)
    zopt = res.x
    return convmrec_pn(Q, qe, zopt, nmax, max_order), zopt


def mrec_maxent_pn(Q, qe, nmax=0, max_order=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W, S, F = mrec_matrices(qe, mmax, nmax, max_order)
    moments = S.dot(F).dot(Q)
    P, _ = reconstruct(moments.astype(np.float64), rndvar=np.arange(nmax))
    return normalize(P)


def convmrec_analytical(Q, qe, z, nmax, max_order):
    def mtxelem(m, n):
        return sum(
            z ** (m - n) * qe ** (-n) * ((qe-1) / (qe * z) - 1) ** (k-n) *
            binom(m, k) * binom(k, n) for k in range(max_order))

    P = np.zeros(nmax)
    for n in range(nmax):
        P[n] = sum(mtxelem(m, n) * Q[m] for m in lrange(Q))
    return P
