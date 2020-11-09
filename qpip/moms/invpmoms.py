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

from .pymaxent import reconstruct
from .._numpy_core import fact, DPREC, normalize, lrange

# pinv = np.linalg.pinv
vec = np.vectorize


def precarray(arr):
    X = np.array(arr, dtype=DPREC)
    for i, x in np.ndenumerate(X):
        try:
            X[i] = DPREC(x)
        except:
            X[i] = DPREC(np.float64(x))
            np.warnings.warn_explicit("Conversion to DPREC %s via np.float64" % DPREC,
                                      RuntimeWarning, __file__, 27)
    return X


# =============== CENTRAL MOMENTS ======================
def cmoms_matrix(K, C):
    K = range(K)
    return np.array([[DPREC(binom(n, k) * (-1)**(n-k)) * C**(n-k) for k in K]
                     for n in K])


def central_moments(moms, C=None):
    """
    https://mathworld.wolfram.com/Centralnmaxoment.html

    Parameters
    ----------
    moms : array
        array of initial moments.

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
def vandermonde(nmax, K):
    return np.array([[DPREC(n**k) for n in range(nmax)] for k in range(K)])


def bvandermonde(nmax, K):
    return np.array([[binom(n, k) for n in range(nmax)] for k in range(K)])


def convandermonde(nmax, z, qe, K):
    return np.array([[(1 - qe + qe * z) ** (n - k) * binom(n, k) for n in range(nmax)] for k in range(K)])


# ================== MATRICES TO COMPUTE P =======================
def mrec_matrices(qe, mmax, nmax, K):
    F = np.array([[
        DPREC(qe ** -s * fact(i) / fact(i - s) if i >= s else 0)
        for i in range(mmax)] for s in range(K)], dtype=DPREC)
    S = np.array([[
        DPREC(int(stirling(k, s, kind=2))) for s in range(K)]
        for k in range(K)], dtype=DPREC)
    W = vandermonde(nmax, K)
    return W, S, F


def bmrec_matrices(qe, mmax, nmax, K):
    B = np.array([[
        DPREC(qe) ** -s * binom(i, s) if i >= s else 0 for i in range(mmax)]
        for s in range(K)], dtype=DPREC)
    W = bvandermonde(nmax, K)
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
def pn_moms(Q, qe, K):
    mmax = len(Q)
    W, S, F = mrec_matrices(qe, mmax, 2, K)
    return S.dot(F).dot(Q)


def precond_moms(W, moms):
    wmax = np.max(W, axis=1)
    W = np.array([w / np.max(W, axis=1) for w in W.T]).T
    moms = moms / wmax
    return W, moms


def mrec_cond(mmax, nmax, qe, K=2):
    W, S, F = mrec_matrices(qe, mmax, nmax, K)
    S1 = pinv(W).dot(S).dot(F)
    return np.linalg.norm(S1, 2) * np.linalg.norm(pinv(S1), 2)


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
    P = lstsq(W, moms)[0]
    return normalize(P)


def mrec_maxent_pn(Q, qe, nmax=0, K=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W, S, F = mrec_matrices(qe, mmax, nmax, K)
    moments = S.dot(F).dot(Q)
    P, _ = reconstruct(moments.astype(np.float64), rndvar=np.arange(nmax))
    return normalize(P)
