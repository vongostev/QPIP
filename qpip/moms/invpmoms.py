# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 02:16:22 2020

@author: von.gostev
"""

import numpy as np
from scipy.special import comb, perm
from scipy.linalg import pinv, lstsq
from scipy.optimize import minimize_scalar

from .pymaxent import reconstruct
from .._numpy_core import fact, DPREC, normalize


@np.vectorize
def stirling2(n, k):
    # https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind#Explicit_formula
    K = np.arange(k + 1)
    s2 = np.sum((-1) ** (k - K) * comb(k, K) * K ** n)
    return DPREC(s2 / fact(k))


# =================== VANDERMONDE MATRICES =======================
def vandermonde(nmax, max_order, *args):
    k = np.arange(max_order).reshape((-1, 1))
    n = np.arange(nmax).reshape((1, -1))
    return DPREC(n**k)


def bvandermonde(nmax, max_order, *args):
    """
    Vandermonde type matrix to calculate binomial moments from a PDF

    Parameters
    ----------
    nmax : int
        Length of the PDF.
    max_order : int
        Maximum moment order.
    *args :
        Args for the compatibility.

    Returns
    -------
    np.array((max_order, nmax))
        DESCRIPTION.

    """

    return convandermonde(nmax, max_order, 1., 1.)


def convandermonde(nmax, max_order, qe, z, *args):
    k = np.arange(max_order).reshape((-1, 1))
    n = np.arange(nmax).reshape((1, -1))
    return np.power(1 - qe + qe * z, n - k) * comb(n, k)


# ================== MATRICES TO COMPUTE P MOMENTS =======================
def q2p_mrec_matrices(mmax, max_order, qe):
    k = np.arange(max_order).reshape((-1, 1))
    m = np.arange(mmax).reshape((1, -1))
    F = DPREC(qe ** -k * perm(m, k))
    S = stirling2(k, k.T)
    return S, F


def q2p_convmoms_matrix(mmax, max_order, qe, z):
    k = np.arange(max_order).reshape((-1, 1))
    B = convandermonde(mmax, max_order, 1., z)
    return qe ** -k * B


# ================= COMPUTE MOMENTS =======================
def convmoms(Q, max_order, qe, z):
    B = q2p_convmoms_matrix(len(Q), max_order, qe, z)
    return B.dot(Q)


def bmoms(Q, max_order, qe):
    return convmoms(Q, max_order, qe, 1.)


def imoms(Q, max_order, qe):
    """
    Initial moments

    """

    S, F = q2p_mrec_matrices(len(Q), max_order, qe)
    return S.dot(F).dot(Q)


# ================= MOMENTS UTILS =========================
def precond_moms(W, moms, Q=None, qe=None):
    wmax = np.max(W, axis=1)
    W = np.array([w / np.max(W, axis=1) for w in W.T]).T
    moms = moms / wmax
    # moms = np.append(moms, 1)
    # W = np.vstack((W, np.ones(W.shape[1])))
    # if Q is not None:
    #     moms = np.append(moms, g2(Q))
    #     W = np.vstack((W, [((n - 1) * n if n >= 1 else 0) / mean(Q) ** 2 * qe ** 2 for n in range(W.shape[1])]))
    return W, moms


def mrec_cond(mmax, nmax, qe, max_order=2):
    W = vandermonde(nmax, max_order)
    S, F = q2p_mrec_matrices(mmax, max_order, qe)
    S1 = pinv(W).dot(S).dot(F)
    return np.linalg.norm(S1, 2) * np.linalg.norm(pinv(S1), 2)


# ====================== SOLVERS ==========================
def mrec_maxent_pn(Q, qe, nmax=0, max_order=2):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    moments = imoms(Q, max_order, qe)
    P, _ = reconstruct(moments.astype(np.float64), rndvar=np.arange(nmax))
    return normalize(P)


def rec_pn_generator(vandermonde_fun, moms_fun, Q, nmax, max_order, args):
    """


    Parameters
    ----------
    vandermonde_fun : function
        Vandermonde matrix of definite type of moments.
    moms_fun : function
        Function to calculate moments vector.
    Q : np.array
        Photocounting statistics.
    nmax : int
        Max photon-number to reconstruct.
    max_order : int
        Max order of moments.
    args : tuple
        Args of moms_fun.

    Returns
    -------
    np.array
        Recovered photon-number statistics estimation.

    """
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    moms = moms_fun(Q, max_order, *args)
    vandermonde_matrix = vandermonde_fun(nmax, max_order, *args)
    vandermonde_matrix, moms = precond_moms(vandermonde_matrix, moms)
    return lstsq(vandermonde_matrix, moms)[0]


def convmrec_pn(Q, qe, z, nmax=0, max_order=2):
    return rec_pn_generator(convandermonde,
                            convmoms, Q, nmax, max_order, (qe, z))


def Q2PIM(Q, qe, nmax=0, max_order=2):
    # Reconstruct P from Q with initial moments
    return rec_pn_generator(vandermonde,
                            imoms, Q, nmax, max_order, (qe,))


def Q2PBM(Q, qe, nmax=0, max_order=2):
    # Reconstruct P from Q with binomial moments
    return convmrec_pn(Q, qe, 1, nmax, max_order)


def Q2PCM(Q, qe, nmax=0, max_order=2, zopt=None):
    # Reconstruct P from Q with convergent moments
    if zopt is None:
        res = minimize_scalar(
            lambda z: -sum(x for x in convmrec_pn(Q, qe,
                           z, nmax, max_order) if x < 0),
            bounds=(-1, 1), method="Bounded")
        zopt = res.x
    return convmrec_pn(Q, qe, zopt, nmax, max_order), zopt
