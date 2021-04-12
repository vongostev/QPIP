# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:38:13 2020

@author: von.gostev

Основные сущности для анализа ошибок
"""
import numpy as np
from scipy.special import binom
from scipy.linalg import pinv

from .invpmoms import convandermonde, mrec_matrices, vandermonde
from .. import lrange, DPREC
from ..detection_core import invt_matrix


def covm_mltnomial(success_prob_distr, meas_num):
    d = success_prob_distr
    N = len(d)
    covm = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            if i != j:
                covm[i, j] = - d[i] * d[j] / DPREC(meas_num)
            else:
                covm[i, i] = d[i] * (1 - d[j]) / DPREC(meas_num)
    return covm


def covm_transform(cov_matrix, transform_matrix):
    return np.dot(np.dot(transform_matrix, cov_matrix),
                  np.transpose(transform_matrix))


def covm_convmoms(Q, qe, z, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    T = np.array([[
        DPREC(qe ** -s * z ** (i - s) * binom(i, s)) if i >= s else 0 for i in lrange(Q)]
        for s in range(max_order)], dtype=DPREC)
    return covm_transform(convm, T)


def covm_conv_pn(Q, qe, z, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, z, N0, max_order)
    T = pinv(convandermonde(nmax, qe, z, max_order))
    return covm_transform(convm, T).astype(DPREC)


def covm_bmoms(Q, qe, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    T = np.array([[
        DPREC(qe ** -s * binom(i, s)) if i >= s else 0 for i in lrange(Q)]
        for s in range(max_order)], dtype=DPREC)
    return covm_transform(convm, T)


def covm_bmoms_pn(Q, qe, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, 1, N0, max_order)
    T = pinv(convandermonde(nmax, qe, 1, max_order))
    return covm_transform(convm, T).astype(DPREC)


def covm_imoms(Q, qe, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    W, S, F = mrec_matrices(qe, len(Q), 2, max_order)
    return covm_transform(convm, S.dot(F))


def covm_imoms_pn(Q, qe, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, 1, N0, max_order)
    T = pinv(vandermonde(nmax, max_order))
    return covm_transform(convm, T).astype(DPREC)


def covm_inv_pn(Q, qe, N0, nmax):
    convm = covm_mltnomial(Q, N0)
    T = invt_matrix(qe, nmax, len(Q))
    return covm_transform(convm, T).astype(DPREC)
