# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:38:13 2020

@author: von.gostev

Основные сущности для анализа ошибок
"""
import numpy as np
from scipy.special import binom
from scipy.linalg import pinv

from .invpmoms import convandermonde, q2p_mrec_matrices, vandermonde, q2p_convmoms_matrix
from .. import lrange, DPREC
from .._dcore import invd_matrix


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
    return np.dot(transform_matrix, cov_matrix) @ np.transpose(transform_matrix)


def covm_convmoms(Q, qe, z, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    T = q2p_convmoms_matrix(len(Q), max_order, qe, z)
    return covm_transform(convm, T)


def covm_conv_pn(Q, qe, z, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, z, N0, max_order)
    T = pinv(convandermonde(nmax, max_order, qe, z))
    return covm_transform(convm, T).astype(DPREC)


def covm_bmoms(Q, qe, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    T = q2p_convmoms_matrix(len(Q), max_order, qe, 1.)
    return covm_transform(convm, T)


def covm_bmoms_pn(Q, qe, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, 1, N0, max_order)
    T = pinv(convandermonde(nmax, max_order, qe, 1))
    return covm_transform(convm, T).astype(DPREC)


def covm_imoms(Q, qe, N0, max_order):
    convm = covm_mltnomial(Q, N0)
    S, F = q2p_mrec_matrices(len(Q), max_order, qe)
    return covm_transform(convm, S.dot(F))


def covm_imoms_pn(Q, qe, N0, nmax, max_order):
    convm = covm_convmoms(Q, qe, 1., N0, max_order)
    T = pinv(vandermonde(nmax, max_order))
    return covm_transform(convm, T).astype(DPREC)


def covm_inv_pn(Q, qe, N0, nmax):
    convm = covm_mltnomial(Q, N0)
    T = invd_matrix(qe, nmax, len(Q))
    return covm_transform(convm, T).astype(DPREC)
