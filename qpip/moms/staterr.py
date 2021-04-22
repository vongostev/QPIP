# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:38:13 2020

@author: von.gostev

Основные сущности для анализа ошибок
"""
import numpy as np
from scipy.linalg import pinv

from .invpmoms import convandermonde, q2p_mrec_matrices, vandermonde, q2p_convmoms_matrix
from fpdet import DPREC, invd_matrix


def covm_mltnomial(success_prob_distr: np.array, meas_num: int):
    d = success_prob_distr
    covm = d.reshape((-1, 1)) @ d.reshape((1, - 1))
    return (np.diag(d) - covm) / meas_num


def covm_transform(cov_matrix: np.array, transform_matrix: np.array):
    return transform_matrix @ cov_matrix @ np.transpose(transform_matrix)


def covm_convmoms(Q: np.array, qe: float, z: float, N0: int, max_order: int):
    convm = covm_mltnomial(Q, N0)
    T = q2p_convmoms_matrix(len(Q), max_order, qe, z)
    return covm_transform(convm, T)


def covm_conv_pn(Q: np.array, qe: float, z: float, N0: int, nmax: int, max_order: int):
    convm = covm_convmoms(Q, qe, z, N0, max_order)
    T = pinv(convandermonde(nmax, max_order, qe, z))
    return covm_transform(convm, T).astype(DPREC)


def covm_bmoms(Q: np.array, qe: float, N0: int, max_order: int):
    convm = covm_mltnomial(Q, N0)
    T = q2p_convmoms_matrix(len(Q), max_order, qe, 1.)
    return covm_transform(convm, T)


def covm_bmoms_pn(Q: np.array, qe: float, N0: int, nmax: int, max_order: int):
    convm = covm_convmoms(Q, qe, 1, N0, max_order)
    T = pinv(convandermonde(nmax, max_order, qe, 1))
    return covm_transform(convm, T).astype(DPREC)


def covm_imoms(Q: np.array, qe: float, N0: int, max_order: int):
    convm = covm_mltnomial(Q, N0)
    S, F = q2p_mrec_matrices(len(Q), max_order, qe)
    return covm_transform(convm, S @ F)


def covm_imoms_pn(Q: np.array, qe: float, N0: int, nmax: int, max_order: int):
    convm = covm_convmoms(Q, qe, 1., N0, max_order)
    T = pinv(vandermonde(nmax, max_order))
    return covm_transform(convm, T).astype(DPREC)


def covm_inv_pn(Q: np.array, qe: float, N0: int, nmax: int):
    convm = covm_mltnomial(Q, N0)
    T = invd_matrix(qe, nmax, len(Q))
    return covm_transform(convm, T).astype(DPREC)
