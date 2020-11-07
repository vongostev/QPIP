# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:38:13 2020

@author: von.gostev

Основные сущности для анализа ошибок
"""
import numpy as np

from .invpmoms import mrec_matrices, central_moments


def covm_mltnomial(success_prob_distr, meas_num):
    d = success_prob_distr
    N = len(d)
    covm = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            if i != j:
                covm[i, j] = - d[i] * d[j] / meas_num
            else:
                covm[i, i] = d[i] * (1 - d[j]) / meas_num
    return covm


def covm_transform(cov_matrix, transform_matrix):
    return np.dot(np.dot(transform_matrix, cov_matrix),
                  np.transpose(transform_matrix))


def covm_moments_anlt(Q, p, N, M, K):
    sigma = np.zeros((K, K))
    F, Mm, _ = mrec_matrices(p, M, K)
    T = Mm.dot(F)
    for k in range(K):
        for l in range(K):
            if k == l:
                sigma[k, k] = (sum(Q[m]*(1 - Q[m])*T[k, m]**2 for m in range(M)) -
                               sum(sum(Q[m]*Q[s]*T[k, m]*T[k, s] for m in range(M)) for s in range(M)))/N
            else:
                sigma[k, l] = (sum(Q[m]*(1 - Q[m])*T[k, m]*T[l, m] for m in range(M)) -
                               sum(sum(Q[m]*Q[s]*T[k, m]*T[l, s] for m in range(M)) for s in range(M)))/N
    return sigma
