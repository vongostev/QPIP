#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:11:13 2020

@author: vong

numpy.floqt128 version
"""

import functools
import numpy as np
from scipy.special import factorial, binom

DPREC = np.float128

def compose(*functions):
    def pack(x): return x if type(x) is tuple else (x,)

    return functools.reduce(
        lambda acc, f: lambda *y: f(*pack(acc(*pack(y)))), reversed(functions), lambda *x: x)


def lrange(iterable):
    return compose(range, len)(iterable)


def fact(n):
    return DPREC(factorial(n, exact=True))


def positive_index(i, data):
    if i < 0:
        return len(data) + i
    return i


def p_convolve(p_signal, p_noise):
    grid_signal = lrange(p_signal)
    grid_noise = lrange(p_noise)

    def sfun(m): return \
        sum([p_signal[positive_index(m - n, grid_noise)] * p_noise[n]
             for n in grid_noise])
    return np.vectorize(sfun)(grid_signal)


def moment(p, N):
    return sum(i ** N * p[i] for i in lrange(p))


def mean(p):
    return moment(p, 1)


def g2(p):
    m = mean(p)
    s = moment(p, 2)
    if m == 0:
        m = 1
        print('Uncorrect g2. Zero mean change to 1')
    return (s - m) / m ** 2


def fmoment(p, N):
    return sum(fact(i) / fact(i - N) * p[i] for i in lrange(p) if i >= N)


def bmoment(p, N):
    return sum(binom(i, N) * p[i] for i in lrange(p) if i >= N)


def normalize(p):
    return np.array(p) / moment(p, 0)


def abssum(p):
    return sum((p[i]) ** 2 for i in lrange(p))


def fidelity(p1, p2):
    if len(p1) != len(p2):
        print('WARNING:',
              'Probabilities for the fidelity calculation must have the same length')
    plen = min(len(p1), len(p2))
    prod = [p1[i] * p2[i] for i in range(plen)]
    return sum(np.sign(p)*np.sqrt(abs(p)) for p in prod) ** 2


def entropy(p):
    return sum(- e * np.log(e) for e in p if e > 0)


def average_entropy(p):
    return - np.mean([np.log(e / np.mean(p)) for e in p])


def pprint_stats(p):
    print('norm', moment(p, 0))
    print('mean', mean(p))
    print('g2', g2(p))
