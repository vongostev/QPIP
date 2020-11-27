#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 01:27:06 2020

@author: vong
"""

import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, normalize, fidelity
import matplotlib.pyplot as plt
import numpy as np


def make_qmodel(P, qe, mtype='binomial', n_cells=0, M=0, N0=int(1e6), ERR=0):
    if M == 0:
        M = len(P)
    Q = P2Q(P, qe, M, mtype, n_cells)
    QND = np.random.choice(range(M), size=N0, p=Q.astype(float))
    QN = np.histogram(QND, bins=range(M + 1), density=True)[0]
    if ERR:
        QN = cmp(normalize, np.abs)(QN*(1 + np.random.uniform(-ERR, ERR, size=len(QN))))
    return Q, QN


max_order = 8
N = 32
qe = 0.4
z = 0.3

P = normalize(ppoisson(1, N) + ppoisson(7, N))

errs = []
sporders = np.arange(4, 9)
for order in sporders:
    errs.append([])
    for i in range(20):
        Q, Q1 = make_qmodel(P, qe, M=N, N0=10**order)
        cP = convmrec_pn(Q1, qe, z, max_order=max_order)
        #plt.plot(cP)
        errs[-1].append(np.linalg.norm(P - cP))

#plt.show()
plt.boxplot(errs, labels=[r'$10^{%d}$'% s for s in sporders])
plt.yscale('log')
plt.show()
    
