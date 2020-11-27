#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 02:18:13 2020

@author: vong
"""
import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, Q2P, normalize, fidelity
from qpip.detection_core import invt_matrix
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

def make_qmodel(P, qe, mtype='binomial', n_cells=0, M=0, N0=int(1e6), ERR=0):
    if M == 0:
        M = len(P)
    Q = P2Q(P, qe, M, mtype, n_cells)
    QND = np.random.choice(range(M), size=N0, p=Q.astype(float))
    QN = np.histogram(QND, bins=range(M + 1), density=True)[0]
    if ERR:
        QN = cmp(normalize, np.abs)(QN*(1 + np.random.uniform(-ERR, ERR, size=len(QN))))
    return Q, QN

def dispsum(Q, qe, z, N0, N, max_order):
    #return np.linalg.norm(convmrec_pn(Q - Q1, qe, z, max_order=max_order))
    return np.diag(covm_convmoms(Q, qe, z, N0, max_order))


max_order = 10
N = 25
qe = 0.3
N0 = 10**20
   
P = normalize(ppoisson(1, N) + ppoisson(7, N) + ppoisson(17, N))
P = ppoisson(9, N)
#P = pthermal(2, N)
Q = P2Q(P, qe)
#_, Q = make_qmodel(P, qe, N0=N0)
zlist = 10. ** np.arange(-10, 0, 0.1)
errs = []
disps = []
for z in zlist:
    cP = convmrec_pn(Q, qe, z, max_order=max_order)
    errs.append(np.linalg.norm(cP - P))
    disps.append(abs(sum(cP[cP < 0])))

plt.loglog(zlist, errs)
plt.loglog(zlist, disps)
plt.show()

cP = convmrec_pn(Q, qe, zlist[np.argmin(disps)], max_order=max_order)
d = np.diag(covm_conv_pn(Q, qe, zlist[np.argmin(disps)], N0, N, max_order)) ** 0.5
dP = Q2P(Q, qe)
d2 = covm_transform(covm_mltnomial(Q, N0), invt_matrix(qe, N, len(Q)))
plt.fill_between(range(N), cP - 3 * d, cP + 3 * d, alpha=0.5)
plt.plot(cP)
plt.plot(P)
plt.show()
plt.plot(dP)
plt.plot(P)
plt.semilogy(dP + 3 * np.diag(d2) ** 0.5)
plt.plot(dP - 3 * np.diag(d2) ** 0.5)
plt.show()