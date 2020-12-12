# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 23:36:57 2020

@author: von.gostev
"""
import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, Q2P, normalize, fidelity, entropy
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "light"
plt.rcParams["font.size"] = 12


def SM_maximums(N, K):
    F, M, S = mrec_matrices(qe, N, K+1)
    return [max(np.abs(np.linalg.pinv(S).dot(M)[:, k])) for k in range(K+1)]


def make_qmodel(P, qe, mtype='binomial', n_cells=0, M=0, N0=int(1e6), ERR=0):
    if M == 0:
        M = len(P)
    Q = P2Q(P, qe, M, mtype, n_cells)
    QND = np.random.choice(range(M), size=N0, p=Q.astype(float))
    QN = np.histogram(QND, bins=range(M + 1), density=True)[0]
    if ERR:
        QN = cmp(normalize, np.abs)(QN*(1 + np.random.uniform(-ERR, ERR, size=len(QN))))
    return Q, QN


mean = 4
MO = 7
N = 40
qe = 0.3

#P = psqueezed_vacuumM(N, 2, 1.5, 0)
#P = normalize(ppoisson(1, N) + ppoisson(7, N))
#P = pfock(4, N)
P = pthermal(3, N)
P = ppoisson(10, N)

qe_list = np.arange(0.1, 1.05, 0.1)
mpfids = []
cpfids = []
for qe in qe_list:
    #iP = mrec_pn(Q1, qe, max_order=MO)
    Q, Q1 = make_qmodel(P, qe, M=N, N0=int(1E6))
    #bP = bmrec_pn(Q1, qe, max_order=MO)
    mP = Q2P(Q1, qe)
    z = 0.5
    cP = convmrec_pn(Q1, qe, z, max_order=MO)
    cpfids.append(fidelity(cP,P))
    mpfids.append(fidelity(mP,P))

#plt.plot(P)
#plt.plot(iP,
#    label='initial moments,\n' + r'$\Delta=%.4f$' % np.linalg.norm(P-iP))
#plt.plot(bP, '--',
#    label='binomial moments,\n' + r'$\Delta=%.4f$' % np.linalg.norm(P-bP))
#plt.plot(cP,
#    label='convergent moments,\nz=%.1e, ' % z + r'$\Delta=%.4f$' % np.linalg.norm(P-cP))
plt.semilogy(qe_list, mpfids)
plt.plot(qe_list, cpfids)
plt.legend(frameon=0)
plt.show()
