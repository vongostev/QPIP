# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 23:36:57 2020

@author: von.gostev
"""
import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, normalize, fidelity
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
MO = 8
N = 60
qe = 0.7

#P = psqueezed_vacuumM(N, 2, 1.5, 0)
P = normalize(ppoisson(1, N) + ppoisson(7, N))
#P = pfock(4, N)
#P = pthermal(1.5, N)
#P = ppoisson(5, N)
Q, Q1 = make_qmodel(P, qe, M=N, N0=100000)
#Q[65:] = 0
#Q[:10] = 0
qe_list = np.arange(0.1, 1.05, 0.1)
iP = mrec_pn(Q1, qe, max_order=MO)
bP = bmrec_pn(Q1, qe, max_order=MO)
z = 0.00001
cP = convmrec_pn(Q1, qe, z, max_order=MO)

plt.plot(P)
plt.plot(iP,
    label='initial moments,\n' + r'$\Delta=%.4f$' % np.linalg.norm(P-iP))
plt.plot(bP, '--',
    label='binomial moments,\n' + r'$\Delta=%.4f$' % np.linalg.norm(P-bP))
plt.plot(cP,
    label='convergent moments,\nz=%.1e, ' % z + r'$\Delta=%.4f$' % np.linalg.norm(P-cP))
plt.legend(frameon=0)
plt.show()

# plt.semilogy(qe_list, [np.linalg.norm(np.array(mrec_pn(Q1, qe, max_order=MO), dtype=np.float)
#                                      - ss.correct_poisson(mean / qe, N, dtype=float)) for qe in qe_list])
"""
K = 9
plt.semilogy(SM_maximums(25,K), 'o--', label='N=25')
#plt.semilogy(np.exp(-np.arange(0,K+1,1)**2*0.2367) * 1.451)
plt.semilogy(SM_maximums(50,K), 's--', label='N=50')
#plt.semilogy(np.exp(-np.arange(0,K+1,1)**2*0.2974) * 0.07252)
plt.semilogy(SM_maximums(100,K), 'd--', label='N=100')
#plt.semilogy(np.exp(-np.arange(0,K+1,1)**2*0.325) * 3e-04)

plt.xlabel('s', fontsize=14)
plt.ylabel(r'$\max |S^{+}\cdot M|_{ns}$')
plt.legend(frameon=False)
plt.savefig('./img/max_sm.eps', dpi=300)

plt.show()
"""
