# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:49:12 2020

@author: vonGostev
"""
import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, Q2P, normalize, fidelity
from qpip.detection_core import invt_matrix
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')


def dsigma(x):
    return np.diag(x)


max_order = 10
N = 50
qe = 0.3
N0 = 10**8
z = 0.5

offset = 9
mult = 0.5
nlist = np.arange(offset + 1, N)
mdisp = []
ddisp = []
mfid = []
dfid = []

for n in nlist:
    max_order = min(n - 2, N // 2) #int(n * (1 - n / 2 / N))
    P = pfock(n, n + 1)
    Q = P2Q(P, qe)
    d = dsigma(covm_conv_pn(Q, qe, z, N0, n + 1, max_order))
    d2 = dsigma(covm_transform(covm_mltnomial(Q, N0), invt_matrix(qe, n + 1, len(Q))))
    mdisp.append(np.linalg.norm(d, 1))
    ddisp.append(np.linalg.norm(d2, 1))
    dfid.append(fidelity(P, Q2P(Q, qe)))
    mfid.append(fidelity(P, convmrec_pn(Q, qe, z, max_order=max_order)))
    
plt.semilogy(nlist, mdisp)
plt.semilogy(nlist, ddisp)
plt.ylabel("СКО восст. фок. распр.")
plt.xlabel("Число фотонов в фок. сост.")
plt.show()

plt.plot(nlist, mfid)
plt.plot(nlist, dfid)
plt.xlabel("Число фотонов в фок. сост.")
plt.ylabel("Достоверность восст. распр.")
plt.show()