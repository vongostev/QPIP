# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:20:29 2020

@author: vonGostev
"""
import __init__
from qpip.moms import *
from qpip.stat import pthermal, psqueezed_vacuumM, ppoisson, pfock
from qpip import P2Q, Q2P, normalize, fidelity, entropy
import matplotlib.pyplot as plt
import numpy as np


#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.weight"] = "light"
#plt.rcParams["font.size"] = 12
plt.style.use('bmh')

mean = 4
MO = 7
N = 40

P = psqueezed_vacuumM(N, 2, 1.5, 0)
P = normalize(ppoisson(1, N) + ppoisson(7, N))
#P = pfock(20, 21)
P = pthermal(4, N)
#P = ppoisson(10, N)

qe_list = [0.1, 0.3, 0.6, 0.9]
z_list  = [0.001, 0.01, 0.45, 0.7]
#z_list  = [0.001, 0.7, 0.3, 0.5]
mo_list = np.arange(4, 15)
mpfids = []
cpfids = []
for z, qe in zip(z_list, qe_list):
    Q = P2Q(P, qe)
    cpfids.append([])
    for MO in mo_list:
        cP = convmrec_pn(Q, qe, z, max_order=MO)
        cpfids[-1].append(fidelity(cP,P))

cpfids = np.array(cpfids)
for f, q, z in zip(cpfids, qe_list, z_list):
    plt.plot(mo_list, f, label=r'$\eta=%g,z=%g$' % (q, z))
plt.legend(frameon=False)
plt.xlabel('Число моментов')
plt.ylabel('Достоверность')
plt.title('Восстановление теплового распределения')
plt.show()