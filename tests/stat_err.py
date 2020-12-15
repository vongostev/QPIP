# -*- coding: utf-8 -*-
"""
Created on nmaxon Sep  7 19:36:20 2020

@author: von.gostev
"""
import os
import sys
sys.path.append(os.path.abspath('..'))

from lstat_core.sc_moments import *
import lstat_core.detection_core as dc
import lstat_core._sympy_statistics as ss
from labellines import labelLines
import matplotlib.pyplot as plt
import numpy as np


# from scipy.special import factorial
# from sympy.functions.combinatorial.numbers import stirling
# from decimal import Decimal

# import matplotlib.gridspec as gridspec
# from matplotlib.ticker import FuncFormatter


# from lstat_core.detection_matrix import binomial_t_matrix
from lstat_core.qutip_statistics import squeezed_vacuumM
# from lstat_core.sc_recurrent import get_photon_distribution


DPREC = np.float128


qe = 0.6
mmax = 30
n0max = mmax
nmax = mmax
K = 6
N = int(1E7)
Kmax = 20

P = dc.normalize(ss.correct_poisson(6, n0max, dtype=DPREC) + ss.correct_poisson(2.7, n0max, dtype=DPREC))
# P = precarray(ss.thermal_photonsub(n0max, 3,1))
# P = ss.correct_thermal(3.2, nmax, dtype=DPREC)
# P = dc.p_convolve(ss.correct_thermal(1, nmax, dtype=DPREC), 
#                  ss.correct_poisson(3, nmax, dtype=DPREC))
# P = ss.correct_poisson(4, n0max)
# P = squeezed_vacuumM(mmax, 2, 1.3, 0)
# P = np.zeros(n0max)
# P[1:5] = 1/4
# P = ss.correct_poisson(8, n0max)
Q = dc.normalize(dc.get_pm(P, qe, mmax))
Q1E = np.random.choice(np.arange(mmax), size=N, p=Q.astype(np.float64))
Q1, _ = np.histogram(Q1E, bins=np.arange(mmax), density=True)
"""
Fm = []
dFm = []
for i in range(10):
    Fm.append([])
    dFm.append([])
    # N = 10**i
    qe = (i + 1) * 0.1
    for k in range(2, Kmax):
        f = []
        for l in range(30):
            Q = dc.normalize(dc.get_pm(P, qe, mmax).astype(np.float64))
            Q1E = np.random.choice(np.arange(mmax), size=N, p=Q)
            Q1, _ = np.histogram(Q1E, bins=np.arange(mmax), density=True)

            P2 = bmrec_pn(Q1, qe, nmax, k)
            f.append(dc.fidelity(P, P2))
        Fm[-1].append(np.mean(f))
        dFm[-1].append(np.std(f))


for i, p in enumerate(Fm):
    plt.errorbar(range(2, Kmax), p, yerr=dFm[i], label='%.1f' % ((i + 1) * 0.1),
    capsize=5)
plt.yscale('log')
plt.legend(title='p', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.xlabel("K")
plt.xticks(np.arange(2, Kmax, 2))
plt.ylabel(r"$F(P,P')$", rotation=0)
plt.ylim((0.1, 10))
plt.show()

F = np.log10(np.abs(1 - np.array(Fm)))
mdata = plt.contourf(np.arange(2, Kmax), np.arange(1,11)*0.1, F, levels=50)
plt.colorbar()

mcontours = plt.contour(mdata, levels=[-2, -3, -4],
            colors='w')
plt.clabel(mcontours, inline=1, fmt={-4: r'$10^{-4}$', -3: r'$10^{-3}$', -2: r'$10^{-2}$'}, fontsize=10)
plt.title(r"$\log_{10}(1 - F(P,P'))$")
plt.xlabel("K")
plt.ylabel("p", rotation=0)
plt.show()
"""
"""
Nrange = 10**np.arange(3, 9, 1)

Fm = []
dFm = []

Fi = []
dFi = []

Q = dc.normalize(dc.get_pm(P, qe, mmax).astype(np.float64))

for N in Nrange:
    fm = []
    fi = []
    for l in range(50):
        Q1E = np.random.choice(np.arange(mmax), size=N, p=Q)
        Q1, _ = np.histogram(Q1E, bins=np.arange(mmax), density=True)

        P1 = dc.get_pn(Q1, qe, nmax)
        P2 = bmrec_pn(Q1, qe, nmax, K)

        fi.append(dc.fidelity(P, P1))
        fm.append(dc.fidelity(P, P2))

    Fm.append(np.mean(fm))
    dFm.append(np.std(fm))
    Fi.append(np.mean(fi))
    dFi.append(np.std(fi))


plt.errorbar(Nrange, Fi, yerr=dFi, label='Inverse matrix',
             capsize=5)
plt.errorbar(Nrange, Fm, yerr=dFm, label='nmaxethod of moments',
             capsize=5)
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='center left', bbox_to_anchor=(0.6, -0.15), frameon=False, fontsize=10)
plt.xlabel("N")
# plt.xticks(np.arange(2, Kmax, 2))
plt.ylabel(r"$F(P,P')$", rotation=0)
# plt.ylim((0.1, 10))
plt.title("Composite Poisson 2.7 + 13.4")
plt.show()
"""
