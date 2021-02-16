#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:46:23 2020

@author: vong
"""
import __init__

from qpip.moms import Q2PCM, covm_mltnomial, covm_conv_pn
import matplotlib.pyplot as plt
from qpip import g2, normalize, lrange, Q2P
from oscsipm.oscsipm import QStatisticsMaker, compensate, optimize_pcrosstalk, Q2total_pcrosstalk, single_pcrosstalk

import numpy as np


plt.style.use('bmh')


def makeq(path, qe, pct, N, max_order, discrete=0.017):
    qm = QStatisticsMaker(path, discrete, plot=1, peak_width=1, method='fit',
                          remove_pedestal=0)
    return qm.hist, 0, 0

    q0 = normalize(qm.getq())
    qsigma0 = np.diag(covm_mltnomial(q0, sum(qm.hist))) ** 0.5
    q = compensate(q0, pct)
    qsigma = np.diag(covm_mltnomial(q, sum(qm.hist))) ** 0.5

    p, zopt = Q2PCM(q, qe, N, max_order)
    psigma = np.diag(covm_conv_pn(
        q, qe, zopt, sum(qm.hist), N, max_order)) ** 0.5

    plt.errorbar(lrange(q0), q0, yerr=3 * qsigma0[:len(q0)], capsize=5,
                 label=r"$g_2(Q_{exp}) = %.2f$, $N=%d$" % (g2(q0), sum(qm.hist)))
    plt.errorbar(lrange(q), q, yerr=3 * qsigma[:len(q)], capsize=5,
                 label=r"$g_2(Q_{dn}) = %.2f$" % g2(q))
    plt.errorbar(lrange(p), p, yerr=3 * psigma[:len(p)], capsize=5,
                 label=r"$g_2(P) = %.2f$, $K=%d$" % (g2(p), max_order))
    #plt.plot(Q2P(q0, qe))
    plt.legend(frameon=0, fontsize=14)
    plt.xlabel('Number of photons / photocounts', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    # plt.xscale('symlog')
    plt.show()
    print(sum(qm.hist), zopt)
    #p0 = Q2PCMO(q, qe, N, max_order=max_order, z=zopt)
    return q0, q, p


qe = 0.44 #0.46
# Crosstalk probability
# Based on data from 16122020/01_5kHz_0.81V_ampl10V.csv
# Total crosstalk probability 0.05828426959386601
pct =  0.01490082301594439#single_pcrosstalk(0.07884047241411873)#0.01490082301594439
N = 11
q0, q, p = makeq("03022021/210203_192751.csv", qe, pct, 19, 5, discrete=1e-10)

#q0, q, p = makeq("28012021/210128_185401.csv", qe, pct, 18, 5, discrete=0.0017)

#q0, q, p = makeq("21012021/020101_051951.csv", qe, pct, 15, 5, discrete=0.0017)
#q0, q, p = makeq("21012021/020101_070414.csv", qe, pct, 12, 5, discrete=0.0017)

#q0, q, p = makeq("25122020/201225_220301.csv", qe, pct, 13, 5, discrete=0.0021)
#q0, q, p = makeq("28122020/201228_220459.csv", qe, pct, 21, 6, discrete=0.0024)
#q0, q, p = makeq("16122020/01_5kHz_0.81V_ampl10V_diskstop.csv", qe, pct, 13, 6)
#q0, q, p = makeq("16122020/01_5kHz_0.81V_ampl10V.csv", qe, pct, 15, 5)
#q0, q, p = makeq("16122020/01_5kHz_0.81V_ampl10V_disk5_V.csv", qe, pct, 11, 6)
#q0, q, p = makeq("16122020/01_5kHz_0.81V_ampl10V_disk5_H2.csv", qe, pct, 11, 6)
