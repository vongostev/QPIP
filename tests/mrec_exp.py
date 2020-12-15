#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:46:23 2020

@author: vong
"""
import __init__
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../..'))

from oscsipm.oscsipm import hist2Q, QStatisticsMaker, compensate
from qpip import g2, normalize, lrange
import matplotlib.pyplot as plt
from qpip.moms import convmrec_pn, covm_mltnomial, covm_conv_pn

qm = QStatisticsMaker("15122020/01_2kHz_0.83V_ampl9V_pinhole.csv", 0.002, plot=1, peak_width=1.1, method='fit')

sigma = np.diag(covm_mltnomial(qm.getq(), sum(qm.hist))) ** 0.5
q0 = normalize(qm.getq()[:5])
plt.errorbar(lrange(q0), q0, yerr=3 * sigma[:len(q0)], capsize=5, label="g2(Q) = %.2f" % g2(q0))

# q = compensate(q0, 0.026)
# sigma = np.diag(covm_mltnomial(q, sum(qm.hist))) ** 0.5
# plt.errorbar(lrange(q), q, yerr=3 * sigma[:len(q)], capsize=5)

p = convmrec_pn(q0, 0.55, 0.1, 10, 5)
sigmap = np.diag(covm_conv_pn(q0, 0.55, 0.1, sum(qm.hist), 10, 5)) ** 0.5
plt.errorbar(lrange(p), p, yerr=3 * sigmap[:len(p)], capsize=5, label="g2(P) = %.2f" % g2(p))
plt.legend(frameon=0)
plt.xlabel('Number of photons / photocounts')
plt.ylabel('Probability')
plt.show()

qmt = QStatisticsMaker("15122020/01_2kHz_0.83V_ampl9V_pinhole_disk5.csv", 0.002, plot=1, peak_width=1.1, method='fit')

sigma = np.diag(covm_mltnomial(qmt.getq(), sum(qmt.hist))) ** 0.5
q0 = normalize(qmt.getq()[:5])
plt.errorbar(lrange(q0), q0, yerr=3 * sigma[:len(q0)], capsize=5, label="g2(Q) = %.2f" % g2(q0))

# q = compensate(q0, 0.026)
# sigma = np.diag(covm_mltnomial(q, sum(qm.hist))) ** 0.5
# plt.errorbar(lrange(q), q, yerr=3 * sigma[:len(q)], capsize=5)

p = convmrec_pn(q0, 0.55, 0.1, 10, 5)
sigmap = np.diag(covm_conv_pn(q0, 0.55, 0.1, sum(qmt.hist), 10, 5)) ** 0.5
plt.errorbar(lrange(p), p, yerr=3 * sigmap[:len(p)], capsize=5, label="g2(P) = %.2f" % g2(p))
plt.legend(frameon=0)
plt.xlabel('Number of photons / photocounts')
plt.ylabel('Probability')
plt.show()

qmp = QStatisticsMaker("15122020/02_2kHz_0.83V_ampl9V_pinhole_disk5_3.csv", 0.002, plot=1, peak_width=1.1, method='fit')

sigma = np.diag(covm_mltnomial(qmp.getq(), sum(qmp.hist))) ** 0.5
q0 = normalize(qmp.getq()[:10])
plt.errorbar(lrange(q0), q0, yerr=3 * sigma[:len(q0)], capsize=5, label="g2(Q) = %.2f" % g2(q0))

# q = compensate(q0, 0.026)
# sigma = np.diag(covm_mltnomial(q, sum(qm.hist))) ** 0.5
# plt.errorbar(lrange(q), q, yerr=3 * sigma[:len(q)], capsize=5)

p = convmrec_pn(q0, 0.55, 0.2, 20, 4)
sigmap = np.diag(covm_conv_pn(q0, 0.55, 0.2, sum(qmp.hist), 20, 4)) ** 0.5
plt.errorbar(lrange(p), p, yerr=3 * sigmap[:len(p)], capsize=5, label="g2(P) = %.2f" % g2(p))
plt.legend(frameon=0)
plt.xlabel('Number of photons / photocounts')
plt.ylabel('Probability')
plt.show()
