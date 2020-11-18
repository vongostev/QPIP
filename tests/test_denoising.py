# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 01:58:53 2019

@author: vonGostev
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from compose_data import EXP,  ideal_pn_model, N, pnoise, pm_processed, QE, pm_exp

from qpip import g2, mean, fidelity, p_convolve, lrange
from qpip.stat import ppoisson, pthermal
from qpip.denoise import denoiseopt, DenoisePBaseModel
from qpip.denoise.denoise import noisemean_by_g2
from qpip.epscon import InvPBaseModel, invpopt
#from qpip.sipm import d_crosstalk_4n, include_crosstalk_4n
from qpip.stat import qthermal_unpolarized

N = len(pm_processed)
M = N
eps_tol = 0
CT = 0
name = 'exp'
#P_CROSSTALK = 0.00624033520243927
#p0 = d_crosstalk_4n(P_CROSSTALK)[1:]  # ss.hyper_poisson(range(N), 0.6, 4)#
# p0 = ss.correct_thermal(1, M) #ss.hyper_poisson(range(M), 0.6, 4)
p_grid = range(M)
#invpmodel = InvPBaseModel(pm_processed, QE, 25)
#res = invpopt(invpmodel, disp=True, eps_tol=1e-6)
#P = res.x
P = pm_processed
pnoise = qthermal_unpolarized(np.arange(M), noisemean_by_g2(P, 1.5), 1)

ds = []
ms = []
dnmodel = DenoisePBaseModel(P, M, is_zero_n=0)
#for g22 in np.arange(1, 2.5, 0.1):
res = denoiseopt(dnmodel, g2_lbound=1.5, disp=1)
#print(res.y)
ds.append(res.y['discrepancy'])
ms.append(res.y['noise_mean'])
plt.plot(res.x, label=M)
#plt.plot(pnoise)
#plt.legend(frameon=0)
#plt.show()
#plt.plot(np.arange(1,2.5, 0.1), ds)
plt.show()

#plt.plot(np.arange(1,2.5, 0.1), ds)
#plt.show()
    #print(fidelity(pnoise, res.x))
# Отображение распределения шума восстановленного и исходного с ошибкой
plt.bar(lrange(pnoise), pnoise, width=1/2,
        color='#1F618D', label=r'$E_{init}(n)$')
plt.bar(lrange(pnoise), res.x[:len(pnoise)], width=1/4, color='#F39C12',
        label=r'$E_{rec}(n)$' + '\n' + r'$F=%.3f$' % fidelity(pnoise, res.x))

plt.legend(frameon=False, fontsize=14)
plt.xticks(p_grid, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of photons', fontsize=14)
plt.ylabel('Probability', fontsize=14)
#plt.yscale('log')

handles, labels = plt.gca().get_legend_handles_labels()
patch = Line2D([0], [0], color='black', linewidth=1, marker='v',
               linestyle=':', label=r'$\delta E$')
handles.append(patch)
plt.legend(handles=handles, frameon=False, fontsize=12)

ax = plt.twinx()
ax.semilogy(np.abs(pnoise - res.x[:len(pnoise)]),
            'v:', color='black', markersize=8)
ax.set_ylabel(r'$\delta E$', fontsize=14, rotation=0)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.set_label_coords(1.05, 0.95)

plt.tight_layout()

plt.savefig('./img/deconv_%s+p.png' % name, dpi=300)
plt.show()

plt.bar(range(N), P, width=1/2, color='#1F618D', label=r'$P(n)$')
plt.bar(range(N), ppoisson(mean(P), N),
        width=1/4, color='#F39C12',
        label=r'$Poi(n)$' + '\n' + '$F=%.3f$' % fidelity(P, ppoisson(mean(P), N)))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of photons', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.legend(frameon=False, fontsize=14)

plt.tight_layout()

plt.savefig('./img/deconv_%s+ppoi.png' % name, dpi=300)
plt.show()
"""
# Отображение средней ошибки в зависимости от верхней границы среднего
plt.loglog(means, errs, 'D--', color='#1F618D')
#plt.axvline(mean(p0), linestyle=':', color='#F39C12')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(axis='x', which='minor', labelsize=12)
plt.xlabel('Upper mean bound', fontsize=14)
plt.ylabel('Sum square error', fontsize=14)
ax = plt.twinx()
ax.semilogx(means,
            [g2(p) for p in ps_noise], 'D--', color='#C0392B')
#ax.axhline(g2(p0), linestyle=':', color='#F39C12')
#ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.tick_params(axis='both', labelsize=12)
ax.set_ylabel(r'$g_2(0)$', rotation=0, fontsize=14)
plt.tight_layout()

plt.savefig('./img/deconv_%s+upb.png' % name, dpi=300)
plt.show()

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
def g(x, pos): return "${}$".format(f._formatSciNotation('%1.2e' % x))


fmt = mticker.FuncFormatter(g)

MS = 4

# Отображение ошибки поэлементно в зависимости от верхней границы среднего
for i, p in enumerate(ps_noise):
    delta = abs(p[:len(p0)] - p0) / p0
    delta = delta[p[:len(p0)] >= 1e-8][:len(p_grid)]
    plt.semilogy(delta,
                 'D--' if errs[i] == min(errs) else 'x:',
                 label="{}".format(fmt(float(errs[i]))),
                 markersize=(MS + 1 if errs[i] == min(errs) else MS))

leg = plt.legend(loc='upper left', bbox_to_anchor=(0, 1.1),
                 ncol=2, frameon=True, fancybox=False, fontsize=11)
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

plt.xticks(np.arange(0, len(p0), 2), fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Number of photons', fontsize=14)
plt.ylabel(r'$\delta P$', fontsize=14, rotation=0)

plt.tight_layout()

plt.savefig('./img/deconv_%s+err.png' % name, dpi=300)
plt.show()

# Отображение распределения шума восстановленного и исходного с ошибкой
plt.bar(p_grid, p0[:len(p_grid)], width=1/2,
        color='#1F618D', label=r'$E_{init}(n)$')
for i, p in enumerate(ps_noise):
    if errs[i] == min(errs):
        plt.bar(p_grid, p[:len(p_grid)], width=1/4, color='#F39C12',
                label=r'$E_{rec}(n)$' + '\n' + r'$F=%.3f$' % fidelity(p0, p))
        break
plt.legend(frameon=False, fontsize=14)
plt.xticks(p_grid, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of photons', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.yscale('log')

handles, labels = plt.gca().get_legend_handles_labels()
patch = Line2D([0], [0], color='black', linewidth=1, marker='v',
               linestyle=':', label=r'$\delta E$')
handles.append(patch)
plt.legend(handles=handles, frameon=False, fontsize=12)

ax = plt.twinx()
ax.semilogy(abs(p[:len(p_grid)] - p0[:len(p_grid)]),
            'v:', color='black', markersize=8)
ax.set_ylabel(r'$\delta E$', fontsize=14, rotation=0)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.set_label_coords(1.05, 0.95)

plt.tight_layout()

plt.savefig('./img/deconv_%s+p.png' % name, dpi=300)
plt.show()

# Отображение распределения восстановленного и исходного
if CT:
    pm_rec2 = include_crosstalk_4n(ppoisson(
        mean(P) - mean(p0), len(P)), P_CROSSTALK)

    fig = plt.figure(figsize=(9, 4))
else:
    fig = plt.figure(figsize=(4, 3))

#p = ps_noise[-2]
pm_rec = p_convolve(ppoisson(
    mean(P) - mean(p), len(P)), p)
plt.bar(range(P.size), P, width=1/2,
        color='#1F618D', label=r'$q_{exp}(n)$')
plt.bar(range(P.size), pm_rec, width=1/4, color='#F39C12',
        label=r'$q_{rec}(n)$' + '\n' + r'$F=%.3f$' % fidelity(p0, p))


handles, labels = plt.gca().get_legend_handles_labels()

if CT:
    plt.semilogy(pm_rec2, 'o', color='black',
                 markerfacecolor='white', markeredgewidth=1.5,
                 label=r'$q_{rec}^{4n}(n)$')
    patch = Line2D([0], [0], color='black', linewidth=1, marker='v',
                   linestyle=':', label=r'$\delta q_{4n}$')
    handles.append(patch)

patch = Line2D([0], [0], color='black', linewidth=1, marker='^',
               linestyle=':', label=r'$\delta q_{opt}$')
handles.append(patch)
if CT:
    leg = plt.legend(handles=handles, frameon=True, fancybox=False,
                     fontsize=12, bbox_to_anchor=(1.4, 1))
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
else:
    plt.legend(handles=handles, frameon=False, fancybox=False,
               fontsize=12)

plt.xticks(range(P.size), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of photocounts', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.yscale('log')

ax = plt.twinx()
ax.semilogy(abs(pm_rec - P), '^:',
            color='black', markersize=7)
if CT:
    ax.semilogy(abs(pm_rec2 - P), 'v:',
                color='black', markersize=7)
ax.set_ylabel(r'$\delta q$', fontsize=14, rotation=0)
ax.tick_params(axis='both', labelsize=12)
ax.yaxis.set_label_coords(1.05, 0.95)
ax.set_ylim(1e-6, 1e-1)
plt.tight_layout()

plt.savefig('./img/deconv_%s+ct_test.png' % name, dpi=300)
plt.show()

# Сравнение с Пуассоном
plt.bar(range(N), P, width=1/2, color='#1F618D', label=r'$P(n)$')
plt.bar(range(N), ppoisson(mean(P) - mean(p), N),
        width=1/4, color='#F39C12',         label=r'$Poi(n)$')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Number of photons', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.yscale('log')
plt.legend(frameon=False, fontsize=14)

plt.tight_layout()

plt.savefig('./img/deconv_%s+ppoi.png' % name, dpi=300)
plt.show()
"""