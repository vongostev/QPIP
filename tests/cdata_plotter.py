# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:02:38 2019

@author: vonGostev
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np

from qpip import Q2P


def pre_plot(pn_rec, pm_processed):
    plt.ylabel(r'$P$' + '\t', rotation=0, fontsize=14)

    plt.bar(np.arange(len(pn_rec)) + 1/4, pn_rec, width=1/2, zorder=3,
            label=r'$P_{rec}$',
            facecolor='#C0392B', alpha=0.9)


def plot_exp_data(pm_processed,
                  pn_rec,
                  pm_rec,
                  M, DISPLAY_ERR=False):
    pre_plot(pn_rec, pm_processed)
    plt.xlabel('Number of photons / photocounts')

    plt.bar(np.arange(pm_processed.size), pm_processed,
            width=1/2, zorder=4,
            facecolor='#1F618D',
            label=r'$q_0$', alpha=0.9)
    plt.plot(pm_rec, 'o', ms=4,
             markerfacecolor='white', markeredgewidth=1.5,
             markeredgecolor='#F39C12', zorder=5,
             label=r'$q_{rec}$')

    handles, labels = plt.gca().get_legend_handles_labels()

    if DISPLAY_ERR:
        ERR_CLR = 'black'
        delta = abs(pm_processed[:M] - pm_rec)[pm_processed != 0]

        ax = plt.twinx()
        ax.semilogy(delta,
                    'x:', linewidth=1, color=ERR_CLR, label=r'$\delta(q)$')
        ax.spines['right'].set_color(ERR_CLR)
        ax.yaxis.label.set_color(ERR_CLR)
        ax.tick_params(axis='y', colors=ERR_CLR)

        ax.set_ylabel(r'$\delta(q)$', rotation=0)
        ax.set_ylim((10**-5, 5 * 10**-3))
        # ax.legend(frameon=False, loc=1, fontsize=10,
        #          bbox_to_anchor=(0.97, 0.62))

        # manually define a new patch
        patch = Line2D([0], [0], color=ERR_CLR, linewidth=1, marker='x',
                       linestyle=':', label=r'$\delta(q)$')

        # handles is a list, so append manual patch
        handles.append(patch)
    plt.legend(handles=handles, frameon=False, fontsize=10)

    plt.savefig('./img/rec_exp_pm_poisson.png', dpi=300)
    plt.show()


def plot_model_data(pm_processed,
                    pn_rec, pn_model,
                    pm_rec,
                    M, QE, DISPLAY_ERR=False):
    pre_plot(pn_rec, pm_processed)
    plt.xlabel('Number of photons, n', fontsize=12)
    plt.tick_params(labelsize=12)
    N = len(pn_model)

    ph_analytic = Q2P(pm_processed, QE, N)
    plt.bar(np.arange(pn_model.size), pn_model, width=1/2,
            color='#1F618D', label=r'$P_{model}$')
    handles, labels = plt.gca().get_legend_handles_labels()

    ERR_CLR = 'black'

    ax = plt.twinx()
    ax.semilogy(
        [abs(pn_model - pn_rec)[i] for i in range(N)],
        'x:', color=ERR_CLR, label=r'$\delta(P)$', linewidth=1)
    if DISPLAY_ERR:
        ax.semilogy(abs(pn_model - ph_analytic),
                    '+', color='#FFA500', label='Analytical error')
    ax.spines['right'].set_color(ERR_CLR)
    ax.yaxis.label.set_color(ERR_CLR)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_ylabel(r'$\delta(P)$', rotation=0, fontsize=14)

    patch = Line2D([0], [0], color=ERR_CLR, linewidth=1, marker='x',
                   linestyle=':', label=r'$\delta(P)$')
    # handles is a list, so append manual patch
    handles.append(patch)
    plt.legend(handles=handles, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig('./img/rec_model.png', dpi=300)
    plt.show()
