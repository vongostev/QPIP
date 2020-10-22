# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:38:48 2020

@author: vonGostev
"""

import numpy as np
import matplotlib.pyplot as plt

import test_maxent_algorithm as ta

# ['composite_poisson', 'hyper_poisson', 'thermal', 'poisson']
dist_types = ['squezeed_vacuum']
qes = [0.32, 0.5, 0.75]
M = dict(zip(qes, [10, 15, 20]))


def execute_test(t, qe):
    ta.parameters_dict['type'] = t
    ta.parameters_dict['QE'] = qe
    ta.parameters_dict['M'] = M[qe]
    results_dict = ta.run()
    return [float(x) for x in results_dict['F[pn][rec, ideal]']]


data_by_qe = {}
for qe in qes:
    data_by_type = {}
    for t in dist_types:
        data_by_type[t] = [execute_test(t, qe) for n in range(10)]
    data_by_qe[qe] = data_by_type
    ta.info(qe)
    ta.info('\n'.join([k + ' ' + str(v) for k, v in data_by_type.items()]))


types = list(data_by_qe[0.32].keys())
data_by_type = {t: {qe: d[t] for qe, d in data_by_qe.items()} for t in types}
MS = 2
CS = 5

for t, data_qe in data_by_type.items():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for qe, d in data_qe.items():
        if len(d[0]) == 0:
            continue
        ax.errorbar(ta.parameters_dict['ERR'], np.mean(1 - np.array(d), axis=0), yerr=np.std(1 - np.array(d), axis=0),
                    linestyle="--", marker='D', markersize=MS, capsize=CS, label='%.2f' % qe)
        if 1 - np.min(d) > 1e-2:
            plt.axhline(1e-2, linestyle=':', color='black', linewidth=1)

    ax.yaxis.set_label_coords(-0.07, 0.915)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative error in %', fontsize=16)
    plt.ylabel(r'$1-F$', rotation=0, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    #plt.ylim(1e-6, 1e-1)
    leg = plt.legend(frameon=False, fontsize=12, ncol=3,
                     loc='upper left')
    leg.set_title(' '.join(t.split('_')).capitalize() +
                  ' distribution', prop={'size': 12})
    leg._legend_box.align = "left"
    plt.tight_layout()
    plt.savefig('./img/%s_maxent_f_by_err.png' % t, dpi=300)
    plt.show()


for t, data_qe in data_by_type.items():
    for qe, d in data_qe.items():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if len(d[0]) == 0:
            continue
        ax.boxplot(1 - np.array(d), showfliers=0,
                   labels=ta.parameters_dict['ERR'], whis=[5, 95])
        if 1 - np.max(d) < 1e-2:
            plt.axhline(1e-2, linestyle=':', color='black', linewidth=1)

        ax.yaxis.set_label_coords(-0.07, 0.915)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Relative error in %', fontsize=16)
        plt.ylabel(r'$1-F$', rotation=0, fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        leg = plt.legend(frameon=False, fontsize=12, ncol=3,
                         loc='upper left')
        leg.set_title(' '.join(t.split('_')).capitalize() +
                      ' distribution' + ' %.2f' % qe, prop={'size': 12})
        leg._legend_box.align = "left"
        plt.tight_layout()
        #plt.savefig('./img/maxent_%s_f_by_err.png' % t, dpi=300)
        plt.show()


"""
for qe, data_by_type in data_by_qe.items():
    MS = 2
    CS = 3    
    for t, data in data_by_type.items():
        plt.errorbar(ta.parameters_dict['ERR'], 1 - np.mean(data, axis=0), yerr=np.std(data, axis=0),
                     linestyle="--", marker='D', markersize=MS, capsize=CS, label=t)
    
    plt.axhline(1e-2, linestyle=':', color='black', linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative error in %')
    plt.ylabel(r'$1-F$', rotation=0)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig('./img/maxent_errorbar_f_by_err_%f.png' % qe, dpi=300)
    plt.show()

    for t, data in data_by_type.items():
        plt.plot(ta.parameters_dict['ERR'], 1 - np.mean(data, axis=0), 
             linestyle='--', marker='D', markersize=MS, label=t)
    
    plt.axhline(1e-2, linestyle=':', color='black', linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative error in %')
    plt.ylabel(r'$1-F$', rotation=0)
    plt.legend(frameon=False, fontsize=10)
    plt.savefig('./img/maxent_plot_f_by_err_%f.png' % qe, dpi=300)
    plt.show()
    """
