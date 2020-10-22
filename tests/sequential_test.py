# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:07:15 2019

@author: vonGostev
"""
import numpy as np
import matplotlib.pyplot as plt

import test_algorithm as ta

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
    plt.savefig('./img/errorbar_f_by_err_%f.png' % qe, dpi=300)
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
    plt.savefig('./img/plot_f_by_err_%f.png' % qe, dpi=300)
    plt.show()
