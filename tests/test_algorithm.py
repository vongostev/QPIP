# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:49:20 2019

@author: vonGostev
"""
import os
import sys
sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt

from qpip import compose as cmp
from qpip import lrange, normalize, mean, g2, fidelity
from qpip import P2Q, Q2P
from qpip.epscon import epsopt, InvPBaseModel
from qpip.stat import *

import numpy as np
import logging

import datetime
import copy


FLOG = True
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()

if FLOG:
    fh = logging.FileHandler(
        "./log/test_data %s.log" % datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    optimizelog = logging.getLogger('sc_pyomo')
    optimizelog.addHandler(fh)

pyomolog = logging.getLogger('pyomo.core')
pyomolog.setLevel(logging.ERROR)

info = logger.info

dictzip = cmp(dict, zip)


def make_model_pn(P, N, N0):

    return 


def make_qmodel(P, mtype, n_cells, qe=0, M=0, N0=0, ERR=0, **kwargs):
    Q = P2Q(P, qe, M, mtype, n_cells)
    QND = np.random.choice(range(M), size=N0, p=Q.astype(float))
    QN = np.histogram(QND, bins=range(M + 1), density=True)[0]
    if ERR:
        QN = cmp(normalize, np.abs)(QN*(1 + np.random.uniform(-ERR, ERR, size=len(QN))))
    return Q, QN


def execute_test(QN, Q, P, qe, N, mtype, n_cells, eps_tol):
    invpmodel = InvPBaseModel(QN, qe, N, mtype=mtype, n_cells=n_cells)
    res = epsopt(invpmodel, eps_tol=eps_tol)
    PREC, QREC = res.x, P2Q(res.x, qe, len(Q))
    F_pn = fidelity(PREC, P)
    F_pn_rn = fidelity(PREC, P)
    F_pm = fidelity(QN, QREC)
    F_pm_ideal = fidelity(Q, QREC)
    info('RI>> F[P] %f' % F_pn)
    info('RN>> F[P] %f' % F_pn_rn)
    info('RN>> F[QN] %f' % F_pm)
    info('RI>> F[QN] %f' % F_pm_ideal)
    return F_pn, F_pn_rn, F_pm, F_pm_ideal


def parse_parameters(parameters_set, template, var_label, fix_strategies):
    parameters_dict = dictzip(template, parameters_set)
    for var, fun in fix_strategies.items():
        if var != var_label:
            parameters_dict[var] = fun(parameters_dict[var])
    return parameters_dict


def iterate_parameters(parameters_dict, var_label):
    var_list = parameters_dict[var_label]
    _d = copy.deepcopy(parameters_dict)
    for parameter in var_list:
        _d[var_label] = parameter
        yield _d


def plot_results(parameters_dict, var_label, results_dict, xlog=True):
    X = parameters_dict[var_label]
    plt.xlabel(var_label)
    plt.xticks(X)

    plt.ylabel('F', rotation=False)

    ftitle = ' '.join([k + ' ' + str(v)
                       for k, v in parameters_dict.items() if k != var_label])
    plt.title('F by %s ' % var_label + '[' + ftitle + ']\n')

    for label, data in results_dict.items():
        if xlog:
            plt.semilogx(X, data, 'o--', label=label,
                         markersize=5)
        else:
            plt.plot(X, data, 'o--', label=label,
                     markersize=5)

    plt.legend(frameon=False, fontsize=12)
    plt.savefig(fname='./img/%s test %s.png' %
                (datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"), ftitle),
                dpi=300, bbox_inches='tight')
    plt.show()


N = 25
MTYPE = 'binomial'
N_CELLS = 667

TEST_PARAMETERS_TEMPLATE = ['type', 'N', 'N0', 'M', 'qe', 'ERR', 'eps_tol']
XLOG_TEMPLATE = [False, False, True, False, False, True, True]
XLOG = dictzip(TEST_PARAMETERS_TEMPLATE, XLOG_TEMPLATE)
PN_DISTRIBUTIONS = {
    # 'hyper_poisson': ss.hyper_poisson(range(N), 9, 6.5),
    'composite_poisson': normalize(
        ppoisson(9, N) + ppoisson(1, N)),
    'thermal': pthermal(3, N),
    'poisson': ppoisson(6, N),
    'squezeed_vacuum': psqueezed_vacuumM(N, 2, 1.4, 0),
    'squezeed': psqueezed_coherent1(N, 3, 0.3+0.3j),
}

dist_type = 'poisson'

TEST_PARAMETERS_SET = [
    dist_type,
    N,
    [int(1E5), int(1E6), int(1E7), int(1E8)],
    [10, 11, 12, 13, 14, 15],
    [0.32, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    [1e-3],#[1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 1e-2, 1e-1],
    [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
]
STEP_RESULTS_TEMPLATE = [
    'F[P][rec, ideal]',
    'F[P][rec, noised]',
    'F[QN][rec, noised]',
    'F[QN][rec, ideal]'
]
results_lists = [[] for i in lrange(STEP_RESULTS_TEMPLATE)]

var_label = 'ERR'
fix_strategies = {
    'type': lambda x: x,
    'N': lambda x: x,
    'N0': lambda x: x[1],
    'M': lambda x: 25,
    'qe': lambda x: 0.5,
    'ERR': lambda x: x[-2],
    'eps_tol': lambda x: x[2]
}


parameters_dict = parse_parameters(TEST_PARAMETERS_SET,
                                   TEST_PARAMETERS_TEMPLATE,
                                   var_label, fix_strategies)


def run():
    results_lists = [[] for i in lrange(STEP_RESULTS_TEMPLATE)]

    P = PN_DISTRIBUTIONS[parameters_dict['type']]

    for parameters in iterate_parameters(parameters_dict, var_label):
        Q, QN = make_qmodel(P, MTYPE, N_CELLS, **parameters)
        info('<QN> %f g2(QN) %f' % (mean(QN), g2(QN)))
        info('<Q> %f g2(Q) %f' %
             (mean(Q), g2(Q)))
        info('<P> %f g2(P) %f' %
             (mean(P), g2(P)))
        info(str(parameters))
        step_results = execute_test(QN, Q, P,
                                    parameters['qe'],
                                    parameters['N'],
                                    MTYPE, N_CELLS,
                                    parameters['eps_tol'])
        for i, e in enumerate(step_results):
            results_lists[i].append(e)

    results_dict = dictzip(STEP_RESULTS_TEMPLATE, results_lists)
    plot_results(parameters_dict, var_label,
                 results_dict, xlog=XLOG[var_label])
    info('\n'.join([k + ' ' + str(v) for k, v in results_dict.items()]))

    return results_dict


if __name__ == "__main__":
    results_dict = run()