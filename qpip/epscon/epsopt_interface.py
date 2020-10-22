# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:24:30 2019

@author: Pavel Gostev

Interface to
Epsilon-constraint algorithm with arbitrary p
Kirlik, G., & Sayın, S. (2014).
A new algorithm for generating all nondominated solutions
of multiobjective discrete optimization problems.
European Journal of Operational Research, 232(3), 479–488.
doi:10.1016/j.ejor.2013.08.001
"""
from __future__ import division

import numpy as np
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

from ._eps_optimization import iterate
from .._numpy_core import lrange

import logging
logger = logging.getLogger('inverse_eps')
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
info = logger.info
error = logger.error


def epsopt(invpmodel, solver=None,
           eps_tol=0, save_all_nondom_x=False,
           save_all_nondom_y=False, disp=False, plot=False):

    if solver is None:
        solver = SolverFactory('ipopt', solver_io="nl")

    negentropy_bounds = [[-np.log(len(invpmodel.PSET) + 1)], [0]]

    y_vars = ['rel_entropy', 'negentropy']
    x_var = 'P'

    res = iterate(invpmodel, solver, y_vars, x_var,
                  eps_bounds=negentropy_bounds,
                  eps_tol=eps_tol,
                  save_all_nondom_x=save_all_nondom_x,
                  save_all_nondom_y=save_all_nondom_y,
                  disp=disp)
    
    info('Optimization ended in %d iterations with eps_tol %e' %
         (res.nit, eps_tol))
    info(str(res.status))
    print(res)
    
    if plot and res.nit > 1:
        for i in lrange(y_vars):
            plt.plot([e[i] for e in res.ndy], label=y_vars[i])
            plt.legend(frameon=False)
            plt.show()
        if len(y_vars) == 2:
            plt.plot([e[1] for e in res.ndy], [e[0] for e in res.ndy])
            plt.ylabel(y_vars[1])
            plt.xlabel(y_vars[0])
            plt.show()

    return res
