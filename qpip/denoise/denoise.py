# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:15:10 2019

@author: Pavel Gostev
"""
import numpy as np
from pyomo.opt import SolverFactory
from scipy.optimize import brute
import matplotlib.pyplot as plt

from ..epscon._eps_optimization import iterate, info
from .._numpy_core import lrange, p_convolve, mean, fidelity
from ..stat import ppoisson, pthermal


def denoiseopt(dnmodel, mean_lbound=0, mean_ubound=1,
               g2_lbound=1, g2_ubound=50, eps_tol=0,
               save_all_nondom_x=False, plot=False,
               save_all_nondom_y=False, disp=False):
    solver = SolverFactory('ipopt', solver_io="nl")

    y_vars = ['disprepancy', 'noise_mean', 'negentropy', 'noise_g2']
    x_var = 'p_noise'

    noise_bounds = [[mean_lbound, -np.log(len(dnmodel.NSET)), g2_lbound],
                    [mean_ubound, 0, g2_ubound]]

    res = iterate(dnmodel, solver, y_vars, x_var,
                  eps_bounds=noise_bounds,
                  eps_tol=eps_tol,
                  save_all_nondom_x=save_all_nondom_x,
                  save_all_nondom_y=save_all_nondom_y,
                  disp=disp)

    info('Optimization ended in %d iterations with eps_tol %e' %
         (res.nit, eps_tol))
    info(str(res.status))

    if plot and res.nit > 1:
        for i in lrange(y_vars):
            plt.plot([e[i] for e in res.ndy], label=y_vars[i])
            plt.legend(frameon=False)
            plt.show()

    return res


def predict_nmean(P):
    N = len(P)
    opt = lambda x: 1 - fidelity(p_convolve(ppoisson(x[0], N), pthermal(x[1], N)), P)
    res = brute(opt, [(0, mean(P)), (0.0, mean(P))])
    smean, nmean = res
    return nmean
    
    