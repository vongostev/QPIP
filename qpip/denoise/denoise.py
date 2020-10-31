# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:15:10 2019

@author: Pavel Gostev
"""
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.environ import Constraint
from scipy.optimize import brute
import matplotlib.pyplot as plt

from ..epscon._eps_optimization import iterate, info
from .._numpy_core import lrange, p_convolve, mean, fidelity
from ..stat import ppoisson, pthermal


def denoiseopt(dnmodel, mean_lbound=0, mean_ubound=1,
               g2_lbound=1.5, g2_ubound=50, eps_tol=0,
               solver=None,
               save_all_nondom_x=False, plot=False,
               save_all_nondom_y=False, disp=False):
    """
    Interface to epsilon-constrained algorithm
    Made to denoise photon-number or photounting statistics
    from the coherent background.
    It is usable for recovering spectrum of fluctuations, excess noise or
    single-photon signal.

    Parameters
    ----------
    dnmodel : qpip.denoise.DenoisePBaseModel
        A model of the problem to solve.
    mean_lbound : float, optional
        Minimal possible value of noise's mean. The default is 0.
    mean_ubound : float, optional
        Maximal possible value of noise's mean. The default is 1.
    g2_lbound : float, optional
        Minimal possible value of noise's g2. The default is 1.5.
    g2_ubound : float, optional
        Maximal possible value of noise's g2. The default is 50.
    solver : pyomo.opt.SolverFactory, optional
        Solver to solve the problem.
        The default is ipopt if installed.
    eps_tol : float, optional
        Tolarance of additional variables to finish iterations.
        The default is 0.
    save_all_nondom_x : bool, optional
        Save all nondominate solutions in output.
        The default is False.
    save_all_nondom_y : bool, optional
        Save y vector values for all nondominate solutions.
        The default is False.
    disp : bool, optional
        Display calculation progress.
        Messages have following format:
            'iteration number' 'status (1 is success, 0 is fail)' ['disprepancy', 'noise_mean', 'negentropy', 'noise_g2']
        The default is False.
    plot : bool, optional
        Plot dependency of vars from an iteration number
        after the end of calculations.
        The default is False.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        see help(qpip.epscon.iterate).

    """

    if solver is None:
        solver = SolverFactory('ipopt', solver_io="nl")

    y_vars = ['disprepancy', 'noise_mean', 'negentropy', 'noise_g2']
    x_var = 'p_noise'

    dnmodel.c_low_g2 = Constraint(expr=dnmodel.noise_g2 >= g2_lbound)
    dnmodel.c_top_g2 = Constraint(expr=dnmodel.noise_g2 <= g2_ubound)
    dnmodel.c_top_mean = Constraint(expr=dnmodel.noise_mean <= mean_ubound)
    dnmodel.c_low_mean = Constraint(expr=dnmodel.noise_mean >= mean_lbound)

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
    res = brute(opt, [(0, mean(P)), (0.0, mean(P))], Ns=100, full_output=True)
    smean, nmean = res[0]
    fval = res[1]
    info("Optimum thermal noise mean found with disprepancy {0}".format(fval))
    return nmean
