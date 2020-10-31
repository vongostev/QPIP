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

from ._eps_optimization import iterate, info


def invpopt(invpmodel, solver=None,
            eps_tol=0, save_all_nondom_x=False,
            save_all_nondom_y=False, disp=False, plot=False):
    """
    Interface to epsilon-constrained algorithm
    Made to solve photocounting inverse problem

    Parameters
    ----------
    invpmodel : pyomo.environ.ConcreteModel
        The pyomo model described a problem.
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
            'iteration number' 'status (1 is success, 0 is fail)' ['rel_entropy', 'negentropy']
        The default is False.
    plot : bool, optional
        Plot dependency of 'rel_entropy' and 'negentropy' from an iteration number
        after the end of calculations.
        The default is False.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        see help(qpip.epscon.iterate).

    """

    if solver is None:
        solver = SolverFactory('ipopt', solver_io="nl")

    negentropy_bounds = [[-np.log(len(invpmodel.PSET))], [0]]

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

    if plot and res.nit > 1:
        plt.plot([e[1] for e in res.ndy], [e[0] for e in res.ndy])
        plt.ylabel(y_vars[1])
        plt.xlabel(y_vars[0])
        plt.show()

    return res
