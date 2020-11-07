# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:15:10 2019

@author: Pavel Gostev
"""
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.environ import Constraint
from scipy.optimize import brute, OptimizeResult
import matplotlib.pyplot as plt

from ..epscon._eps_optimization import iterate, info
from .._numpy_core import lrange, p_convolve, mean, fidelity, g2
from ..stat import ppoisson, pthermal

def warn(*args): return  np.warnings.warn_explicit(*args)

def thermalnoise_mean(P):
    """
    Calculate mean value of the noise thermal distribution.
    Convolution of this distribution and a poisson distributiob is
    optimal in the sense of minimal discrepancy

    Parameters
    ----------
    P : array_like
        Noised probability distribution.

    Returns
    -------
    nmean : float
        Mean value of the thermal distribution.

    """

    N = len(P)
    opt = lambda x: 1 - fidelity(p_convolve(ppoisson(x[0], N), pthermal(x[1], N)), P)
    res = brute(opt, [(0, mean(P)), (0.0, mean(P))], Ns=100, full_output=True)
    smean, nmean = res[0]
    fval = res[1]
    info("Optimum thermal noise mean found with discrepancy {0}".format(fval))
    return nmean


def noisemean_by_g2(P, noise_g2):
    """
    Calculate mean value of the noise distribution for given noised distribution
    and :math:`g^{(2)}` value of the noise distribution.

    Parameters
    ----------
    P : array_like
        Noised probability distribution.
    noise_g2 : float
        Expected :math:`g^{(2)}` value of the noise distribution.

    Returns
    -------
    mean
        Expected noise mean.

    Notes
    -----
    Analytical formula of the function is:

    .. math:: M[E]=M[P]\\sqrt{\\frac{g^{(2)}(P) - 1}{g^{(2)}(E) - 1}}

    Here :math:`E` is a noise distribution,
    :math:`P` is a noised distribution,
    :math:`M[P]` is a mean of P

    .. math:: g^{(2)}(P)=\\dfrac{M^2[P] - M[P]}{M[P]^2}
    """

    return np.sqrt((g2(P) - 1) / (noise_g2 - 1)) * mean(P)


def noiseg2_by_mean(P, nmean):
    """
    Calculate :math:`g^{(2)}` value of the noise distribution for
    given noised distribution and mean value of the noise distribution.

    Parameters
    ----------
    P : array_like
        Noised probability distribution.
    nmean : float
        Expected noise mean.

    Returns
    -------
    g2 : float
        Expected :math:`g^{(2)}` value of the noise distribution.

    Notes
    -----
    Analytical formula of the function is:

    .. math:: g^{(2)}(E)=\\frac{M[P]^2g^{(2)}(P) - M[L]^2 - 2M[L]M[E]}{M[E]^2}

    Here :math:`E` is a noise distribution,
    :math:`L` is a poisson distribution,
    :math:`P` is a noised distribution,
    :math:`M[P]` is a mean of P

    .. math:: g^{(2)}(P)=\\dfrac{M^2[P] - M[P]}{M[P]^2}

    """

    pmean = mean(P) - nmean
    return (mean(P) ** 2 * g2(P) - pmean ** 2 - 2 * nmean * pmean) / nmean ** 2


def denoiseopt(dnmodel, mean_lbound=0, mean_ubound=0,
               g2_lbound=0, g2_ubound=0,
               eps_tol=0, solver=None,
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
        Minimal possible value of noise's mean. The default is 0 and ignored.
        Also ignored if g2 bounds exist.
    mean_ubound : float, optional
        Maximal possible value of noise's mean. The default is 0 and ignored.
        Also ignored if g2 bounds exist.
    g2_lbound : float, optional
        Minimal possible value of noise's g2. The default is 0 and ignored.
        Also ignored if mean bounds exist.
    g2_ubound : float, optional
        Maximal possible value of noise's g2. The default is 0 and ignored.
        Also ignored if mean bounds exist.
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
            'iteration number' 'status (1 is success, 0 is fail)' ['discrepancy', 'noise_mean', 'negentropy', 'noise_g2']
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
    if g2(dnmodel.P) < 1:
        warn('g2 of the given distribution is less than 1. Algorithm can not use g2 and mean bounds.',
             RuntimeWarning, __file__, 171)
 
    if solver is None:
        solver = SolverFactory('ipopt', solver_io="nl")

    y_vars = ['discrepancy', 'noise_mean', 'negentropy', 'noise_g2']
    x_var = 'p_noise'

    if mean_ubound * g2_ubound or mean_ubound * g2_lbound or mean_lbound * g2_ubound or mean_lbound * g2_lbound:
        mean_ubound = 0
        mean_lbound = 0
        warn('It is not possible to set g2 and mean bounds together. Mean bounds are ignored.',
             RuntimeWarning, __file__, 101)

    if mean_ubound:
        dnmodel.c_low_g2 = Constraint(
            expr=dnmodel.noise_g2 >= noiseg2_by_mean(dnmodel.P, mean_ubound))
        dnmodel.c_top_mean = Constraint(expr=dnmodel.noise_mean <= mean_ubound)

    if mean_lbound:
        dnmodel.c_top_g2 = Constraint(
            expr=dnmodel.noise_g2 <= noiseg2_by_mean(dnmodel.P, mean_lbound))
        dnmodel.c_low_mean = Constraint(expr=dnmodel.noise_mean >= mean_lbound)

    if g2_ubound:
        dnmodel.c_top_g2 = Constraint(
            expr=dnmodel.noise_g2 <= g2_ubound)
        dnmodel.c_low_mean = Constraint(
            expr=dnmodel.noise_mean >= noisemean_by_g2(dnmodel.P, g2_ubound))

    if g2_lbound:
        dnmodel.c_low_g2 = Constraint(
            expr=dnmodel.noise_g2 >= g2_lbound)
        dnmodel.c_top_mean = Constraint(
            expr=dnmodel.noise_mean <= noisemean_by_g2(dnmodel.P, g2_lbound))

    mean_bound = mean(dnmodel.P) if not mean_ubound else mean_ubound
    noise_bounds = [[max(0, mean_lbound), -np.log(len(dnmodel.NSET)), max(1, g2_lbound)],
                    [mean_bound, 0, max(20, g2_ubound)]]

    res = iterate(dnmodel, solver, y_vars, x_var,
                  eps_bounds=noise_bounds,
                  eps_tol=eps_tol,
                  save_all_nondom_x=save_all_nondom_x,
                  save_all_nondom_y=save_all_nondom_y,
                  disp=disp)

    info('Optimization ended in %d iterations with eps_tol %.1e' % (res.nit, eps_tol))
    info('Status "{0}", "{1}"'.format(*res.status))

    if plot and res.nit > 1:
        for i in lrange(y_vars):
            plt.plot([e[i] for e in res.ndy], label=y_vars[i])
            plt.legend(frameon=False)
            plt.show()

    return res
