# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:46:55 2020

@author: von.gostev
"""
import numpy as np

from pyomo.environ import (RangeSet, Set, ConcreteModel, Param,
                           Var, Expression, Constraint, value,
                           quicksum, log)
from pyomo.core.expr.numvalue import RegisterNumericType

from ..detection_core import compose, moment, t_matrix, DPREC

RegisterNumericType(DPREC)

dictenum = compose(dict, enumerate)


def pyomo_values(var):
    """
    Util for extraction of number values from pyomo vector variable

    Parameters
    ----------
    var : pyomo.environ.Var
        Pyomo vector variable.

    Returns
    -------
    numpy.ndarray
        Array of values stored in variable.

    """

    return np.array([value(e) for e in var.values()])


def c_sum_qest(model):
    """

    Parameters
    ----------
    model : InvPBaseModel

    Returns
    -------
    Constraint rule for equality of sum(QEST) and 1

    """
    return moment(model.QEST, 0) == 1


def c_qbounds(model, m):
    """
    Constraint rule for upper bound for absolute difference:
        |Q[m]-QEST[m]| < exp_precision*QEST[m].

    Parameters
    ----------
    model : InvPBaseModel
    m : int
        photocounting number.

    """

    return (0, abs(model.QEST[m] - model.Q[m]),
            model.exp_precision * model.Q[m])


def c_moments(model, k):
    """
    Constraint rule for difference QEST initial moment of order k
    and the same moment of Q.
    This difference must be less than 1%

    Parameters
    ----------
    model : InvPBaseModel
    k : int
        Order of moment.

    """

    return (1 - 1e-2, moment(model.QEST, k)/moment(model.Q, k), 1 + 1e-2)


def e_qest(model, m):
    """
    Calculation of photocounting statistics estimation from
    photon-number statistics estimation

    Parameters
    ----------
    model : InvPBaseModel
    m : int
        Photocount number.

    """
    return quicksum(model.T[m, n] * model.PEST[n]
                    for n in model.PSET)


def e_negentropy(model):
    """
    Calculation of the negative Shannon entropy for minimization problem

    Parameters
    ----------
    model : InvPBaseModel

    """
    return quicksum(model.PEST[n] * log(model.PEST[n])
                    for n in model.NZPSET)


def e_rel_entropy(model):
    """
    Calculation of Kullback-Leibler divergence between
    experimental photocounting statistics and it's estimation

    Parameters
    ----------
    model : InvPBaseModel

    """
    return quicksum(model.QEST[m] * log(model.QEST[m]/model.Q[m])
                    for m in model.QSET)


def e_pest(model, n):
    """
    It's a monkey patch
    Without this simply expression there is a error in solver.solve
    ValueError: Unsupported expression type
                (<class 'pyomo.core.expr.numeric_expr.LinearExpression'>)
                in _print_nonlinear_terms_NL

    Parameters
    ----------
    model : InvPBaseModel
    n : int
        photon number

    """
    return model.P[n]


class InvPBaseModel(ConcreteModel):
    """
    Pyomo model for inverse photocounting problem
    It is made for solve bi-criterium problem
    to find solution with minimum disprepancy and negentropy

    __init__ arguments
    ----------
    Q : numpy.ndarray or list
        Photocounting statistics.
    qe : float
        Photon Detection Efficiency (PDE) of the detector.
    N : int
        Maximum numbers of photons
    exp_precision : float, optional
        Upper bound for absolute difference: |Q-QEST| < exp_precision*QEST.
        The default is 0 and ignored.
    state_odd : bool, optional
        Fix to 0 every even photon-number.
        The default is False and ignored.
    state_even : bool, optional
        Fix to 0 every odd photon-number.
        The default is False and ignored.
    is_zero_n : numpy.ndarray or list or tuple, optional
        Fix to zero photon-number statistics for all photon-number stored in iterable.
        The default is 0 and ignored.
    mtype : {'binomial', 'subbinomial'}, optional
        Type of the detector: ideal is binomial, realistic is subbinomial,
        but in most applications one can consider the detector as binomial
        The default is 'binomial'.
    n_cells : TYPE, optional
        Number of photocounting cells is subbinomial case. The default is 0.

    Variables
    ---------
    P : pyomo.environ.Var
        Photon-number statistics estimation

    Expressions
    ----------
    PEST : pyomo.environ.Expression
        Photon-number statistics estimation
    QEST : pyomo.environ.Expression
        Photocounting statistics estimation from PEST
    negentropy : pyomo.environ.Expression
        Negentropy of PEST
    rel_entropy:
        Relative entropy between Q end QEST

    Constraints
    -----------
    c_sum_qest : pyomo.environ.Constraint
        Constraint rule for equality of sum(QEST) and 1
    c_qbounds : pyomo.environ.Constraint
        Constraint rule for upper bound for absolute difference:
            |Q-QEST| < exp_precision*QEST
        for any photocouning number
    c_moments : pyomo.environ.Constraint
        Constraint rule for difference QEST initial moment of order k
        and the same moment of Q.

    """

    def __init__(self, Q, qe, N,
                 exp_precision=0,
                 state_odd=False,
                 state_even=False,
                 is_zero_n=0,
                 mtype='binomial',
                 n_cells=0):

        super().__init__()

        Q = Q[Q > 0]
        M = len(Q)
        init_pn = np.ones(N) / N
        T = t_matrix(qe, N, M, mtype, n_cells)

        self.PSET = RangeSet(0, N - 1)
        self.NZPSET = RangeSet(0, N - 1)
        self.P = Var(self.PSET,
                     initialize=dictenum(init_pn),
                     bounds=(1e-31, 1))
        if state_odd:
            for n in self.PSET:
                if n % 2 != 0:
                    self.P[n].fix(0)
            self.NZPSET = RangeSet(0, N - 1, 2)
        if state_even:
            for n in self.PSET:
                if n % 2 == 0:
                    self.P[n].fix(0)
            self.NZPSET = RangeSet(1, N - 1, 2)
        if is_zero_n:
            for n in is_zero_n:
                self.P[n].fix(0)
            self.NZPSET = self.PSET - Set(initialize=is_zero_n)

        self.QSET = RangeSet(0, M - 1)
        self.MOM_SET = RangeSet(1, 2)

        self.QEST = Param(initialize=qe)
        self.T = Param(self.QSET, self.PSET,
                       initialize={
                           (i, j): DPREC(T[i, j]) for
                           i in self.QSET for j in self.PSET})

        self.Q = Param(self.QSET, initialize=dictenum(Q))
        self.PEST = Expression(self.PSET, rule=e_pest)
        self.QEST = Expression(self.QSET, rule=e_qest)
        self.negentropy = Expression(rule=e_negentropy)
        self.rel_entropy = Expression(rule=e_rel_entropy)

        self.c_sum_qest = Constraint(rule=c_sum_qest)
        self.c_moms = Constraint(self.MOM_SET, rule=c_moments)

        if exp_precision > 0:
            self.exp_precision = Param(initialize=exp_precision)
            self.c_qbounds = Constraint(self.QSET, rule=c_qbounds)
