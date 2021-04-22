#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:49:11 2020

@author: Pavel Gostev
"""
import numpy as np
from pyomo.core.expr.numvalue import RegisterNumericType
from pyomo.environ import (ConcreteModel, Expression, RangeSet,
                           Var, Param, Constraint, Set,
                           exp, log, quicksum)
from pyomo.dae.plugins.colloc import conv

from fpdet import moment, g2, fact, DPREC


RegisterNumericType(DPREC)


def e_ppoisson(model, n):
    pmean = moment(model.P, 1) - moment(model.p_noise, 1)
    return exp(-pmean) * pmean ** n / fact(n)


def e_p_estimation(model, n):
    return conv(model.ppoisson, model.p_noise)[n]


def e_discrepancy(model):
    return quicksum([(model.PEST[n] - model.P[n]) ** 2 for n in model.PSET])


def e_noisemean(model):
    return moment(model.p_noise, 1)


def e_noiseg2(model):
    return g2(model.p_noise)


def c_noisesum(model):
    return moment(model.p_noise, 0) == 1


def e_negentropy(model):
    """
    Calculation of the negative Shannon entropy for minimization problem

    Parameters
    ----------
    model : InvPBaseModel

    """
    return quicksum(model.p_noise[n] * log(model.p_noise[n])
                    for n in model.NZSET)


class DenoisePBaseModel(ConcreteModel):
    """
    Pyomo model for denoising problem
    It is made for solve four-criterium problem
    to find solution with minimum
    discrepancy, negentropy, g2 and mean of noise distribution

    __init__ arguments
    ----------
    P : numpy.ndarray
        Noised distribution.
    M : int
        Length of noise distribution.
    is_zero_n : numpy.ndarray or list or tuple, optional
        Fix to zero noise distribution for all photon-number stored in iterable.
        The default is 0 and ignored.

    Variables
    ---------
    p_noise : pyomo.environ.Var
        Noise distribution estimation

    Expressions
    ----------
    PEST : pyomo.environ.Expression
        Noised distribution P estimation
    negentropy : pyomo.environ.Expression
        Negentropy of p_noise
    rel_entropy:
        Quadratic discrepancy between P end PEST

    """

    def __init__(self, P, M, is_zero_n=0):
        super().__init__()

        assert type(P) == np.ndarray

        P = P[P != 0]
        N = P.size
        init_noise = np.ones(M) / M

        self.PSET = RangeSet(0, N - 1)
        self.NSET = RangeSet(0, M - 1)
        self.NZSET = RangeSet(0, M - 1)

        self.P = Param(self.PSET, initialize=dict(enumerate(P)))
        self.p_noise = Var(self.NSET, initialize=dict(enumerate(init_noise)),
                           bounds=(0, 1))
        if is_zero_n:
            for n in is_zero_n:
                self.p_noise[n].fix(0)
            self.NZSET = self.NSET - Set(initialize=is_zero_n)

        self.noise_mean = Expression(rule=e_noisemean)
        self.noise_g2 = Expression(rule=e_noiseg2)

        self.ppoisson = Expression(self.PSET, rule=e_ppoisson)
        self.PEST = Expression(self.PSET,
                               rule=e_p_estimation)

        self.discrepancy = Expression(rule=e_discrepancy)
        self.negentropy = Expression(rule=e_negentropy)

        self.c_noisesum = Constraint(rule=c_noisesum)
