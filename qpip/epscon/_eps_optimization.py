# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:14:53 2019

@author: vonGostev

Epsilon-constraint algorithm with arbitrary number of variables

Kirlik, G., & Sayın, S. (2014).
A new algorithm for generating all nondominated solutions
of multiobjective discrete optimization problems.
European Journal of Operational Research, 232(3), 479–488.
doi:10.1016/j.ejor.2013.08.001
"""
from pyomo.environ import (Objective, Constraint, ConstraintList,
                           value, minimize)

import numpy as np
from scipy.optimize import OptimizeResult

import logging

logger = logging.getLogger('epsopt')
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
info = logger.info
error = logger.error


def pyomo_values(var):
    return np.array([value(e) for e in var.values()])


def greater(l1, l2):
    return all([e1 > e2 for e1, e2 in zip(l1, l2)])


def P1(epsilon, model, y_vars):
    """

    Parameters
    ----------
    epsilon : list
        Upper bound of y_vars[1:].
    model : pyomo.environ.ConcreteModel
        pyomo model of the problem in the initial state.
    y_vars : list
        Names of model expressions to optimize.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model in the final state

    """

    if hasattr(model, 'Obj') and hasattr(model, 'z_constraint'):
        model.del_component(model.z_constraint)
        model.del_component(model.Obj)
        model.epsilon_constraints.clear()

    model.Obj = Objective(expr=getattr(model, y_vars[0]), sense=minimize)
    for n, v in enumerate(y_vars[1:]):
        model.epsilon_constraints.add(expr=getattr(model, v) <= epsilon[n])

    return model


def Q1(epsilon, z_optimal, model, y_vars):
    """

    Parameters
    ----------
    epsilon : list
        Upper bound of y_vars[1:].
    z_optimal : float
        Value for equality of y_vars[0].
    model : pyomo.environ.ConcreteModel
        pyomo model of the problem in the initial state.
    y_vars : list
        Names of model expressions to optimize.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
        The model in the final state

    """

    if hasattr(model, 'Obj'):
        model.del_component(model.Obj)
        model.epsilon_constraints.clear()

    model.Obj = Objective(
        expr=sum(getattr(model, v) for v in y_vars[1:]),
        sense=minimize)
    for n, v in enumerate(y_vars[1:]):
        model.epsilon_constraints.add(expr=getattr(model, v) <= epsilon[n])
    model.z_constraint = Constraint(
        expr=getattr(model, y_vars[0]) == z_optimal)
    return model


def step_solve(epsilon, model, solver, y_vars, x_var):
    """

    Parameters
    ----------
    epsilon : list
        Upper bound of y_vars[1:].
    model : pyomo.environ.ConcreteModel
        pyomo model of the problem.
    solver : pyomo.opt.Solver
    y_vars : list
        Names of model expressions to optimize.
    x_var : string
        Name of model variable.

    Raises
    ------
    ValueError
        Error of values extraction from y_vars.

    Returns
    -------
    solve_flag : int
        Flag of solution: 1 is OK, 0 is fail.
    res : list
        List of y_vars values.

    """
    solve_flag = 1

    try:
        P = P1(epsilon, model, y_vars)
        solver.solve(P, tee=0)
    except Exception as E:
        error('epsilon = %s. Error in P: %s' % (epsilon, E))
        solve_flag = 0

    try:
        z_optimal = value(getattr(model, y_vars[0]))
        Q = Q1(epsilon, z_optimal, model, y_vars)
        solver.solve(Q, tee=0)
    except Exception as E:
        error('epsilon = %s. Error in Q: %s' % (epsilon, E))
        solve_flag = 0

    try:
        res = [value(getattr(model, v)) for v in y_vars]
    except ValueError as E:
        raise ValueError(
            '\nepsilon= %s. Error in numeric evaluation of y_vars.' % epsilon +
            '\n\tMaybe you try to calculate log(0).' +
            '\n\tError: %s' % E)
    return solve_flag, res


def rectangle_volume(u, u_min):
    """
    Calculation of rectangle volume

    Parameters
    ----------
    u : list
        Upper bound of rectangle.
    u_min : list
        Lower bound of rectangle.

    Returns
    -------
    float
        Volume of specified rectangle.

    """

    return np.prod([e - u_min[i] for i, e in enumerate(u)])


def remove_rectangle(rlist, u_low, u):
    """
    Function to remove non-optimal rectangle
    from the list of rectangles

    Parameters
    ----------
    rlist : list
        List of rectangles.
    u_low : list
        Lower bound of the rectangle to remove.
    u : list
        Upper bound of the rectangle to remove.

    Returns
    -------
    list
        Updated list of rectangles.

    """

    return [r for r in rlist if r != [u_low, u]]


def update_rlist(rlist, f_optimal):
    """
    Function to update the list of rectangles
    by splitting of rectangles

    Parameters
    ----------
    rlist : list
        List of rectangles.
    f_optimal : list
        Optimal solution from current iteration.

    Returns
    -------
    new_rlist : list
        Updated list of rectangles.

    """

    new_rlist = []
    for r in rlist:
        if greater(f_optimal, r[0]) and greater(r[1], f_optimal):
            new_rlist.append([r[0], f_optimal])
            new_rlist.append([f_optimal, r[1]])
    return new_rlist


def get_epsilon(rlist):
    volumes = [rectangle_volume(r[1], rlist[0][0]) for r in rlist]
    return rlist[np.argmax(volumes)][1]


def iterate(model, solver, y_vars, x_var, eps_bounds,
            eps_tol=0, maxiter=0, save_all_nondom_x=False,
            save_all_nondom_y=False, disp=False):
    """
    Iteration process to find optimal solution
    in quadratic sense

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        pyomo model of the problem.
    solver : pyomo.opt.Solver
    y_vars : list
        Names of model expressions to optimize.
    x_var : string
        Name of model variable.
    eps_bounds : list of lists
        Lower and upper bound of expression to optimize.
    eps_tol : float, optional
        Tolerance of y_vars to terminate iterations.
        The default is 0 and ignored.
    maxiter : int, optional
        Maximum number of iterations.
        The default is 0 and ignored.
    save_all_nondom_x : bool, optional
        Save all x_var values.
        The default is False.
    save_all_nondom_y : bool, optional
        Save all y_vars variables.
        The default is False.
    disp : bool, optional
        Display optimization progress.
        The default is False.

    Returns
    -------
    scipy.optimize.OptimizeResult
            status : (str, str)
                Short and long definition of termination status
            nit : int
                Number of iterations
            model : pyomo.environ.ConcreteModel
                Final state of the model of the problem
            x : ndarray
                Final solution
            y : dict
                Criteria values related to x
            ndx : ndarray
                List of all nondominate solutions
            ndy : ndarray
                List of critera values related to ndx

    References
    ----------
    .. [1]
    Kirlik, G., & Sayın, S. (2014), "A new algorithm for generating all
    nondominated solutions of multiobjective discrete optimization problems."
    European Journal of Operational Research, 232(3), 479–488.
    doi:10.1016/j.ejor.2013.08.001
    """

    u_min, u_max = eps_bounds

    rectangle_list = [[u_min, u_max]]
    if hasattr(model, 'epsilon_constraints'):
        model.epsilon_constraints.clear()
    else:
        model.epsilon_constraints = ConstraintList()

    nondom_y = []
    nondom_x = []

    badloop_flag = 0
    status = 'noopt'
    status_messages = {
        'noopt':   'Optimization did not started',
        'success': 'Optimization terminated successfully.',
        'maxiter': 'Maximum number of iterations has been '
                   'exceeded.',
        'badloop': 'Iterations fall to cycle due model errors',
        'nosolve': 'Problem is infeasible in given constraints',
        'ftolsuccess': 'Optimization terminated successfully with finite tolerance '
                       'result may be inaccurate.'
        }

    while len(rectangle_list) != 0:
        epsilon = get_epsilon(rectangle_list)

        if len(nondom_y) > 2:
            delta_norm = sum((x - y) ** 2 for x, y in
                             zip(nondom_y[-1][1:], nondom_y[-2][1:])) ** 0.5
            if delta_norm < eps_tol:
                status = 'ftolsuccess'
                break

        if badloop_flag > 2:
            for n, e in enumerate(epsilon):
                epsilon[n] = e - 0.01 * abs(e)

        solve_flag, f_optimal = step_solve(
            epsilon, model, solver, y_vars, x_var)

        if not solve_flag:
            rectangle_list = remove_rectangle(
                rectangle_list, u_min, f_optimal[1:])
            badloop_flag += 1
            if badloop_flag > 5:
                status = 'badloop'
                break
            else:
                continue
        else:
            badloop_flag = 0

        if f_optimal not in nondom_y:
            nondom_y.append(f_optimal)
            rectangle_list = update_rlist(
                rectangle_list, f_optimal[1:])
        else:
            status = 'success'
            break

        rectangle_list = remove_rectangle(
            rectangle_list, f_optimal[1:], epsilon)

        if disp:
            info('%d %d %s' % (len(nondom_y), solve_flag, f_optimal))

        if save_all_nondom_x:
            nondom_x.append(pyomo_values(getattr(model, x_var)))

        if maxiter and len(nondom_y) > maxiter:
            status = 'maxiter'
            break

    if len(nondom_y) > 0 and solve_flag == 1:
        status = 'success'
    if len(nondom_y) == 0:
        status = 'nosolve'

    res = {
        'status': (status, status_messages[status]),
        'nit': len(nondom_y),
        'model': model,
        'x': pyomo_values(getattr(model, x_var)),
        'y': dict(zip(y_vars, nondom_y[-1] if len(nondom_y) > 0 else np.zeros_like(y_vars))),
        'ndx': np.array(nondom_x) if save_all_nondom_x else np.array([]),
        'ndy': np.array(nondom_y) if save_all_nondom_y else np.array([])
    }

    return OptimizeResult(res)
