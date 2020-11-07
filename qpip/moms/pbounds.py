#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 01:29:57 2020

@article{prekopa1990sharp,
  title={Sharp bounds on probabilities using linear programming},
  author={Pr{\'e}kopa, Andr{\'a}s},
  journal={Operations Research},
  volume={38},
  number={2},
  pages={227--239},
  year={1990},
  publisher={INFORMS}
}

@author: Pavel Gostev
"""
import numpy as np
# import numba as nb
from scipy.linalg import solve, hankel, det
from scipy.special import binom
from scipy.optimize import linprog
from more_itertools import random_combination
# from joblib import Parallel, delayed

from .invpmoms import bmrec_matrices, mrec_matrices, central_moments
from .staterr import covm_mltnomial, covm_transform

sel = np.select


def iseven(x):
    return x % 2 == 0


def isodd(x):
    return x % 2 == 1


def index(arr, val):
    mask = np.argwhere(arr == val)
    return mask[0][0]


def condlen(cond):
    return len(np.where(cond == 1)[0])


def min_feasible_basis(r, K, nmax, i, j, k, t):
    if r > nmax:
        raise ValueError("r must be less than %d" % nmax)

    evenk = iseven(K)
    oddk = isodd(K)

    I1 = np.array([i, i + 1, j, j + 1, r - 1, r, r + 1, k, k + 1, t, t + 1])

    if evenk and r == 0:
        I1 = np.array([0, 1, i, i + 1, j, j + 1])
    elif (evenk and r >= 1 and r <= (nmax - 2)):
        I1 = np.append(I1, nmax)
    elif (evenk and r == nmax - 1):
        I1 = np.append([0], I1)
    elif (evenk and r == nmax):
        I1 = np.array([i, i + 1, j, j + 1, nmax - 1, nmax])

    if oddk and r == 0:
        I1 = np.array([0, 1, i, i + 1, j, j + 1, nmax])
    elif oddk and r >= 1 and r <= (nmax - 1):
        pass
    # elif oddk and r > 1 and r < nmax - 1:
    #    I1 = np.concatenate(([0], I1, [nmax]))
    elif oddk and r == nmax:
        I1 = np.array([0, i, i + 1, j, j + 1, nmax - 1, nmax])

    I1 = np.unique(I1)
    if len(I1) > K:
        if nmax not in I1:
            I1 = I1[:K]
        else:
            I1 = np.append(I1[:K-1], [nmax])
        if r not in I1:
            I1 = np.append(I1, r)[1:]
    elif len(I1) < K:
        sl = np.arange(i + 1, nmax)
        for s in sl:
            if s not in I1 and len(I1) < K:
                I1 = np.append(I1, s)

    I1.sort()
    return I1


def max_feasible_basis(r, K, nmax, i, j, k, t):
    if r > nmax:
        raise ValueError("r must be less than %d" % nmax)

    evenk = iseven(K)
    oddk = isodd(K)

    I1 = np.array([i, i + 1, j, j + 1, r, k, k + 1, t, t + 1])

    if (evenk and r >= 0 and r <= (nmax - 1)):
        I1 = np.append(I1, nmax)
    elif (evenk and r == nmax):
        I1 = np.append(0, I1)

    if oddk:
        pass

    if nmax + 1 in I1:
        I1 = np.delete(I1, index(I1, nmax + 1))

    I1 = np.unique(I1)
    if len(I1) > K:
        I1 = I1[:K]
        if r not in I1:
            I1 = np.append(I1, r)[1:]
    elif len(I1) < K:
        sl = np.arange(i + 1, nmax)
        for s in sl:
            if s not in I1 and len(I1) < K:
                I1 = np.append(I1, s)

    I1.sort()
    return I1


def correct_basis(basis_fun, r, nmax, K):
    initb = []
    while len(initb) != K:
        indexes = random_combination(range(1, nmax - 2), 4)
        initb = basis_fun(r, K, nmax, *indexes)
    return initb


def find_opt_basis(r, W, moms, basis_fun, r_bound):
    K = len(moms)
    nmax = W.shape[1] - 1
    bset = np.arange(nmax + 1)
    Pb = np.zeros(K) - 1
    imem = []

    initb = correct_basis(basis_fun, r, nmax, K)

    optbasis = initb
    optPb = Pb - 1

    npos_cycle = 0
    chbasis = 0
    it = 0
    while any(Pb < 0):
        it += 1

        #try:
        B = W[:, initb]
        Pb = solve(B, moms)
        ilist = initb[(Pb < 0) & (np.abs(initb - r) > r_bound)]
        #except Exception as E:
        #    print(E, r, initb, len(initb), len(moms), Pb)
        #    raise ValueError(E)
        #print(r, initb, ilist, Pb)
        # Step 2
        if len(ilist) >= 1:
            i = ilist[np.random.randint(0, len(ilist))]

        if Pb[index(initb, r)] < 0:
            i = random_combination(initb[np.abs(initb - r) > r_bound], 1)[0]
        elif len(ilist) == 0:
            npos_cycle += 1
            if npos_cycle > 2:
                chbasis += 1
                # print('r = %d, Cycling occured. Changing basis %d' % (r, chbasis),
                #       {r: Pb[i] for i, r in enumerate(initb) if Pb[i] < 0})
                npos_cycle = 0
                if sum([x for x in Pb if x < 0]) > sum([x for x in optPb if x < 0]):
                    optbasis = initb
                    optPb = Pb
                initb = correct_basis(basis_fun, r, nmax, K)
            if chbasis > 99:
                print('r = %d, Iteration is unsuccessful. Result is uncorrect' % r)
                break
            continue
        
        #if it > 1000:
        #    print(1)
        #    initb = correct_basis(basis_fun, r, nmax, K)
        #    continue
        
        if i in imem:
            i = random_combination(
                [k for k in initb if abs(k - r) > r_bound], 1)[0]

        jraw_list = [k for k in bset if k not in initb]
        jlow_list = [k for k in jraw_list if k < i]
        jtop_list = [k for k in jraw_list if k > i]
        j = min(jraw_list)

        if (i < r) and (condlen(initb <= i) == i + 1) and (condlen(initb <= i) > 0):
            # Step 3
            if isodd(condlen((initb > i) & (initb < r))):
                # Step 4
                j = min(jtop_list)
            if iseven(condlen((initb > i) & (initb < r))):
                # Step 5
                j = max(jraw_list)
        elif (i < r) and condlen(initb <= i) != i + 1 and condlen(initb < i) > 0:
            # Step 6
            if jlow_list != []:
                j1 = max(jlow_list)
                if isodd(i - j1):
                    j = j1
            if jtop_list != []:
                j2 = min(jtop_list)
                if isodd(j2 - i):
                    j = j2

        if (i > r) and condlen(initb >= i) == nmax - i:
            # Step 7
            if isodd(condlen((initb > r) & (initb < i))):
                # Step 8
                j = max(jlow_list)
            if iseven(condlen((initb > r) & (initb < i))):
                # Step 9
                j = min(jraw_list)
        elif (i > r) and condlen(initb >= i) != nmax - i and condlen(initb >= i) > 0:
            # Step 10
            if jlow_list != []:
                j1 = max(jlow_list)
                if isodd(i - j1):
                    j = j1
            if jtop_list != []:
                j2 = min(jtop_list)
                if isodd(j2 - i):
                    j = j2

        initb = np.delete(initb, index(initb, i))
        imem.append(i)
        initb = np.sort(np.append(initb, [j]))

    if condlen(Pb < 0) == 0:
        optbasis = initb
        optPb = Pb
    print(r, it, optbasis, optPb)
    return optbasis, optPb


# @nb.jit(forceobj=True)
def minopt_solution(n, W, moms):
    return find_opt_basis(n, W, moms, min_feasible_basis, 1)


# @nb.jit(forceobj=True)
def maxopt_solution(n, W, moms):
    return find_opt_basis(n, W, moms, max_feasible_basis, 0)


def nbound(n, basis, Pb):
    def lagrange(r):
        return np.prod([(r - i) / (n - i) for i in basis if i != n])
    return sum(lagrange(basis[k]) * pk for k, pk in enumerate(Pb))


def linsolve(r, W, moms, sense):
    c = np.zeros(W.shape[1])
    c[r] = sense
    res = linprog(c, A_eq=W, b_eq=moms, bounds=(0, 1), 
                  method='revised simplex',
                  options={'rr': 0, 'autoscale': 1})
    Pb = res.x
    # optbasis = np.sort(np.argsort(Pb)[-len(moms):])
    # c = c[optbasis]
    # W = W[:, optbasis]
    # res = linprog(c, A_eq=W, b_eq=moms, bounds=(0, 1))
    # Pb = res.x
    return Pb / sum(Pb)


def atom_bound(n, W, moms, sense, nmax, method):
    if method == 'prekopa':
        optbasis, Pb = maxopt_solution(n, W, moms)
    if method == 'linsolve':
        Pb = linsolve(n, W, moms, sense)
        optbasis = np.arange(nmax)
    return nbound(n, optbasis, Pb)


def pn_lowbound(Q, qe, nmax, K, method='linsolve'):
    mmax = len(Q)
    W, B = bmrec_matrices(qe, mmax, nmax, K)
    moms = B.dot(Q)
    wmax = np.max(W, axis=1)
    W = np.array([w / np.max(W, axis=1) for w in W.T]).T
    moms = moms / wmax

    bound = [atom_bound(n, W, moms, 1, nmax, method) for n in range(nmax)]
    return bound


def pn_topbound(Q, qe, nmax, K, method='linsolve'):
    mmax = len(Q)
    W, B = bmrec_matrices(qe, mmax, nmax, K)
    moms = B.dot(Q)
    wmax = np.max(W, axis=1)
    W = np.array([w / np.max(W, axis=1) for w in W.T]).T
    moms = moms / wmax

    bound = [atom_bound(n, W, moms, -1, nmax, method) for n in range(nmax)]
    return bound

# ============ OLD BOUNDS ==================
def hdet(moms):
    K = len(moms) // 2
    if len(moms) <= 1:
        return 1
    return abs(det(hankel(moms[:K], moms[K:2*K])))


def hxdet(moms, x):
    K = len(moms) // 2
    H = hankel(moms[:K], moms[K:2*K])
    H[-1] = np.array([x**k for k in range(K)])
    return abs(det(H))


def xpoly(x, k, moms):
    Dnp = hdet(moms[:2*(k-1)])
    Dn = hdet(moms[:2*k])
    Dx = hxdet(moms[:2*k], x)
    return Dx / np.sqrt(Dnp*Dn)


def mrec_bound(Q, qe, N, nmax, K):
    """
    A moments based distribution bounding method
    """
    if K % 2 == 1:
        K -= 1
        print("K is odd, K is decreased to %d" % K)

    W, S, F = mrec_matrices(qe, len(Q), nmax, K)
    T = S.dot(F)
    moms = T.dot(Q)

    COV_nmaxLTQ = covm_mltnomial(Q, N)
    COV_nmaxOnmax = covm_transform(COV_nmaxLTQ, T)
    sigma_mom = np.sqrt(np.abs(np.diag(COV_nmaxOnmax)))
    moms_over = moms + sigma_mom
    moms_under = moms - sigma_mom
    print("Relative delta of moments is %s" %
          {i: '%.5f%%' % (d*100) for i, d in enumerate(sigma_mom / moms)})

    rhos = []
    for n in range(nmax):
        opolysum = sum(np.abs(xpoly(n, k, moms_over)) **
                       2 for k in range(1, K // 2, 1))
        upolysum = sum(np.abs(xpoly(n, k, moms_under))
                       ** 2 for k in range(1, K // 2, 1))
        polysum = min(opolysum, upolysum)
        rhos.append(1/(1 + polysum))
    return rhos


def mrec_bound2(Q, qe, N, nmax, K):
    """
    nmaxoments Determine the Tail of a Distribution (But Not much Else)
    """
    if K % 2 == 1:
        K -= 1
        print("K is odd, K is decreased to %d" % K)
    COV_MLTQ = covm_mltnomial(Q, N)
    W, S, F = mrec_matrices(qe, len(Q), nmax, K)
    T = S.dot(F)
    moms = T.dot(Q)
    cmoms = central_moments(T.dot(Q))
    COV_M0M = covm_transform(COV_MLTQ, T)
    sigma_mom = np.sqrt(np.abs(np.diag(COV_M0M)))
    sigma_cmom = central_moments(sigma_mom)
    moms_over = cmoms + sigma_cmom
    moms_under = cmoms - sigma_cmom
    if any(moms_under[2:] < 0):
        K = len(moms_under[moms_under >= 0])
        if K % 2 == 1:
            K -= 1
        print("Lower bound of moments with statistical errors (N = %d) < 0. K is decreased to %d" % (N, K))
    print("Relative delta of moments is %s" %
          {i: '%.5f%%' % (d*100) for i, d in enumerate(sigma_cmom / cmoms)})
    H_OVER = hankel(moms_over[:K//2], moms_over[K//2:K])
    H_UNDER = hankel(moms_under[:K//2], moms_under[K//2:K])
    deltaPn = []
    for n in range(nmax):
        v = (n - moms[1])**np.arange(K//2)
        den_over = abs(v.T.dot(H_OVER).dot(v))
        den_under = abs(v.T.dot(H_UNDER).dot(v))
        den = min(den_over, den_under)
        deltaPn.append(1/den)
    return deltaPn


def bmrec_bonferonni(Q, qe, N, nmax, K, i=0, j=0):
    mmax = len(Q)
    if nmax == 0:
        nmax = mmax
    W, B = bmrec_matrices(qe, mmax, nmax, nmax)
    moms = B.dot(Q)

    COV_nmaxLTQ = covm_mltnomial(Q, N)
    COV_nmaxOnmax = covm_transform(COV_nmaxLTQ, B)
    sigma_mom = np.sqrt(np.abs(np.diag(COV_nmaxOnmax)))
    print("Relative delta of moments is %s" %
          {i: '%.5f%%' % (d*100) for i, d in enumerate(sigma_mom / moms)})

    def bsum(a, k, sigma, kmax):
        return sum((-1)**(r-k)*binom(r, k)*(moms[r] + sigma)
                   for r in range(k, min(k + kmax, nmax), 1))

    return [bsum(moms, m, 0, i) for m in range(nmax)]
    #low = [bsum(moms, m, -sigma_mom[m], 2*i + 1) for m in range(nmax-2*i-1)]
    #top = [bsum(moms, m, sigma_mom[m], 2*j) for m in range(nmax-2*j)]
    # return low, top
