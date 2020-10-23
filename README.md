# QPIP
 Instruments and methods for solve quantum photocounting inverse problem

QPIP module consists of some parts:
- core: Function for calculate photocounting statistics from photon-number statistics and vice versa. Also a bit of utility functions in _numpy_core
- epscon: 
    Pyomo model and epsilon-constrainted algorithm to recover photon-number statistics from photocounting statistics in the presence of noise
    The approach based on minimization of differential entropy of photocounting statistics and its estimation and maximization of photon-number statistics estimation entropy
    qpip.epscon can be used for recover any photon-number statistics with any quantum efficiency of detector.
    One can use both of binomial and subbinomial models of photodetection.
- stat: Some examples of photon-number and photocounting statistics corresponding to quantum light fields

# How to use qpip.epscon?
Import necessary modules:
```
import numpy as np
from qpip.stat import ppoisson
from qpip import normalize, P2Q, fidelity
from qpip.epscon import InvPBaseModel, epsopt
```
Make photocounting statistics for the sample of finite size in presence of noise
```
N = 25 # length of photon-number statistics
M = 25 # length of photocounting statistics
N0 = int(1E6) # number of photocounting events
qe = 0.3 # quantum efficiency
ERR = 1e-2 # relative error for photocounting statistics

P = ppoisson(7, N) # photon-number distribution
Q = P2Q(P, qe, M) # photocounting distribution
QND = np.random.choice(range(M), size=N0, p=Q.astype(float)) # random photocounting events
QN = np.histogram(QND, bins=range(M + 1), density=True)[0] 
QN = np.abs(QN*(1 + np.random.uniform(-ERR, ERR, size=len(QN))))
QN = normalize(QN) # photocounting statistics for the sample of N0 size
```
Find optimal estimation
```
invpmodel = InvPBaseModel(QN, qe, N) # make optimization model
res = epsopt(invpmodel, eps_tol=1e-5) # optimize it!
print(res) # print result (OptimizeResult)
print(fidelity(res.x, P))
```

# Requirements
- numpy, scipy, pyomo, qutip (for qpip.stat), matplotlib
- numpy.float128
