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
- denoise:
    Pyomo model and epsilon-constrainted algorithm to recover noise statistics of laser source or signal statistics against the background of coherent noise (for example in collinear SPDC analysing). Here we use a method to determine photon-number (or photocounting) statistics of excess noise in laser radiation from measured photocounting statistics. The method based on the multi-objective optimization approach is applied to blind deconvolution problem to determine excess noise distribution from the convolution of this one and poissonian photon-number distribution of laser radiation. See [related material](www.researchgate.net/publication/345087870) for additional details.

- sipm:
    Instruments to make photocounting statistics from histograms and raw oscillograms (maked by lecroy oscilloscope) of SiPM signal. One can correct baseline of oscillogram and crosstalk noise of photocounting statistics.

# How to use qpip.epscon?
Import necessary modules:
```python
import numpy as np
from qpip.stat import ppoisson
from qpip import normalize, P2Q, fidelity
from qpip.epscon import InvPBaseModel, invpopt
```
Make photocounting statistics for the sample of finite size in presence of noise
```python
N = 25 # length of the photon-number statistics
M = 25 # length of the photocounting statistics
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
```python
invpmodel = InvPBaseModel(QN, qe, N) # make optimization model
res = invpopt(invpmodel, eps_tol=1e-5) # optimize it!
print(res) # print result (OptimizeResult)
print(fidelity(res.x, P))
```

# How to use qpip.denoise?
Import necessary modules:
```python
import numpy as np
from qpip.stat import ppoisson, pthermal
from qpip import normalize, P2Q, fidelity, p_convolve
from qpip.denoise import DenoisePBaseModel, denoiseopt
```
Make noised statistics for the sample of finite size in presence of noise
```python
N = 25 # length of the laser distribution
M = 10 # length of the noise distribution
N0 = int(1E6) # number of events
ERR = 1e-2 # relative error for photocounting statistics

noise = pthermal(1, M)
P = normalize(p_convolve(ppoisson(6, N), noise)) # noised distribution
PND = np.random.choice(range(N), size=N0, p=P.astype(float)) # random events
PN = np.histogram(PND, bins=range(N + 1), density=True)[0] 
PN = np.abs(PN*(1 + np.random.uniform(-ERR, ERR, size=N)))
PN = normalize(PN) # statistics for the sample of N0 size
```
Find optimal estimation
```python
dnpmodel = DenoisePBaseModel(PN, M) # make optimization model
res = denoiseopt(dnpmodel, eps_tol=1e-5, mean_ubound=2) # optimize it!
print(res) # print result (OptimizeResult)
print(fidelity(res.x, noise))
```

# Requirements
See requirements.txt
