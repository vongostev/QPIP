# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:25:54 2019

@author: von.gostev
"""
import gc
from dataclasses import dataclass
from scipy.signal import find_peaks

import lecroyparser
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
from os.path import isfile, join
from os import listdir
import time
from joblib import Parallel, delayed

gc.enable()


def parse_file(datafile):
    data = lecroyparser.ScopeData(datafile)
    return data


def parse_files(datadir, fnum=0):
    trc_files = [join(datadir, f) for f in listdir(datadir)
                 if isfile(join(datadir, f)) and f.endswith('.trc')]
    if fnum > 0:
        trc_files = trc_files[:fnum]
    data = []
    for datafile in trc_files:
        data.append(parse_file(datafile))

    return data


def windowed(data, div_start, div_width):
    div_points = len(data.x) / data.horizInterval
    wstart = int(div_start * div_points)
    wwidth = int(div_width * div_points)
    return data.x[wstart:wstart + wwidth], data.y[wstart:wstart + wwidth]


def baseline_als(y, lam, p, niter=2):
    # "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def correct_baseline(y, lam=1e5, p=0.1):
    baseline_values = baseline_als(y, lam=lam, p=p)
    return y - baseline_values


@dataclass
class PulsesHistMaker:
    datadir: str
    method: str
    discrete: float
    fsnum: int = 100
    fchunksize: int = 10
    parallel: bool = False
    parallel_jobs: int = -1
    memo: bool = False
    parallel: bool = False
    histbins: int = 2000

    def __post_init__(self):
        if self.method not in ('max', 'counts'):
            raise ValueError('method must be "max" or "counts", not %s' %
                             self.method)

    def read(self):
        self.rawdata = parse_files(self.datadir, self.fsnum)
        self.filesnum = len(self.rawdata)

    def single_pulse(self, y):
        if self.method == 'counts':
            peaks = find_peaks(y, 0.02, 100)
            ypeaks = y[peaks]
            count = 0
            for p in ypeaks:
                for m in range(100):
                    if abs(p / self.discrete - m) < 0.25:
                        count += m
                        break
            return count
        elif self.method == 'max':
            return max(y)

    def single_pulse_hist(self, div_start=5.9, div_width=0.27):
        discretedata = []
        i = 1
        for d in self.rawdata:
            x, y = windowed(d, div_start, div_width)
            discretedata.append(self.single_pulse(y))
            i += 1
        self.hist, self.bins = np.histogram(discretedata, bins=self.nbins)

    def periodic_pulse(self, data, frequency, time_window):
        discretedata = np.array([])
        points_period = int(1 / frequency / data.horizInterval) + 1
        points_window = int(time_window / data.horizInterval) + 1

        y = correct_baseline(data.y)
        init_point = np.argmax(y)
        pulses_points = np.append(
                np.arange(init_point, 0, -points_period)[::-1],
                np.arange(init_point, len(y), points_period))
        for p in pulses_points:
            if p < points_window:
                low = 0
                top = points_window
            else:
                low = p - points_window // 2
                top = p + points_window // 2
            np.append(discretedata, self.single_pulse(y[low:top]))
        return discretedata

    def multi_pulse_histogram(self, frequency=2.5e6, time_window=7.5e-9):
        discretedata = np.array([])

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)
            if self.parallel:
                pulsesdata = Parallel(n_jobs=self.parallel_jobs)(
                    delayed(self.periodic_pulse)(data, frequency, time_window)
                    for data in self.rawdata[i:hb])
            else:
                pulsesdata = [self.periodic_pulse(data, frequency, time_window)
                              for data in self.rawdata[i:hb]]
            discretedata = np.append(discretedata, pulsesdata)
            print('Files ##%d-%d time %.2f s' % (i, hb, time.time() - t), end='\t')
            del pulsesdata
            gc.collect()

        self.hist, self.bins = np.histogram(discretedata, bins=self.nbins)

    def scope_unwindowed(self, data, time_discrete):
        points_discrete = int(time_discrete // data.horizInterval)
        y = data.y
        y -= min(y)
        y = correct_baseline(y)

        points_discrete += 1
        discretedata = [self.single_pulse(y[i:i + points_discrete])
                        for i in range(0, len(y), points_discrete)]
        return discretedata

    def unwindowed_histogram(self, time_discrete=15e-9):
        discretedata = np.array([])

        for i in range(0, self.filesnum, self.fchunksize):
            t = time.time()
            hb = min(i + self.fchunksize, self.filesnum)
            if self.parallel:
                pulsesdata = Parallel(n_jobs=self.parallel_jobs)(
                    delayed(self.scope_unwindowed)(data, time_discrete)
                    for data in self.rawdata[i:hb])
            else:
                pulsesdata = [self.scope_unwindowed(data, time_discrete)
                              for data in self.rawdata[i:hb]]
            discretedata = np.append(discretedata, pulsesdata)
            print('Files ##%d-%d time %.2f s' % (i, hb, time.time() - t), end='\t')
            del pulsesdata
            gc.collect()

        self.hist, self.bins = np.histogram(discretedata, bins=self.nbins)
