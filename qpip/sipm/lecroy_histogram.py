# -*- coding: utf-8 -*-
from ..detection_core import normalize, mean, g2, lrange, abssum

import os
import numpy as np
from scipy.signal import find_peaks
import pybobyqa

import matplotlib.pyplot as plt
import logging

log = logging.getLogger('hist')
log.setLevel(logging.INFO)
if (log.hasHandlers()):
    log.handlers.clear()
info = log.info

EXT = ".png"


def hist2Q(hist, discrete=5, threshold=1, plot=False):
    hist = np.concatenate(([0], hist))
    peaks, _ = find_peaks(hist, threshold=threshold, distance=discrete)
    downs, _ = find_peaks(-hist, distance=discrete)
    downs = np.append([0], downs)
    if plot:
        plt.plot(hist)
        plt.scatter(peaks, hist[peaks])
        plt.scatter(downs, hist[downs])
        plt.show()

    Q = []
    for i in lrange(downs)[:-1]:
        for p in peaks:
            if p > downs[i] and p < downs[i+1]:
                Q.append(sum(hist[downs[i]:downs[i + 1]]))
    return normalize(Q)


class QStatisticsMaker:

    def __init__(self, fname, photon_discrete,
                 skiprows=0, offset=0, method='auto', plot=False):

        self.photon_discrete = photon_discrete
        self.fname = fname
        self.Q = []
        self.plot = plot

        self._extract_data(skiprows)

        if method == 'auto':
            self.Q = hist2Q(self.hist, discrete=self.points_discrete // 2,
                            plot=self.plot)
        else:
            self._nonauto_make_hist(method, offset)
        # All attributes
        # hist, photocounts_stats, intervals_num, fname, photon_discrete

    # Reading information from file
    def _extract_data(self, skiprows):
        bins, hist = np.loadtxt(self.fname, skiprows=skiprows).T
        if bins[0] > -1.5e-11:
            delta = bins[1] - bins[0]
            N = int((bins[0] + 1.5e-11) // delta)
            hist = np.concatenate((np.zeros(N), hist))
            bins = np.concatenate(
                (bins[0] - np.arange(N + 1, 1, -1) * delta, bins))

        self.hist = hist
        self.bins = bins
        self.points_discrete = int(self.photon_discrete /
                                   (self.bins[1] - self.bins[0]))

    def _nonauto_make_hist(self, method, offset):
        raise ValueError('Only auto method can be applied')

    def getq(self):
        self.Q = self.Q[self.Q > 0]
        return normalize(self.Q)
