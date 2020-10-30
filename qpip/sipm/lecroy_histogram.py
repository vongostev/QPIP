# -*- coding: utf-8 -*-
"""
@author: Pavel Gostev
"""
from ..detection_core import normalize, lrange

import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import logging

log = logging.getLogger('hist')
log.setLevel(logging.INFO)
if (log.hasHandlers()):
    log.handlers.clear()
info = log.info


def hist2Q(hist, discrete, threshold=2, peak_width=1, plot=False):
    """
    Build photocounting statistics from an experimental histogram

    Parameters
    ----------
    hist : iterable
        The experimental histogram.
    discrete : int
        The amplitude of single photocount pulse in points.
    threshold : int, optional
        Minimal number of events to find histogram peak.
        The default is 2.
    peak_width : int, optional
        The width of peaks.
        It must be 1 if the histogram is made by 'count' method.
        It must be greater if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.

    Returns
    -------
    Q : ndarray
        The photocounting statistics.

    """
    hist = np.concatenate(([0], hist))
    peaks, _ = find_peaks(hist, threshold=threshold, distance=discrete,
                          width=peak_width)
    downs, _ = find_peaks(-hist, distance=discrete, width=peak_width)
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
    """
    Class to make photocounting statistics from histogram

    __init__ arguments
    ----------
    fname : string
        File name contains the histogram.
    photon_discrete : float
        The amplitude of the single-photocount pulse.
    peak_width : int, optional
        The width of peaks.
        It must be 1 if the histogram is made by 'count' method.
        It must be greater if the histogram is made by oscilloscope or 'max' method.
        The default is 1.
    skiprows : int, optional
        Number of preamble rows in the file. The default is 0.
    plot : bool, optional
        Flag to plot hist and results of find_peaks.
        The default is False.

    Methods
    -------
    getq : ndarray
        Returns the photocounting statistics was made
        It is self.Q
    """

    def __init__(self, fname, photon_discrete,
                 peak_width=1, skiprows=0, plot=False):
        self.photon_discrete = photon_discrete
        self.fname = fname
        self.Q = []
        self.plot = plot

        self._extract_data(skiprows)
        self.Q = hist2Q(self.hist, discrete=self.points_discrete // 2,
                        peak_width=peak_width, plot=self.plot)

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

    def getq(self):
        self.Q = self.Q[self.Q > 0]
        return normalize(self.Q)
