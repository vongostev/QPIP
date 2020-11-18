# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:27:47 2019

@author: vonGostev
"""
import os
import __init__
from qpip import normalize, P2Q, Q2P, mean, g2, fidelity, p_convolve, entropy
from qpip.epscon import invpopt, InvPBaseModel
import qpip.sipm as scc
import cdata_plotter as cdp
from qpip.stat import psqueezed_coherent1, psqueezed_vacuumM, ppoisson, pthermal
from qpip.sipm import QStatisticsMaker, optimize_pcrosstalk, compensate
import numpy as np
import matplotlib.pyplot as plt
import logging

from test_algorithm import make_qmodel

FLOG = False
logger = logging.getLogger('compose_data')
logger.setLevel(logging.CRITICAL)
if (logger.hasHandlers()):
    logger.handlers.clear()
if FLOG:
    fh = logging.FileHandler("./log/compose_exp_0410.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    optimizelog = logging.getLogger('sc_pyomo')
    optimizelog.addHandler(fh)

pyomolog = logging.getLogger('pyomo.core')
pyomolog.setLevel(logging.ERROR)

info = logger.critical


def file2Q(fname, photon_discrete, skiprows=0, peak_width=3.5):
    pe_signal = QStatisticsMaker(
        fname, peak_width=peak_width, photon_discrete=photon_discrete, 
        skiprows=skiprows, plot=1, method='sum').getq()
    return pe_signal


"""============ Управляющие константы ===================="""
# Размерность сетки расчета
N = 20
# Максимальное число фотоотсчетов для обрезки (моделирование
# экспериментальной ситуации)
M = N
# Ключи использования разных данных
# Использовать экспериментальные данные? Если ЛОЖЬ, используются модельные
# данные
EXP = 4
# Квантовая эффективность
QE_T = 0.12  # 812 nm
QE_M = 0.15  # 660 nm
QE = [0.3, QE_M, QE_T, 0.36, 0.32][EXP]

ADJ_CROSSTALK = False
DISPLAY_ERR = False
HIST_METHOD = ['fit', 'sum', 'auto'][2]

# Parameters of photodetection
MTYPE = 'binomial'
N_CELLS = 667

""" =========== Обработка модельных данных ==============="""
if EXP == 0:
    N0 = int(1E5)
    #ideal_pn_model = ss.correct_thermal(4, N)  # [0.8, 0.15, 0.05, 0] #
    # ideal_pn_model = normalize(p_convolve(ss.correct_poisson(12, N),
    #                               ss.hyper_poisson(range(N), 0.6, 4)))
    #pnoise = [1 - 3e-3, 0, 3e-3]#pthermal(1, 10) # 
    #ideal_pn_model = normalize(np.convolve(ppoisson(0.1, N), pnoise)[:N])
    #ideal_pn_model = normalize(
    #    ppoisson(15, N) + ppoisson(5, N))

    #ideal_pn_model = squeezed_vacuumM(N, 2, 1.3, 0)
    #ideal_pn_model = np.zeros(N)
    #ideal_pn_model[6] = 1
    # ideal_pn_model = np.array([sp.Float(x) for x in
    #                           normalize(ket2dm(coherent(N, 3) + coherent(N, -3)).diag())])
    #ideal_pn_model = ss.hyper_poisson(range(N), 9, 6.5)
    #ideal_pn_model = ket2dm(psqueezed_coherent1(N, 3, 1)).diag()
    ideal_pn_model = ppoisson(7, N)
    ERR = 0.5e-2
    pm_exp, pm_processed = make_qmodel(
        ideal_pn_model, MTYPE, N_CELLS, qe=QE, M=M, N0=N0, ERR=ERR)
    #pm_processed[-5:] = 0
    info('M %d' % M)
    #info('H[pn] %f' % entropy(pn_model))
    info('H[pn, ideal] %f' % entropy(ideal_pn_model))

    plt.plot(ideal_pn_model, linewidth=2, label='P[ideal]')
    #plt.plot(pm_exp, linewidth=2, label='q[ideal]')
    plt.plot(pm_processed, linewidth=2, label='q')
    plt.plot(Q2P(pm_processed, QE), linewidth=2, label=r'$P_{rec}$')
    plt.legend(frameon=False, fontsize=16)
    plt.xlabel('Number of photons/photocounts', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    #assert type(pn_model[0]) in [sp.numbers.Float]

""" =========== Обработка экспериментальных данных ==============="""
# ============ CONSTANTS =============
if EXP == 1:
    # It's minimum P_CROSSTALK for mean = 2.3 (test_data_mean2_5)
    P_CROSSTALK = 0.021344390144970308  # 0.00486707  # 0.021344390144970308
    PEAK_AREA_LASER = {
        '34v': 4.2e-11,
        '35v': 7.1e-11,
        '36v': 6.9e-11,  # 7.4e-11 #for laser,
        '37v': 9.5e-11
    }  # pV*s  - amplituda '1' pika, ostalnii v 'n' raz bolshe

    PEAK_AREA_SPDC = {
        '36v': 6.4e-11
    }

    data_dir = r"../../SiPM MEPHI Results/35v"
    test_data_mean2_5 = os.path.join(
        data_dir, 'F10_100_660nm_Stats00000(2).txt')
    test_data_mean0_5 = os.path.join(
        data_dir, 'F100100_0794_660nm_Stats00000.txt')
    test_data_mean0_4 = os.path.join(
        data_dir, 'F100010_0794_660nm_Stats00000.txt')
    test_data_mean2_3 = os.path.join(
        data_dir, 'F101000_0794_660nm_Stats00000.txt')
    test_data_mean6_9 = os.path.join(
        data_dir, 'F105000_0501_660nm_Stats00000.txt')
    test_data_mean3_3 = os.path.join(
        data_dir, 'F105000_0251_660nm_Stats00000.txt')
    test_data_mean15 = os.path.join(data_dir, 'F10_500_660nm_Stats00000.txt')
    test_data_mean28 = os.path.join(data_dir, 'F10_000_660nm_Stats00000.txt')
    test_data_mean1_6 = os.path.join(
        data_dir, 'F101000_0631_660nm_Stats00000.txt')
    test_data_spdc_noised = os.path.join(
        data_dir, 'F100100_1000_660nm_Stats00000.txt')

    pm_exp = file2Q(test_data_mean0_5,
                                 PEAK_AREA_LASER['36v'],
                                 skiprows=1)
    ideal_pn_model = ppoisson(mean(pm_exp) / QE, N)
    pnoise = np.zeros(N)
    
elif EXP == 2:
    P_CROSSTALK = 0.013292422154555481
    dcr_distribution = normalize([242754, 89, 7])
    test_data_24_05_3 = normalize(
        [4785, 13084, 19884, 19150, 13628, 6448, 3620, 1632, 496, 225, 90, 24, 18])
    test_data_24_05_3_ambient = normalize(
        [25780, 47830, 49853, 33462, 15959, 6724, 2150, 761, 171, 33])
    test_data_004 = normalize([117609.2683, 4098, 373, 16])
    test_data_2 = normalize([3773.464655, 12714.21906, 16774.05141,
                                16591.58704, 12407.57754, 9070, 3644,
                                1648, 615, 180, 70, 49])
    test_data_5 = normalize([2734.396601, 10662.83842, 14854.70766,
                                20760.48099, 16722.99395, 12863.43894,
                                9085, 4613, 1717, 744, 208, 103])
    pm_exp = test_data_5
    ideal_pn_model = ppoisson(mean(pm_exp) / QE, N)
    pnoise = np.zeros(N)
    
elif EXP == 3:
    dir_path = r"I:/MGU/Laser"
    hist, bins = lo.single_pulse_histogram(dir_path)
    pm_exp = lo.hist2pc(hist)

elif EXP == 4:
    #P_CROSSTALK = 0.008017335158230492
    #P_CROSSTALK = 0.008282296845160672
    #P_CROSSTALK = 0.008129662598555697
    #P_CROSSTALK = 0.00442979044775124
    #P_CROSSTALK = 0.005260079621802079
    P_CROSSTALK = 0.00434783902854902 #0.00624033520243927  # Correct
    P_SUM_CT = 0.03168573186533852

    ampl_discrete = 0.03
    pvs_discrete = 2.2e-10
    dir_name = r"../../histograms"
    ampl_data_laser = os.path.join(dir_name, "F4Ampl_laser_only00001.txt")
    pvs_data_laser = os.path.join(dir_name, "F8pVs_laser_only00001.txt")

    ampl_data_lasernlamp = os.path.join(dir_name, "F4Ampl_laser_lamp00000.txt")
    #ampl_discrete = 0.021
    ampl_data_laserlamp_osc5000 = os.path.join(
        dir_name, "rec_hist_laserlamp.txt")
    ampl_data_lamp_osc5000 = os.path.join(
        dir_name, "rec_hist_lamp.txt")
    ampl_data_laserlamp_3ns = os.path.join(
        dir_name, "rec_hist_laserlamp_3.0nsMAX200.txt")
    ampl_data_lasernlamp_5ns = os.path.join(
        dir_name, "rec_hist_laserlamp_5ns.txt")
    ampl_data_lasernlamp_7ns = os.path.join(
        dir_name, "rec_hist_laserlamp_7.5ns.txt")
    ampl_data_lamp_7ns = os.path.join(
        dir_name, "rec_hist_lamp_7.5ns.txt")

    #ampl_discrete = 1
    ampl_data_lamp_correctM = os.path.join(
        dir_name, 'hist_lamp_16.3ns.txt')
    ampl_data_laserlamp_correctM = os.path.join(
        dir_name, 'hist_laserlamp_16.3ns.txt')
    # Offset 150 and skiprows 5 for histograms from oscilloscope
    pm_exp = file2Q(ampl_data_laser,
                    ampl_discrete, skiprows=5)#, offset=150)
    ideal_pn_model = ppoisson(mean(pm_exp) / QE, N)
    pnoise = np.zeros(N)
    pm_processed = compensate(pm_exp, P_CROSSTALK)


if __name__ == "__main__":
    #assert type(pm_exp[0]) in [sp.numbers.Float]

    #pm_exp = pm_exp[pm_exp >= 0]
    M = len(pm_exp)

    mean_pme = mean(pm_exp)
    g2_pme = g2(pm_exp)

    if EXP:
        info('<pm noised> %f' % mean_pme)
        info('g2(pm noised) %f' % g2_pme)
    else:
        info('<pm ideal> %f' % mean_pme)
        info('g2(pm ideal) %f' % g2_pme)

    if EXP:
        if ADJ_CROSSTALK:
            #correct_g2 = g2(P2Q(ss.correct_poisson(mean_pme / QE, N), QE)[:M])
            P_CROSSTALK, _ = optimize_pcrosstalk(pm_exp, QE, N, mtype=MTYPE, n_cells=N_CELLS)
        pm_processed = compensate(pm_exp, P_CROSSTALK)

        plt.plot(pm_exp)
        plt.plot(pm_processed)
        #plt.plot(scc.include_crosstalk_4n(pm_processed, P_CROSSTALK))
        plt.show()

    #pm_processed = pm_processed[pm_processed > 0]
    #M = len(pm_processed)

    g2_pmp = g2(pm_processed)
    mean_pmp = mean(pm_processed)
    info('<pm> %f' % mean_pmp)
    info('g2(pm) %f' % g2_pmp)
    info('QE %f' % QE)

    """ ================= Восстановление статистики фотонов ==============="""

    invpmodel = InvPBaseModel(pm_processed, QE, N, mtype=MTYPE, n_cells=N_CELLS)
    res = invpopt(invpmodel, eps_tol=1e-7)
    pn_rec = res.x
    pm_rec = P2Q(pn_rec, QE, M)
    
    if EXP:
        cdp.plot_exp_data(pm_processed,
                          pn_rec, pm_rec,
                          M, DISPLAY_ERR=True)
    else:
        cdp.plot_model_data(pm_processed, pn_rec,
                            ideal_pn_model, pm_rec,
                            M, QE, DISPLAY_ERR=False)
        delta_pn = np.abs(pn_rec - ideal_pn_model)
        info('<D>[pn] %f\nmax(D)[pn] %f' %
             (np.mean(delta_pn), np.max(delta_pn)))
        info('F[pn] %f' % fidelity(pn_rec, ideal_pn_model))
        delta_pna = np.abs(
            ideal_pn_model - Q2P(pm_rec, QE, N, MTYPE, N_CELLS))
        info('<D>[pn, analytic] %f\nmax(D)[pn, analytic] %f' % (
            np.mean(delta_pna), np.max(delta_pna)))
        info('F[pn, analytic] %f' %
             fidelity(np.abs(Q2P(pm_rec, QE, N, MTYPE, N_CELLS)), ideal_pn_model))

    info('\ng2[pn] %f\ng2[pm exp] %f\ng2[pm rec] %f' % (g2(pn_rec),
                                                        g2_pmp,
                                                        g2(pm_rec)))
    info('F[pm] %f' % fidelity(pm_processed, pm_rec))

    M = len(pm_rec)
    delta_pm = np.abs(pm_processed[:M] - pm_rec)
    info('\n<D>[pm] %f\nmax(D)[pm] %f' % (np.mean(delta_pm),
                                          np.max(delta_pm)))
