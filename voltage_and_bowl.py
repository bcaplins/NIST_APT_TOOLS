# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import voltage_correction as v_corr
import bowl_correction as b_corr

import numpy as np

def do_voltage_and_bowl(epos,p_volt,p_bowl):

    TOF_BIN_SIZE = 1.0
    ROI = np.array([200, 1000])
    
    p_volt = v_corr.basic_voltage_correction(epos['tof'],epos['v_dc'],p_volt,ROI, TOF_BIN_SIZE)
    tof_vcorr = v_corr.mod_basic_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
    
    p_bowl = b_corr.geometric_bowl_correction(tof_vcorr,epos['x_det'],epos['y_det'],p_bowl,ROI, TOF_BIN_SIZE)
    tof_bcorr = b_corr.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
    
    TOF_BIN_SIZES = [0.125]
    
    for tof_bin_size in TOF_BIN_SIZES:
        p_volt = v_corr.full_voltage_correction(tof_bcorr,epos['v_dc'],p_volt,ROI,tof_bin_size)
        tof_vcorr = v_corr.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
        
    #    plot_vs_time()
    #    plot_vs_time_kde()
    #    plot_histos()
    
        p_bowl = b_corr.geometric_bowl_correction(tof_vcorr,epos['x_det'],epos['y_det'],p_bowl,ROI, tof_bin_size)
        tof_bcorr = b_corr.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])

    #    plot_vs_radius()
    #    plot_vs_time_kde()
    #    plot_histos()
        
    tof_vcorr = v_corr.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
    tof_corr = b_corr.mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])

    #plot_vs_time()
    #plot_vs_radius()
    #plot_histos()
    #plot_vs_time_kde()
    
    return (tof_corr, p_volt, p_bowl)
