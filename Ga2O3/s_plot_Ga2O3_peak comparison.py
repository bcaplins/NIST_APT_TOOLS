# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
import os
parent_path = '..\\nistapttools'
if parent_path not in sys.path:
    sys.path.append(os.path.abspath(parent_path))    
import time

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat

# Read in data
# Long ion count ~ 4M ions
fns = [r"Ga2O3 epos files\R20_28215_3800pA"]

# Constant evaporation rate, incresing E/pulse
#fns = [r"Ga2O3 epos files\R20_28222_69300pA", 
#       r"Ga2O3 epos files\R20_28221_21900pA",
#       r"Ga2O3 epos files\R20_28220_12000pA", 
#       r"Ga2O3 epos files\R20_28216_4000pA",
#       r"Ga2O3 epos files\R20_28224_1740pA",
#        r"Ga2O3 epos files\R20_28218_862pA",
#        r"Ga2O3 epos files\R20_28219_500pA"]

# Constant SV=3800 V, varying E/pulse
#fns = [r"Ga2O3 epos files\R20_28227_68600pA",
#       r"Ga2O3 epos files\R20_28228_22300pA",
#       r"Ga2O3 epos files\R20_28229_12100pA",
#       r"Ga2O3 epos files\R20_28230_3990pA",
#       r"Ga2O3 epos files\R20_28231_1720pA",
#       r"Ga2O3 epos files\R20_28233_505pA",
#       r"Ga2O3 epos files\R20_28234_1720pA",
#       r"Ga2O3 epos files\R20_28235_12100pA",
#       r"Ga2O3 epos files\R20_28236_68600pA"]

# Constant SV=4700 V, varying E/pulse
#fns = [r"Ga2O3 epos files\R20_28254_11500pA",
#       r"Ga2O3 epos files\R20_28261_11500pA",
#       r"Ga2O3 epos files\R20_28253_4300pA",
#       r"Ga2O3 epos files\R20_28262_4300pA",
#       r"Ga2O3 epos files\R20_28255_1770pA",
#       r"Ga2O3 epos files\R20_28260_1770pA",
#       r"Ga2O3 epos files\R20_28256_925pA",
#       r"Ga2O3 epos files\R20_28259_925pA",
#       r"Ga2O3 epos files\R20_28257_500pA",
#       r"Ga2O3 epos files\R20_28258_265pA"]

#Si-implanted non annealed
#fns = [r"Ga2O3 epos files\R20_18170",
#       r"Ga2O3 epos files\R44_03672"]

# R44 250 kHz constant evaporation rate = 0.3, varying E/pulse
#fns = [r"Ga2O3 epos files\R44_03695_200fJ",
#       r"Ga2O3 epos files\R44_03695_500fJ",
#       r"Ga2O3 epos files\R44_03695_1pJ",
#       r"Ga2O3 epos files\R44_03695_5pJ",
#       r"Ga2O3 epos files\R44_03695_10pJ",
#       r"Ga2O3 epos files\R44_03695_20pJ",
#       r"Ga2O3 epos files\R44_03695_40pJ",
#       r"Ga2O3 epos files\R44_03695_80pJ",
#       r"Ga2O3 epos files\R44_03695_120pJ",
#       r"Ga2O3 epos files\R44_03695_160pJ"]

# R44 25 kHz constant evaporation rate = 0.5, varying E/pulse
#fns = [r"Ga2O3 epos files\R44_03696_500fJ",
#       r"Ga2O3 epos files\R44_03696_1pJ",
#       r"Ga2O3 epos files\R44_03696_5pJ",
#       r"Ga2O3 epos files\R44_03697_10pJ",
#       r"Ga2O3 epos files\R44_03697_20pJ",
#       r"Ga2O3 epos files\R44_03697_40pJ",
#       r"Ga2O3 epos files\R44_03697_80pJ",
#       r"Ga2O3 epos files\R44_03698_120pJ",
#       r"Ga2O3 epos files\R44_03698_160pJ"]

# R44 25 kHz constant SV=4900 V, varying E/pulse
#fns = [r"Ga2O3 epos files\R44_03699_1pJ",
#       r"Ga2O3 epos files\R44_03699_5pJ",
#       r"Ga2O3 epos files\R44_03699_10pJ",
#       r"Ga2O3 epos files\R44_03699_20pJ",
#       r"Ga2O3 epos files\R44_03699_40pJ",
#       r"Ga2O3 epos files\R44_03699_80pJ",
##       r"Ga2O3 epos files\R44_03699_81pJ",
#       r"Ga2O3 epos files\R44_03699_120pJ",
#       r"Ga2O3 epos files\R44_03699_160pJ"]

# Comparison high E/pulse NUV vs EUV constant SV
#fns = [r"Ga2O3 epos files\R20_28254_11500pA",
#       r"Ga2O3 epos files\R44_03699_160pJ"]

# Comparison low E/pulse NUV vs EUV constant SV
#fns = [r"Ga2O3 epos files\R20_28257_500pA",
#       r"Ga2O3 epos files\R44_03699_1pJ"]

# This is the number of ranged counts in the above files
# it is used for spectrum normalization below.
#rngedcts = [506050, 371085, 319139, 259340, 204811,259340, 204811,259340, 204811]


fig = plt.figure(num=100)
fig.clear()

# fig200 = plt.figure(num=200)
# fig200.clear()

for i in range(len(fns)):
    fn = fns[i]+'_vbmq_corr.epos'
    epos = apt_fileio.read_epos_numpy(fn)
    #epos = epos[epos.size//2:-1]
    
    # Plot m2q vs event index and show the current ROI selection
    roi_event_idxs = np.arange(1000,epos.size-1000)
    
    #roi_event_idxs = np.arange(epos.size)
    epos = epos[roi_event_idxs]
    
    # Compute some extra information from epos information
    LASER_REP_RATE = 25000.0
    wall_time = np.cumsum(epos['pslep'])/LASER_REP_RATE
    pulse_idx = np.arange(0,epos.size)
    isSingle = np.nonzero(epos['ipp'] == 1)
    
    # Determine the global background
    glob_bg_param = ppd.get_glob_bg(epos['m2q'])
    
    # Plot all the things
    xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 120],isBinAligned=True)
    #ys_sm = ppd.do_smooth_with_gaussian(ys,10)
    ys_sm = ppd.moving_average(ys+5e-2,10)
    
    glob_bg = ppd.physics_bg(xs,glob_bg_param)    
    
   
    ax = fig.gca()
    
    #Normalization by integrating ranged ions between 4 - 120 m/z
#    nm = rngedcts[i]
    nm = epos[(epos['m2q']>4) & (epos['m2q']<120)].size
    
    # epos.size
    
    ax.plot(xs,ys_sm/nm,label=fns[i])
    # ax.plot(xs,glob_bg/nm,label='global bg')
    
#    ax.set(xlim=[15,19])
#    ax.set(xlim=[31,37])
    ax.set(xlabel='m/z', ylabel='~counts')
    ax.grid()
    fig.tight_layout()
    fig.canvas.manager.window.raise_()
    ax.set_yscale('log')    
    ax.legend()
#    plt.savefig('Ga2O3 mass spectra.pdf')

    # plotting_stuff.plot_histo(epos['tof'],200,user_label=fns[i][:-5],clearFigure=False,user_xlim=[0,40000],user_bin_width=100, scale_factor=1, user_color=None)
    
    print(np.mean(epos['v_dc']) )
    
    
    plt.pause(0.1)
#    break 
    
    
# plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,231,clearFigure=True,user_ylim=[0,1200])
# plotting_stuff.plot_histo(epos['tof'],231,user_label='histo',clearFigure=True,user_xlim=[0,40000],user_bin_width=1, scale_factor=1, user_color=None)
    

#### END BASIC ANALYSIS ####

#CSR=[1.34,1.18,0.89,0.94,0.46,0.23,0.02]
#conc_Ga=[42.02,45.57,46.49,49.59,54.28,58.99,77.66]
#conc_O=[57.98,54.43,53.51,50.41,45.72,41.01,22.34]
#uncert=[0.39,0.31,0.32,0.28,0.26,0.23,0.16]
#
#fig = plt.figure(num=200)
#fig.clear()
#ax = fig.gca()
#plt.errorbar(CSR,conc_Ga,yerr=2*np.array(uncert),fmt ='o',label="Ga",capsize=5,markersize=4)#,'ko',label="Ga")
#plt.errorbar(CSR,conc_O,yerr=2*np.array(uncert),fmt = 'o',label="O",capsize=5,markersize=4)#,'rs',label="O")
#plt.legend()
##ax.set(xlabel='m/z (Da)', ylabel='Apparent composition (at. %)')
#ax.set(xlabel='Charge State Ratio ($Ga^{++}/Ga^+$)', ylabel='Apparent composition (at. %)')
#
#
#xxlim = [0, 1.4]
#yylim = [20, 80]
#
#plt.plot(xxlim, [40,40],linestyle='-',c='#1f77b4')
#plt.plot(xxlim, [60,60],linestyle='-',c='#ff7f0e')
#
#
#ax.set(xlim=xxlim, ylim=yylim)






#plt.plot([])


#plt.xscale('log')
plt.legend()

fig.tight_layout()

import sys
sys.exit()
