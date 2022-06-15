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

# R20 Constant evaporation rate, incresing E/pulse
#fns = [r"Ga2O3 epos files\R20_28222_69300pA", 
#       r"Ga2O3 epos files\R20_28221_21900pA",
#       r"Ga2O3 epos files\R20_28220_12000pA", 
#       r"Ga2O3 epos files\R20_28216_4000pA",
#       r"Ga2O3 epos files\R20_28224_1740pA",
#        r"Ga2O3 epos files\R20_28218_862pA",
#        r"Ga2O3 epos files\R20_28219_500pA"]

# R20 Constant SV=3800 V, varying E/pulse
#fns = [r"Ga2O3 epos files\R20_28227_68600pA",
#       r"Ga2O3 epos files\R20_28236_68600pA",
#       r"Ga2O3 epos files\R20_28228_22300pA",
#       r"Ga2O3 epos files\R20_28229_12100pA",
#       r"Ga2O3 epos files\R20_28235_12100pA",
#       r"Ga2O3 epos files\R20_28230_3990pA",
#       r"Ga2O3 epos files\R20_28231_1720pA",
#       r"Ga2O3 epos files\R20_28234_1720pA",
#       r"Ga2O3 epos files\R20_28233_505pA"]

# R20 Constant SV=4700 V, varying E/pulse
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
#       r"Ga2O3 epos files\R44_03699_81pJ",
#       r"Ga2O3 epos files\R44_03699_120pJ",
#       r"Ga2O3 epos files\R44_03699_160pJ"]


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
    ys_sm = ppd.moving_average(ys+5e-2,30)
    
    glob_bg = ppd.physics_bg(xs,glob_bg_param)    
    
   
    ax = fig.gca()
    
#    nm = rngedcts[i]
    nm = epos[(epos['m2q']>4) & (epos['m2q']<120)].size
    
    # epos.size
    
    ax.plot(xs,ys_sm/nm*10**(3*i),label=fns[i])
    # ax.plot(xs,glob_bg/nm,label='global bg')
    
    ax.set(xlabel='m/z (Da)', ylabel='~counts')
    ax.grid()
    fig.tight_layout()
    fig.canvas.manager.window.raise_()
    ax.set_yscale('log')    
    ax.legend()

    # plotting_stuff.plot_histo(epos['tof'],200,user_label=fns[i][:-5],clearFigure=False,user_xlim=[0,40000],user_bin_width=100, scale_factor=1, user_color=None)
    
    print(np.mean(epos['v_dc']) )
    
    
    plt.pause(0.1)
#    break 
    
    
# plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,231,clearFigure=True,user_ylim=[0,1200])
# plotting_stuff.plot_histo(epos['tof'],231,user_label='histo',clearFigure=True,user_xlim=[0,40000],user_bin_width=1, scale_factor=1, user_color=None)
    

#### END BASIC ANALYSIS ####

# R20 Constant evaporation rate, incresing E/pulse
#CSR=[1.34,1.18,0.89,0.94,0.46,0.23,0.02]
#conc_Ga=[42.02,45.57,46.49,49.59,54.11,58.69,77.32]
#conc_O=[57.98,54.43,53.51,50.41,45.89,41.31,22.68]
#uncert=[0.39,0.31,0.32,0.28,0.26,0.23,0.16]

# R20 Constant SV=3800 V, varying E/pulse
#CSR=[0.68,0.72,0.66,0.58,0.43,0.19,0.33,0.56,0.52]
#conc_Ga=[51.54,51.11,50.52,49.50,50.48,45.77,50.64,51.33,52.79]
#conc_O=[48.46,48.89,49.48,50.50,49.52,54.23,49.36,48.67,47.21]
#uncert=[0.82,0.91,0.97,1.25,1.33,1.48,1.44,1.00,0.83]

# R20 Constant SV=4700 V, varying E/pulse
#CSR=[1.31,1.32,1.36,1.25,1.25,1.27,1.21,1.24]
#conc_Ga=[42.22,42.88,41.89,42.71,41.06,42.06,42.38,42.71]
#conc_O=[57.78,57.12,58.11,57.29,58.94,57.94,57.62,57.29]
#uncert=[1.05,0.99,1.16,1.25,1.40,1.17,1.06,1.17]

# R44 250 kHz constant evaporation rate = 0.3, varying E/pulse
#CSR=[1.53,1.46,1.41,1.28,1.03,0.85,0.72,0.55,0.43,0.35]
#conc_Ga=[39.25,39.91,40.89,42.53,43.75,44.65,45.08,47.00,49.61,50.91]
#conc_O=[60.75,60.09,59.11,57.47,56.25,55.35,54.92,53.00,50.39,49.09]
#uncert=[0.27,0.23,0.23,0.19,0.21,0.22,0.22,0.22,0.21,0.21]

# R44 25 kHz constant evaporation rate = 0.5, varying E/pulse
#CSR=[0.97,0.92,0.78,0.73,0.61,0.54,0.37,0.42,0.33]
#conc_Ga=[35.13,36.41,40.33,41.70,43.39,44.73,47.89,46.08,49.36]
#conc_O=[64.87,63.59,59.67,58.31,56.61,55.27,52.11,53.92,50.64]
#uncert=[0.48,0.56,0.41,0.41,0.40,0.42,0.39,0.40,0.36]

# R44 25 kHz constant SV=4900 V, varying E/pulse
#CSR=[0.61,0.54,0.66,0.69,0.71,0.86,0.74,0.78]
#conc_Ga=[35.99,37.93,39.24,40.08,43.45,41.12,43.13,45.55]
#conc_O=[64.01,62.07,60.76,59.92,56.55,58.88,56.87,54.45]
#uncert=[1.11,0.90,0.84,0.73,0.78,0.76,0.68,0.72]

# R44 25 kHz constant SV=4900 V, varying E/pulse
CSR=[5.348369283,6.613236189,2.68074861,8.102977617,0.758625752]
conc_Ga=[49.82,49.55,51.34,49.13,57.96]
conc_O=[50.18,50.46,48.66,50.87,42.04]
uncert=[1.11,0.90,0.84,0.73,0.78]

fig = plt.figure(num=200)
fig.clear()
ax = fig.gca()
plt.errorbar(CSR,conc_Ga,yerr=2*np.array(uncert),fmt ='o',label="Al",capsize=5,markersize=4)#,'ko',label="Ga")
plt.errorbar(CSR,conc_O,yerr=2*np.array(uncert),fmt = 'o',label="O",capsize=5,markersize=4)#,'rs',label="O")
plt.legend()
#ax.set(xlabel='m/z (Da)', ylabel='Apparent composition (at. %)')
ax.set(xlabel='Charge State Ratio ($Al^{++}/Al^+$)', ylabel='Apparent composition (at. %)')


xxlim = [0, 10]
yylim = [20, 80]

plt.plot(xxlim, [40,40],linestyle='--',c='#1f77b4')
plt.plot(xxlim, [60,60],linestyle='--',c='#ff7f0e')
#plt.savefig('Ga2O3_CSR_plot.pdf')


ax.set(xlim=xxlim, ylim=yylim)






plt.plot([])


#plt.xscale('log')
plt.legend()

fig.tight_layout()

import sys
sys.exit()
