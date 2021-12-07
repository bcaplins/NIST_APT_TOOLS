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
fns = [r"SiO2 epos files\R20_28199-200nm.epos", 
       r"SiO2 epos files\R20_28197-400nm.epos",
       r"SiO2 epos files\R20_28199-600nm.epos", 
       r"SiO2 epos files\R20_28197-1000nm.epos",
        r"SiO2 epos files\R20_28197-1200nm.epos"]#,
        # r"SiO2 epos files\R20_28199-1000nm.epos"]

# This is the number of ranged counts in the above files
# it is used for spectrum normalization below.
rngedcts = [506050, 371085, 319139, 259340, 204811]


fig = plt.figure(num=100)
fig.clear()

# fig200 = plt.figure(num=200)
# fig200.clear()

for i in range(len(fns)):
    fn = fns[i][:-5]+'_vbmq_corr.epos'
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
    xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 80],isBinAligned=True)
    #ys_sm = ppd.do_smooth_with_gaussian(ys,10)
    ys_sm = ppd.moving_average(ys,30)
    
    glob_bg = ppd.physics_bg(xs,glob_bg_param)    
    
   
    ax = fig.gca()
    
    nm = rngedcts[i]
    # epos.size
    
    ax.plot(xs,ys_sm/nm,label=fns[i][:-5])
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
    
    
    
    
# plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,231,clearFigure=True,user_ylim=[0,1200])
# plotting_stuff.plot_histo(epos['tof'],231,user_label='histo',clearFigure=True,user_xlim=[0,40000],user_bin_width=1, scale_factor=1, user_color=None)
    

#### END BASIC ANALYSIS ####


import sys
sys.exit()
