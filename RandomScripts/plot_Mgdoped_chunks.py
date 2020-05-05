# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat

import colorcet as cc



# Read in data
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg doped
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbmq_corr.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\190421_AlGaN50p7_A83\R20_07209-v01.epos"
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07167-v03.epos"
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07148-v02.epos"


#fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
roi_event_idxs = np.arange(1000,epos.size-1000)

ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,fig_idx=1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
LASER_REP_RATE = 10000.0
wall_time = np.cumsum(epos['pslep']/LASER_REP_RATE)
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)


num_chunks = 4
wall_time_edges = np.linspace(wall_time[0],wall_time[-1],int(num_chunks+1))




fig = plt.figure(num=100)
fig.clear()
ax = fig.gca()



for idx in np.arange(wall_time_edges.size-1):
    start_idx = (np.abs(wall_time - wall_time_edges[idx])).argmin()
    end_idx = (np.abs(wall_time - wall_time_edges[idx+1])).argmin()
    
    sub_epos = epos[start_idx:end_idx]
    
    # Plot all the things
    xs, ys = bin_dat(sub_epos['m2q'],user_roi=[0.5, 100],isBinAligned=True)
#    ys_sm = ppd.do_smooth_with_gaussian(ys,10)
    ys_sm = ppd.moving_average(ys,30)

    
    color = (*cc.cm.CET_L8(idx/(num_chunks-1))[0:3],0.5)

#    ax.plot(xs,(10**idx)*ys_sm,label='i='+str(roi_edges[idx])+':'+str(roi_edges[idx+1]))
    ax.plot(xs,(1**idx)*ys_sm,
            label='i='+str(start_idx)+':'+str(end_idx),
            color=color)
    plt.pause(1)


ax.set(xlabel='m/z (Da)', ylabel='counts')
ax.grid()
ax.legend()

ax.set_yscale('log')    
fig.tight_layout()




plt.pause(0.1)




#### END BASIC ANALYSIS ####
