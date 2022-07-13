# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

import colorcet as cc
cm = cc.cm.rainbow_r

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




fn_old = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07071-v01.epos"
fn_new = r"SiO2 epos files\R20_28199-200nm.epos"

fig = plt.figure(figsize=(18,6), num=101)
fig.clear()


fn = fn_old[:-5]+'_vbmq_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)

roi_event_idxs = np.arange(1000,epos.size-1000)

#roi_event_idxs = np.arange(epos.size)

epos = epos[roi_event_idxs]

# Compute some extra information from epos information
LASER_REP_RATE = 10000.0
wall_time = np.cumsum(epos['pslep'])/LASER_REP_RATE
print('total time='+str(wall_time[-1]))
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Determine the global background
glob_bg_param = ppd.get_glob_bg(epos['m2q'])

# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0, 70],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,30)*30

glob_bg = ppd.physics_bg(xs,glob_bg_param)    


ax = fig.gca()

# nm = rngedcts[i]
nm = 1
# nm = 40340

# epos.size



ax.plot(xs,ys_sm/nm,label=fn_old[:-5], color='blue')
# ax.plot(xs,glob_bg/nm,label='global bg')

ax.set(xlabel='m/z (Da)', ylabel='~counts')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()
ax.set_yscale('log')    
# ax.legend()

# plotting_stuff.plot_histo(epos['tof'],200,user_label=fns[i][:-5],clearFigure=False,user_xlim=[0,40000],user_bin_width=100, scale_factor=1, user_color=None)

print(np.mean(epos['v_dc']) )


plt.pause(0.1)


ax.set_xlim(0,70)
ax.set_ylim(1e-0,1e5)

ax.xaxis.grid(False)
fig.tight_layout()
    



fig = plt.figure(figsize=(18,6), num=102)
fig.clear()


fn = fn_new[:-5]+'_vbmq_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)

roi_event_idxs = np.arange(1000,epos.size-1000)

#roi_event_idxs = np.arange(epos.size)
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
LASER_REP_RATE = 25000.0
wall_time = np.cumsum(epos['pslep'])/LASER_REP_RATE
print('total time='+str(wall_time[-1]))
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Determine the global background
glob_bg_param = ppd.get_glob_bg(epos['m2q'])

# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0, 70],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,30)*30

glob_bg = ppd.physics_bg(xs,glob_bg_param)    


ax = fig.gca()

# nm = rngedcts[i]
nm = 1
# nm = 204811

# epos.size


ax.plot(xs,ys_sm/nm,label=fn_new[:-5], color='red')
# ax.plot(xs,glob_bg/nm,label='global bg')

ax.set(xlabel='m/z (Da)', ylabel='~counts')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()
ax.set_yscale('log')    
# ax.legend()

# plotting_stuff.plot_histo(epos['tof'],200,user_label=fns[i][:-5],clearFigure=False,user_xlim=[0,40000],user_bin_width=100, scale_factor=1, user_color=None)

print(np.mean(epos['v_dc']) )


plt.pause(0.1)


ax.set_xlim(0,70)
ax.set_ylim(1e-0,1e5)

ax.xaxis.grid(False)
ax.yaxis.grid(True)
fig.tight_layout()

# plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,231,clearFigure=True,user_ylim=[0,1200])
# plotting_stuff.plot_histo(epos['tof'],231,user_label='histo',clearFigure=True,user_xlim=[0,40000],user_bin_width=1, scale_factor=1, user_color=None)
    

#### END BASIC ANALYSIS ####


import sys
sys.exit()
