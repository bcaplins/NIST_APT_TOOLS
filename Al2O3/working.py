# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:12:21 2022

@author: bwc
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

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3
import histogram_functions 




import peak_param_determination as ppd

from histogram_functions import bin_dat
import voltage_and_bowl
from voltage_and_bowl import do_voltage_and_bowl
from voltage_and_bowl import mod_full_vb_correction

import colorcet as cc
# 
# plt.close('all')

fn = r'Jake28300test.epos'

epos = apt_fileio.read_epos_numpy(fn)

plotting_stuff.plot_histo(epos['m2q'],fig_idx=1,user_xlim=[0,200])

plotting_stuff.plot_m2q_vs_time(epos['m2q'], epos, 2)


# Define peaks to range
ed = initElements_P3.initElements()

# Oxygen rich
pk_data =   np.array(    [  (1,     0,      ed['Al'].isotopes[27][0] /3),
                            (1,     0,      ed['Al'].isotopes[27][0] /2),
                            (0,     1,      ed['O'].isotopes[16][0]  /1),
                            (1,     1,      (ed['Al'].isotopes[27][0]+ed['O'].isotopes[16][0])    /2),
                            (2,     1,      (2*ed['Al'].isotopes[27][0]+ed['O'].isotopes[16][0])    /3),
                            (1,     0,      ed['Al'].isotopes[27][0] /1),
                            (2,     2,      (2*ed['Al'].isotopes[27][0]+2*ed['O'].isotopes[16][0])    /3),
                            (0,     2,      2*ed['O'].isotopes[16][0]    /1),
                            (1,     1,      (ed['Al'].isotopes[27][0]+ed['O'].isotopes[16][0])    /1),
                            (1,     2,      (ed['Al'].isotopes[27][0]+2*ed['O'].isotopes[16][0])    /1)
                            ],
                            dtype=[('Al','i4'),('O','i4'),('m2q','f4')] )


# Range the peaks
pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'],peak_height_fraction=0.1)
    
# Determine the global background
glob_bg_param = ppd.get_glob_bg(epos['m2q'])

# Count the peaks, local bg, and global bg
cts = ppd.do_counting(epos,pk_params,glob_bg_param)

# Test for peak S/N and throw out craptastic peaks
B = np.max(np.c_[cts['local_bg'][:,None],cts['global_bg'][:,None]],1)[:,None]
T = cts['total'][:,None]
S = T-B
std_S = np.sqrt(T+B)
# Make up a threshold for peak detection... for the most part this won't matter
# since weak peaks don't contribute to stoichiometry much... except for Mg!
is_peak = S>2*np.sqrt(2*B)
for idx, ct in enumerate(cts):
    if not is_peak[idx]:
        for i in np.arange(len(ct)):
            ct[i] = 0
        
# Calculate compositions
compositions = ppd.do_composition(pk_data,cts)
ppd.pretty_print_compositions(compositions,pk_data)
    
print('Total Ranged Ions: '+str(np.sum(cts['total'])))
print('Total Ranged Local Background Ions: '+str(np.sum(cts['local_bg'])))
print('Total Ranged Global Background Ions: '+str(np.sum(cts['global_bg'])))
print('Total Ions: '+str(epos.size))

# print('Overall CSR (no bg)    : '+str(cts['total'][Si2p_idx]/cts['total'][Si1p_idx]))
# print('Overall CSR (local bg) : '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['local_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['local_bg'][Ga1p_idxs]))))
# print('Overall CSR (global bg): '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['global_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['global_bg'][Ga1p_idxs]))))


# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 100],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,10)

glob_bg = ppd.physics_bg(xs,glob_bg_param)    

fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()

ax.plot(xs,ys_sm,label='hist')
ax.plot(xs,glob_bg,label='global bg')

ax.set(xlabel='m/z (Da)', ylabel='counts')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()
ax.set_yscale('log')    
ax.legend()

for idx,pk_param in enumerate(pk_params):
    
    if is_peak[idx]:
        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
        ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
        ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
        
        ax.plot(np.array([pk_param['pre_bg_rng'],pk_param['post_bg_rng']]) ,np.ones(2)*pk_param['loc_bg'],'g--')
    else:
        ax.plot(np.array([1,1])*pk_param['x0_mean_shift'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
        

        
plt.pause(0.1)

