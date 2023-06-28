# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:03:27 2022

@author: bwc
"""

import pandas as pd

def panda_read_rrng(f):
    """Loads a .rrng file produced by IVAS. Returns two dataframes of 'ions'
    and 'ranges'."""
    # Credit oscarbranson
    # https://github.com/oscarbranson/apt-tools/blob/master/apt_importers.py
    import re

    rf = open(f,'r').readlines()

    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')

    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])

    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True)

    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)

    return ions,rrngs

def convert_to_pk_range_pk_data(ions, rrngs):    
    dtype_spec = []
    for name in ions.name:
        dtype_spec.append((name,'i4'))
    dtype_spec.append(('m2q','f4'))            
    
    pk_data = np.full(rrngs.shape[0], -1, dtype=dtype_spec)
    
    pk_params = np.full(rrngs.shape[0], -1, dtype=[('pre_rng','f4'),
                                                ('post_rng','f4')])
    
    for idx in np.arange(rrngs.shape[0]):
        row = rrngs.iloc[idx]
        pk_params['pre_rng'][idx] = row['lower']
        pk_params['post_rng'][idx] = row['upper']
        
        pk_data['m2q'][idx] = 0.5*(row['lower']+row['upper'])
        
        comp_str = row['comp']
        
        for name in ions.name:
            substr = name+':'            
            loc_idx = comp_str.find(substr)
            if loc_idx<0:
                pk_data[name][idx] = 0
            else:
                pk_data[name][idx] = comp_str[len(substr)+loc_idx]
                    
    
    
    return pk_data, pk_params



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

import peak_param_determination as ppd

import colorcet as cc



import pandas as pd

d = pd.read_csv(r"C:\Users\bwc\Downloads\R20_28299_S9_MS.csv")


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
xs = np.asarray(d.loc[:,"Mass to Charge State Ratio (Da)"])+5e-4
ys = np.asarray(d.loc[:," Uncorrected Count"])
pk_params = ppd.get_peak_ranges_csv(xs, ys ,pk_data['m2q'],peak_height_fraction=0.01)

# Determine the global background
glob_bg_param = ppd.get_glob_bg_csv(xs, ys)

# Count the peaks, local bg, and global bg
cts = ppd.do_counting_csv(xs, ys,pk_params,glob_bg_param)

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
print('Total Ions (in CSV... prob. to 100 m/z): '+str(ys.sum()))

# print('Overall CSR (no bg)    : '+str(cts['total'][Si2p_idx]/cts['total'][Si1p_idx]))
# print('Overall CSR (local bg) : '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['local_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['local_bg'][Ga1p_idxs]))))
# print('Overall CSR (global bg): '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['global_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['global_bg'][Ga1p_idxs]))))


# Plot all the things
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




####################

pk_data, pk_params = convert_to_pk_range_pk_data(*panda_read_rrng(r"C:\Users\bwc\Downloads\R20_28295.RRNG"))


# Count the peaks, local bg, and global bg
cts = ppd.do_counting_csv_simple(xs, ys,pk_params)
bg_cts = ppd.do_counting_csv_simple(xs, glob_bg,pk_params)
cts['global_bg'] = bg_cts['total']
cts['local_bg'] = cts['total']-1

# Calculate compositions
compositions = ppd.do_composition(pk_data,cts)
ppd.pretty_print_compositions(compositions,pk_data)
    
print('Total Ranged Ions: '+str(np.sum(cts['total'])))
print('Total Ions (in CSV... prob. to 100 m/z): '+str(ys.sum()))

# print('Overall CSR (no bg)    : '+str(cts['total'][Si2p_idx]/cts['total'][Si1p_idx]))
# print('Overall CSR (local bg) : '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['local_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['local_bg'][Ga1p_idxs]))))
# print('Overall CSR (global bg): '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['global_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['global_bg'][Ga1p_idxs]))))


# Plot all the things
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,10)

   

fig = plt.figure(num=102)
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
    
    ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,ys_sm.max()]),'k--')
    ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,ys_sm.max()]),'k--')        
    
        

        
plt.pause(0.1)








