# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:08:56 2019

@author: bwc
"""


# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff

import peak_param_determination as ppd

from histogram_functions import bin_dat
import scipy.interpolate
import image_registration.register_images

import sel_align_m2q_log_xcorr_v2

import scipy.interpolate
import time
import m2q_calib
import initElements_P3

from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl 

import colorcet as cc
import matplotlib._color_data as mcd

import pandas as pd




plt.close('all')

fn = r'C:\Users\bwc\Documents\NetBeansProjects\R45_04472\recons\recon-v01\default\R45_04472-v01.epos'
epos = apt_fileio.read_epos_numpy(fn)



# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Only apply bowl correction
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])


cts_per_slice=2**7
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr_v2.get_all_scale_coeffs(tof_bcorr,
                                                         m2q_roi=[50,1000],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.15)
t_end = time.time()
print('Total Time = ',t_end-t_start)




laser_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R45_04472\R45_04472_LaserPositionHist.csv')

lion = laser_df['Ion Sequence Number'].values*((epos.size-1)/laser_df['Ion Sequence Number'].values.max())
lx = laser_df['Laser X Position (mic)'].values
ly = laser_df['Laser Y Position (mic)'].values
lz = laser_df['Laser Z Position (mic)'].values







fig = plt.figure(num=111)
fig.clear()
ax = fig.gca()

ax.plot(epos['m2q'],'.', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='ion #', ylabel='m/z', ylim=[0,70])

ax.grid()






laser_pulse_idx = np.zeros(np.cumsum(epos['pslep']).max())
laser_pulse_idx[np.cumsum(epos['pslep'])-1] = 1

def moving_average(a, n=3) :
    # actual filter is 2*n+1 width
    a = np.r_[0,a[n:0:-1],a,a[-2:-(n+2):-1]]
    ret = np.cumsum(a, dtype=float)
    ret = (ret[2*n+1:] - ret[:-(2*n+1)])/(2*n+1)
    return ret

er = moving_average(laser_pulse_idx,n=500000)
er_samp = er[np.cumsum(epos['pslep'])-1]




fig = plt.figure(num=222)
fig.clear()
ax = fig.gca()


ax.plot(er_samp)
ax2 = ax.twinx()
ax2.plot(pointwise_scales)



fig = plt.figure(num=333)
fig.clear()
ax = fig.gca()
ax.plot(er_samp,pointwise_scales,'.')





ax.plot( (epos['v_dc']-np.mean(epos['v_dc']))/1000 )
ax.plot((pointwise_scales-1)*20)

ax2 = ax.twinx()

ax2.plot(lion,lx)
ax2.plot(lion,ly)
ax2.plot(lion,lz)



