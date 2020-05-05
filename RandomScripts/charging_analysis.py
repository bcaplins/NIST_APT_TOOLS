# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:24:21 2019

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

import sel_align_m2q_log_xcorr

import scipy.interpolate


plt.close('all')


nuv_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"


#euv_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos"
euv_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01.epos"
#euv_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07086-v01.epos"
#euv_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07276-v03.epos"


#epos = apt_fileio.read_epos_numpy(nuv_fn)
#plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,1,clearFigure=True,user_ylim=[0,150])


epos = apt_fileio.read_epos_numpy(euv_fn)
#epos = apt_fileio.read_epos_numpy(nuv_fn)
plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,1,clearFigure=True,user_ylim=[0,80]) 

cts_per_slice=2**10
#m2q_roi = [0.9,190]
m2q_roi = [0.8,80]
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr.get_all_scale_coeffs(
                                                        epos['m2q'],
                                                         m2q_roi=m2q_roi,
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.15)
t_end = time.time()
print('Total Time = ',t_end-t_start)

# Compute corrected data
m2q_corr = epos['m2q']/pointwise_scales

wall_time = np.cumsum(epos['pslep'])/10000.0
wall_time = np.cumsum(epos['pslep'])/500000.0

dt = ppd.moving_average(np.diff(wall_time),n=512)
ra = 1/dt





pks = [14,16,32.2,44.3,60.3]
#pks = [14,16,32,44,60]
wid = 1

mask = None
for pk in pks:
    if mask is None:
        mask = np.abs(m2q_corr-pk)<=wid/2
    else:
        mask = mask | (np.abs(m2q_corr-pk)<=wid/2)

fig = plt.figure(num=111)
ax = fig.gca()

ax.plot(wall_time,m2q_corr,'.', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,80])

ax.grid()

fig.tight_layout()
plt.pause(0.1)





hist, edges = np.histogram(wall_time[mask],bins=piecewise_scales.size,range=(0,np.max(wall_time)))
centers = (edges[1:]+edges[0:-1])/2






fig = plt.figure(num=11111)
fig.clear()
ax = fig.gca()

ax.plot(hist,piecewise_scales,'.',markersize=2)
#ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,80])

ax.grid()

fig.tight_layout()
plt.pause(0.1)






fig = plt.figure(num=1111)
fig.clear()
ax = fig.gca()

ax.plot(centers,hist)
#ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,80])

ax2 = ax.twinx()
ax2.plot(wall_time,pointwise_scales,'-', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa',
        color='tab:red')
ax2.set(ylabel='sc')

ax.grid()

fig.tight_layout()
plt.pause(0.1)


#
#
#fig = plt.figure(num=1111123)
#fig.clear()
#ax = fig.gca()
#
#ax.plot(epos['v_dc'],'.',markersize=2)
##ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,80])
#
#ax.grid()
#
#fig.tight_layout()
#plt.pause(0.1)


f = scipy.interpolate.interp1d(wall_time,pointwise_scales,fill_value='extrapolate')
p_q = f(centers)


fig = plt.figure(num=999)
fig.clear()
ax = fig.gca()

ax.plot(p_q,hist,'.')
ax.plot(p_q,hist,'.')


ax.grid()

fig.tight_layout()
plt.pause(0.1)







fig = plt.figure(num=999)
fig.clear()
ax = fig.gca()

ax.plot(wall_time,(pointwise_scales-1)*20+1,'--',label='sc')
ax.plot(centers,hist/500,'-',label='ct rate')

ax.legend()

ax.grid()

fig.tight_layout()
plt.pause(0.1)






