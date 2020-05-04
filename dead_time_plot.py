# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:17:34 2020

@author: bwc
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio

import peak_param_determination as ppd

from histogram_functions import bin_dat
import scipy.interpolate
import image_registration.register_images

#import sel_align_m2q_log_xcorr

from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl 

import colorcet as cc
import matplotlib._color_data as mcd

fn = r'Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos'
#fn = r'Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01_vbmq_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)



sing_tof = epos['tof'][epos['ipp']==1]
sing_m2q = epos['m2q'][epos['ipp']==1]
sing_x = epos['x_det'][epos['ipp']==1]
sing_y = epos['y_det'][epos['ipp']==1]
sing_ipp = epos['ipp'][epos['ipp']==1]


fig = plt.figure(num=10)
plt.clf()
ax = fig.gca()

#ax.plot(sing_tof[0:-1],sing_tof[1:],',')
ax.plot(sing_m2q[0:-1],sing_m2q[1:],',')


from matplotlib.colors import LogNorm

ax.hist2d(sing_m2q[0:-1],sing_m2q[1:],bins=(np.arange(0,75,0.01), np.arange(0,75,0.01)),norm=LogNorm())





mult_tof = epos['tof'][epos['ipp']!=1]
mult_x = epos['x_det'][epos['ipp']!=1]
mult_y = epos['y_det'][epos['ipp']!=1]
mult_ipp = epos['ipp'][epos['ipp']!=1]

hit1 = mult_tof[np.where(mult_ipp==2)[0]]
hit2 = mult_tof[np.where(mult_ipp==2)[0]+1]

x1 = mult_x[np.where(mult_ipp==2)[0]]
x2 = mult_x[np.where(mult_ipp==2)[0]+1]

y1 = mult_y[np.where(mult_ipp==2)[0]]
y2 = mult_y[np.where(mult_ipp==2)[0]+1]


del_tof = hit2-hit1
del_x = x2-x1
del_y = y2-y1

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(num=1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')

idxs = np.where(np.abs(del_tof)<10.25)[0]


ax.scatter3D(del_x[idxs], del_y[idxs], del_tof[idxs],'.',color=list(plt.cm.inferno(del_tof[idxs]/np.max(del_tof[idxs]))))



idxs = idxs = np.where((del_x**2+del_y**2)<10**2)[0]


fig = plt.figure(num=2)
plt.clf()
ax = fig.gca()

ax.hist(del_tof[idxs], bins=np.arange(-20,20,0.25))


