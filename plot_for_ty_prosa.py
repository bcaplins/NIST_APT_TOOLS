# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:11:59 2019

@author: bwc
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff

import peak_param_determination as ppd

import scipy.interpolate


from sel_align_m2q_log_xcorr import * 


plt.close('all')

# Load data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbm_corr.epos"
#fn = r"D:\Users\clifford\Documents\Python Scripts\NIST_DATA\R20_07094-v03.epos"
#fn = r"D:\Users\clifford\Documents\Python Scripts\NIST_DATA\R45_04472-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07086-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07276-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02.epos"
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"


epos = apt_fileio.read_epos_numpy(fn)
#plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,1,clearFigure=True,user_ylim=[0,150])

#epos = epos[0:100000]

cts_per_slice=2**8
#m2q_roi = [0.9,190]
m2q_roi = [10,100]
import time
t_start = time.time()
pointwise_scales,piecewise_scales = get_all_scale_coeffs(epos,
                                                         m2q_roi=m2q_roi,
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.1)
t_end = time.time()
print('Total Time = ',t_end-t_start)




#lys_corr = np.log(epos['m2q'])-np.log(pointwise_scales)
#N,x_edges,ly_edges = create_histogram(lys_corr,y_roi=m2q_roi,cts_per_slice=cts_per_slice)

## Plot histogram
#fig = plt.figure(figsize=(7,4.5))
#plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
#           extent=extents(x_edges) + extents(ly_edges), origin='lower')


   
    
fig = plt.figure(figsize=(7,4.5))
ax = fig.gca()
#ax.plot(piecewise_scales,'-')

m2q_corr = epos['m2q']/pointwise_scales
ax = plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,111,clearFigure=True,user_ylim=[0,75])

ax.set_ylabel('m/z (Da)')


import pandas as pd


fig = plt.figure(figsize=(7,4.5))
ax = fig.gca()

df = pd.read_csv(r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472_LaserPositionHist.csv")
ax.plot(df['Ion Sequence Number'],df['Laser X Position (mic)'])
ax.plot(df['Ion Sequence Number'],df['Laser Y Position (mic)'])
ax.plot(df['Ion Sequence Number'],df['Laser Z Position (mic)'])



ax.plot(np.arange(pointwise_scales.size)-(pointwise_scales.size-519841) ,  (pointwise_scales-1)*100/5-2)
ax.grid()



fig = plt.figure(figsize=(7,4.5))
ax = fig.gca()

ax.clear()
ax.plot(pointwise_scales)
ax.grid()
ax.plot(np.convolve(np.diff(epos['v_dc']),np.ones(11)/11,mode='same')+1)
#ax.plot(np.convolve(np.diff(epos['v_dc']),np.ones(1)/1,mode='same')+1)

A = pointwise_scales-np.mean(pointwise_scales)
B = np.r_[0,np.diff(epos['v_dc'])]
B = B-np.mean(B)

C = np.convolve(A,B[::-1],'same')

fig = plt.figure(figsize=(7,4.5))
ax = fig.gca()

ax.plot(C)

#
#
#
#wall_time = np.cumsum(epos['pslep'])/500000.0
#
#dt = ppd.moving_average(np.diff(wall_time),n=512)
#ra = 1/dt
#
#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#ax.plot(ra)
#
#dsc_pointwise = np.r_[np.diff(pointwise_scales),0]
#
##ax.plot(dsc_pointwise*10000000)
#
#ax.plot((pointwise_scales-1)*100000/3)
#
#ax.plot((pointwise_scales-1)*100/3)
#
#
#
#
#
#plotting_stuff.plot_TOF_vs_time(m2q_corr,epos,222,clearFigure=True,user_ylim=[0,250])
#
#
#plotting_stuff.plot_histo(m2q_corr,333,user_xlim=[0, 200])
#plotting_stuff.plot_histo(epos['m2q'],333,user_xlim=[0, 200],clearFigure=False)
##
#
#dsc = np.r_[np.diff(piecewise_scales),0]
#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#ax.plot(dsc)
#
#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#
#ax.hist(dsc,bins=64,range=[-0.01,0.01])
#dsc_cut = scipy.stats.trimboth(dsc,0.025)
#
#outlier_lims = [np.mean(dsc_cut)-7*np.std(dsc_cut), np.mean(dsc_cut)+7*np.std(dsc_cut)]