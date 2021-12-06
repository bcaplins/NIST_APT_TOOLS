# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:53:25 2019

@author: bwc
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

plt.close('all')

# Load data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbm_corr.epos"
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"

#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07086-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07276-v03.epos"


epos = apt_fileio.read_epos_numpy(fn)
plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,1,clearFigure=True,user_ylim=[0,150])

# Create histogram

cts_per_histo = 2**10
nx = int(epos.size/cts_per_histo)
xs = np.arange(epos.size)

y_roi = np.log10(np.array([0.5, 200]))
ny = 2**12
ys = np.log10(epos['m2q'])

N,x_edges,y_edges = np.histogram2d(xs,ys,bins=[nx,ny],range=[[1,epos.size],y_roi],density=False)



def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

# Plot histogram

fig = plt.figure(figsize=(8,8))
plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
           extent=extents(x_edges) + extents(y_edges), origin='lower')

#fig = plt.figure(figsize=(8,8))
#plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
#           origin='lower')
#
#


#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#
x_centers = (x_edges[0:-1]+x_edges[1:])/2
#
y_centers = (y_edges[0:-1]+y_edges[1:])/2
#
#for i in np.arange(0,N.shape[0],100):
#    ax.plot(y_centers,N[i,:])


import image_registration
def get_shifts(ref,N):
    shifts = np.zeros(N.shape[0])
    im1 = ref
    for i in np.arange(N.shape[0]):
        im2 = N[i:i+1,:]
        dx,dy = image_registration.register_images(im1,im2,usfac=32,maxoff=150)
        shifts[i] = dx
#        print(dx,dy)
    shifts = shifts - np.mean(shifts)
    return shifts
    

ref = np.mean(N[N.shape[0]//2-10:N.shape[0]//2+11,:],axis=0)[None,:]

shifts0 = get_shifts(ref,N)
    
   

import scipy 
f = scipy.interpolate.interp1d(x_centers,shifts0,fill_value='extrapolate')
all_shifts = f(np.arange(ys.size))
ys_corr = ys.copy() - all_shifts*(y_edges[1]-y_edges[0])

N,x_edges,y_edges = np.histogram2d(xs,ys_corr,bins=[nx,ny],range=[[1,epos.size],y_roi],density=False)
fig = plt.figure(figsize=(8,8))
plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
           extent=extents(x_edges) + extents(y_edges), origin='lower')


ref = np.mean(N[N.shape[0]//4:3*N.shape[0]//4,:],axis=0)[None,:]
shifts1 = get_shifts(ref,N)
f = scipy.interpolate.interp1d(x_centers,shifts1,fill_value='extrapolate')
all_shifts += f(np.arange(ys.size))

ys_corr = ys.copy() - all_shifts*(y_edges[1]-y_edges[0])

N,x_edges,y_edges = np.histogram2d(xs,ys_corr,bins=[nx,ny],range=[[1,epos.size],y_roi],density=False)
fig = plt.figure(figsize=(8,8))
plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
           extent=extents(x_edges) + extents(y_edges), origin='lower')

    
    
scales = 10.0**(all_shifts*(y_edges[1]-y_edges[0]))    

scales_short = 10.0**((shifts0+shifts1)*(y_edges[1]-y_edges[0]))    

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.plot(scales_short,'o-')


m2q_corr = epos['m2q']/scales

plotting_stuff.plot_TOF_vs_time(m2q_corr,epos,111,clearFigure=True,user_ylim=[0,150])


plotting_stuff.plot_histo(m2q_corr,222,user_xlim=[0, 200])
plotting_stuff.plot_histo(epos['m2q'],222,user_xlim=[0, 200],clearFigure=False)

#
#dsc = np.r_[np.diff(scales_short),0]
#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#ax.plot(dsc)
#
#fig = plt.figure(figsize=(8,8))
#ax = fig.gca()
#
#ax.hist(dsc,bins=512,range=[-0.01,0.01])
#dsc_cut = scipy.stats.trimboth(dsc,0.025)
#
#outlier_lims = [np.mean(dsc_cut)-5*np.std(dsc_cut), np.mean(dsc_cut)-5*np.std(dsc_cut)]
#
#for i in np.arange(dsc.size):
#    isBad = (dsc[i]<=outlier_lims[0]) | (dsc[i]>=outlier_lims[1])
#    if isBad:
#        N[i,:] = 0
#

