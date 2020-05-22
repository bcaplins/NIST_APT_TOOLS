# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:53:09 2020

@author: bwc
"""

# importing matplotlib modules 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
from scipy import interpolate

# Read Images and downsampel to reasonable size
dat = mpimg.imread('moon.jpg')[:,:,0].astype('float64')
dat = dat[0::8,0::8]


# Easy method to locate center.  Can get fancier if needed
def loc_center(dat):
    thresh = 0.02
    i0 = int(dat.shape[0]/2)
    j0 = 0
    for loop in np.arange(3):
#        print(i0)
        prof = dat[i0,:]    
        prof = prof-np.min(prof)
        prof = prof/np.max(prof)
        
        j0 = int(0.5*(np.where(prof>thresh)[0].min()+np.where(prof>thresh)[0].max()))
#        print(j0)
        prof = dat[:,j0]    
        prof = prof-np.min(prof)
        prof = prof/np.max(prof)
        
        i0 = int(0.5*(np.where(prof>thresh)[0].min()+np.where(prof>thresh)[0].max()))
    return (i0,j0)

# Polar coord to cartesian
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

# Locate center
i0,j0 = loc_center(dat)

# Generate pixel coordinates
i = np.arange(0,dat.shape[0])-i0
j = np.arange(0,dat.shape[1])-j0

ii, jj = np.meshgrid(i,j)

# Generate slices coordinates
max_r = np.min(dat.shape)/2
n_slice = 90

r = np.arange(-max_r,max_r)
t = np.arange(0,n_slice)*np.pi/n_slice

rr, tt = np.meshgrid(r,t)
iiq, jjq = pol2cart(tt,rr)

# Interpolate slices
f = interpolate.interp2d(j,i,dat,fill_value=0)

slices = np.zeros_like(iiq)

for a in np.arange(iiq.shape[0]):
    for b in np.arange(iiq.shape[1]):
        slices[a,b] = f(iiq[a,b],jjq[a,b])


# Output Images 
plt.figure(num=1)
plt.clf()
plt.imshow(dat) 

# Output Images 
plt.figure(num=2)
plt.clf()
plt.imshow(slices) 

# Compute mean and variation of slice values
mean = np.mean(slices,axis=0)
upp = np.percentile(slices,84,axis=0)
low = np.percentile(slices,16,axis=0)

prop_delta = (upp-low)/(1e-3+mean)

min_prop_delta = np.median(prop_delta)

idxs = np.where(prop_delta<min_prop_delta)

upp[idxs] = mean[idxs]*(1+min_prop_delta/2)
low[idxs] = mean[idxs]*(1-min_prop_delta/2)

# Plot mean as line with variation as shading
plt.figure(num=3)
plt.clf()
plt.plot(r,mean,color='C0')

plt.fill_between(r, upp, low, color='C0', alpha='0.1',linewidth=0.0)

