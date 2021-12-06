# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import GaN_fun
import GaN_type_peak_assignments

import peak_param_determination as ppd
from histogram_functions import bin_dat

plt.close('all')

# Read in data
epos = GaN_fun.load_epos(run_number='R20_07199_redo', 
                         epos_trim=[0, 0],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.In_doped_GaN()
bg_rois=[[0.4,0.9]]
#bg_rois=[[10,11]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.1, 
        bg_rois=bg_rois)

cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=epos, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=2)

# Print out the composition of the full dataset
ppd.pretty_print_compositions(compositions,pk_data)

# Plot the full spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=1)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='full spec')

glob_bg = ppd.physics_bg(xs,glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1)

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e5)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_full_spectrum.pdf')
fig.savefig('InGaN_full_spectrum.jpg', dpi=300)




# Find the pole center and show it
m2q_roi = [3, 100]
sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
xc,yc = GaN_fun.mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])

# Find all the Indium events
is_In = ~np.isfinite(epos['m2q'])
for pk,param in zip(pk_data,pk_params):
    if pk['In']>0:
        is_In = is_In | ((epos['m2q']>=param['pre_rng']) & (epos['m2q']<=param['post_rng']))

# Fit the QW to a plane and rotate the point cloud to 'flatten' wrt z-axis
p = GaN_fun.qw_plane_fit(epos['x'][is_In],epos['y'][is_In],epos['z'][is_In],np.array([0,0,4.2]))
epos['x'],epos['y'],epos['z'] = GaN_fun.rotate_data_flat(p,epos['x'],epos['y'],epos['z'])

# USER DEFINED!!!
# Z-ROIs TO TAKE COMPOSITIONS IN
#z_roi_qw = np.array([1.5, 2.3])
#z_roi_buf = np.array([7.5, 12.5])
#z_roi_gan = np.array([3, 6])


        
# Find all the Ga events
is_Ga = ~np.isfinite(epos['m2q'])
for pk,param in zip(pk_data,pk_params):
    if pk['Ga']>0:
        is_Ga = is_Ga | ((epos['m2q']>=param['pre_rng']) & (epos['m2q']<=param['post_rng']))

import colorcet as cc    
cm=cc.cm.glasbey



# Plot the 'flat' interface to verify vector algebra didn't go awry
fig = plt.figure(num=11)
fig.set_size_inches(3,5)
fig.clear()
ax = fig.gca()

image_scale_factor = 10/8

ax.plot(image_scale_factor*epos['x'][is_Ga],image_scale_factor*epos['z'][is_Ga],'.', alpha=0.05, color=cm(2), ms=2)
ax.plot(image_scale_factor*epos['x'][is_In],image_scale_factor*epos['z'][is_In],'.', alpha=0.5, color=cm(10), ms=2)
ax.set_aspect('equal', 'box')
fig.set_tight_layout(True)

fig.savefig('InGaN_recon.png', format='png', dpi=600)

