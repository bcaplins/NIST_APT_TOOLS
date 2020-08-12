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
epos = GaN_fun.load_epos(run_number='R20_07199', 
                         epos_trim=[5000, 5000],
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
p = GaN_fun.qw_plane_fit(epos['x'][is_In],epos['y'][is_In],epos['z'][is_In],np.array([0,0,1.8]))
epos['x'],epos['y'],epos['z'] = GaN_fun.rotate_data_flat(p,epos['x'],epos['y'],epos['z'])

# USER DEFINED!!!
# Z-ROIs TO TAKE COMPOSITIONS IN
z_roi_qw = [1.5, 2.3]
z_roi_buf = [7.5, 12.5]
z_roi_gan = [3, 6]

# Plot the 'flat' interface to verify vector algebra didn't go awry
fig = plt.figure(num=11)
fig.clear()
ax = fig.gca()
ax.plot(epos['x'][is_In],epos['z'][is_In],'.')
ax.plot(epos['y'][is_In],epos['z'][is_In],'.')

# Plot the ROIs for visual inspection
ax.fill([-8,-8, 8,8], [z_roi_qw[0], z_roi_qw[1], z_roi_qw[1], z_roi_qw[0]], color=[1,0,0,0.5]) 
ax.fill([-8,-8, 8,8], [z_roi_buf[0], z_roi_buf[1], z_roi_buf[1], z_roi_buf[0]], color=[1,0,0,0.5]) 


# Calculate the QW composition
is_roi = (epos['z']>=z_roi_qw[0]) & (epos['z']<=z_roi_qw[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the QW spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=2)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='QW spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e3)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_QW_spectrum.pdf')
fig.savefig('InGaN_QW_spectrum.jpg', dpi=300)



# Calculate the buffer composition
is_roi = (epos['z']>=z_roi_buf[0]) & (epos['z']<=z_roi_buf[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the buffer spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=3)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='buffer spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_buffer_spectrum.pdf')
fig.savefig('InGaN_buffer_spectrum.jpg', dpi=300)

# Calculate the barrier composition
is_roi = (epos['z']>=z_roi_gan[0]) & (epos['z']<=z_roi_gan[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the barrier spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=4)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='barrier spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg$\pm', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_barrier_spectrum.pdf')
fig.savefig('InGaN_barrier_spectrum.jpg', dpi=300)
