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
epos = GaN_fun.load_epos(run_number='R20_07148', 
                         epos_trim=[5000, 5000],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.Mg_doped_GaN()
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

ax.set_xlim(0,80)
ax.set_ylim(5e1,5e5)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('MgGaN_full_spectrum.pdf')
fig.savefig('MgGaN_full_spectrum.jpg', dpi=300)





#### END BASIC ANALYSIS ####



'''
sys.exit([0])

#### START BONUS ANALYSIS ####

# Import some GaN helper functions
import GaN_fun


# Plot some detector hitmaps
GaN_fun.create_det_hit_plots(epos,pk_data,pk_params,fig_idx = 10)

# Find the pole center and show it
ax = plt.gcf().get_axes()[0]
m2q_roi = [3, 100]
sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
xc,yc = GaN_fun.mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])
a_circle = plt.Circle((xc[-1],yc[-1]), 10, facecolor='none', edgecolor='k', lw=2, ls='-')
ax.add_artist(a_circle)



# Find all the Mg events
is_Mg = ~np.isfinite(epos['m2q'])
for pk,param in zip(pk_data,pk_params):
    if pk['Mg']>0:
        is_Mg = is_Mg | ((epos['m2q']>=param['pre_rng']) & (epos['m2q']<=param['post_rng']))
        
        

# Plot the 3d point cloud.  Not really all that useful TBH
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(num=11)
fig.clear()
ax = fig.gca(projection='3d')

X = epos['x'][is_Mg]
Y = epos['y'][is_Mg]
Z = epos['z'][is_Mg]

ax.scatter(X,Y,Z)
GaN_fun.set_axes_equal(ax)
plt.grid()        
        
'''     