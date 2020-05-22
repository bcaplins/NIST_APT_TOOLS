# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Need to put the functions in the path
# Probably not necessary if I understood Python/Git/modules better
import os 
import sys
parent_directory = os.getcwd().rsplit(sep='\\',maxsplit=1)[0]
if parent_directory not in sys.path:
    sys.path.insert(1, parent_directory)

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

# Read in data
#work_dir = r"C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript"
#work_dir = r"C:\\Users\\lnm\\Documents\\Div 686\\Data\\181210_D315_A74"
work_dir = r"C:\\Users\\lnm\\Documents\\Div 686\\Data\\190406_InGaNQW_A82"
#fn = work_dir+"\\"+r"R20_07094-v03.epos" # template
fn = work_dir+"\\"+r"R20_07199.epos" # Mg doped
#fn = work_dir+"\\"+r"R20_07199-v03.epos" # InGaN QW
#fn = work_dir+"\\"+r"R20_07247.epos" # CSR ~ 2
#fn = work_dir+"\\"+r"R20_07248-v01.epos" # CSR ~ 2
#fn = work_dir+"\\"+r"R20_07249-v01.epos" # CSR ~ 0.5
#fn = work_dir+"\\"+r"R20_07250-v01.epos" # CSR ~ 0.1


fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
roi_event_idxs = np.arange(5000,epos.size-5000)


#roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,fig_idx=1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
LASER_REP_RATE = 10000.0
wall_time = np.cumsum(epos['pslep'])/LASER_REP_RATE
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Define peaks to range
ed = initElements_P3.initElements()

import peak_assignments
pk_data = peak_assignments.Mg_doped_GaN()

# Define which peaks to use for CSR calcs
Ga1p_m2qs = [ed['Ga'].isotopes[69][0], ed['Ga'].isotopes[71][0]]
Ga2p_m2qs = [ed['Ga'].isotopes[69][0]/2, ed['Ga'].isotopes[71][0]/2]

Ga1p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga1p_m2qs]
Ga2p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga2p_m2qs]

# Range the peaks
pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'],peak_height_fraction=0.1)
    
# Determine the global background
#glob_bg_param = ppd.fit_uncorr_bg(epos['m2q'],fit_roi=[3.5,6.5])
bg_rois=[[10,11]]
glob_bg_param = ppd.get_glob_bg(epos['m2q'],rois=bg_rois)

glob_bg_param = glob_bg_param*0.9

# Count the peaks, local bg, and global bg
cts = ppd.do_counting(epos,pk_params,glob_bg_param)

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
print('Total Ions: '+str(epos.size))

print('Overall CSR (no bg)    : '+str(np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])))
print('Overall CSR (local bg) : '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['local_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['local_bg'][Ga1p_idxs]))))
print('Overall CSR (global bg): '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['global_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['global_bg'][Ga1p_idxs]))))


# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 120],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,30)
ys_sm = ppd.moving_average(ys,30)

glob_bg = ppd.physics_bg(xs,glob_bg_param)    

fig = plt.figure(num=2)
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

for roi in bg_rois:
    xbox = np.array([roi[0],roi[0],roi[1],roi[1]])
    ybox = np.array([0.1,np.max(ys_sm)/10,np.max(ys_sm)/10,0.1])
    
    ax.fill(xbox,ybox, 'b', alpha=0.2)

        
plt.pause(0.1)



#### END BASIC ANALYSIS ####

#sys.exit([0])

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
        
        