# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:05:04 2020

@author: capli
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

# Read in data
fn = r'C:\Users\capli\Downloads\JAVASS_R44_03187-v01.epos'
epos = apt_fileio.read_epos_numpy(fn)



plotting_stuff.plot_histo(epos['m2q'],1)

isSingle = np.nonzero(epos['ipp'] == 1)

ed = initElements_P3.initElements()


pk_data =   np.array(    [  (1,  ed['Si'].isotopes[28][0]/2),
                            (1,  ed['Si'].isotopes[29][0]/2),
                            (1,  ed['Si'].isotopes[30][0]/2),
                            (1,  ed['Si'].isotopes[28][0]),
                            (1,  ed['Si'].isotopes[29][0]),
                            (1,  ed['Si'].isotopes[30][0])],
                            dtype=[('Si','i4'),('m2q','f4')] )

pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'],peak_height_fraction=0.01)
glob_bg_param = ppd.get_glob_bg(epos['m2q'], rois=[[24,27]])
cts = ppd.do_counting(epos,pk_params,glob_bg_param)

# Test for peak S/N and throw out craptastic peaks
B = np.max(np.c_[cts['local_bg'][:,None],cts['global_bg'][:,None]],1)[:,None]
T = cts['total'][:,None]
S = T-B
std_S = np.sqrt(T+B)
# Make up a threshold for peak detection... for the most part this won't matter
# since weak peaks don't contribute to stoichiometry much... except for Mg!
is_peak = S>1*np.sqrt(2*B)
for idx, ct in enumerate(cts):
    if not is_peak[idx]:
        for i in np.arange(len(ct)):
            ct[i] = 0



glob_bg = [ct[0]-ct[2] for ct in cts]
glob_bg[0:3]/sum(glob_bg[0:3]) 
glob_bg[3:6]/sum(glob_bg[3:6]) 











# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 100],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,10)

glob_bg = ppd.physics_bg(xs,glob_bg_param)    

fig = plt.figure(num=100)
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
#        ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
#        ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
        
#        ax.plot(np.array([pk_param['pre_bg_rng'],pk_param['post_bg_rng']]) ,np.ones(2)*pk_param['loc_bg'],'g--')
    else:
        ax.plot(np.array([1,1])*pk_param['x0_mean_shift'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
        
plt.pause(0.1)





