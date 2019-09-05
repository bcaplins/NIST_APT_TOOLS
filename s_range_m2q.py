# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import time

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat
from voltage_and_bowl import do_voltage_and_bowl

# Read in data
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
new_fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(new_fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
#roi_event_idxs = np.arange(1000,epos.size-1000)
roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Define peaks to range
ed = initElements_P3.initElements()
#                            N      Ga  Da
pk_data =   np.array(    [  (1,     0,  ed['N'].isotopes[14][0]/2),
                            (1,     0,  ed['N'].isotopes[14][0]/1),
                            (1,     0,  ed['N'].isotopes[15][0]/1),
                            (0,     1,  ed['Ga'].isotopes[69][0]/3),
                            (0,     1,  ed['Ga'].isotopes[71][0]/3),
                            (2,     0,  ed['N'].isotopes[14][0]*2),
                            (2,     0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                            (0,     1,  ed['Ga'].isotopes[69][0]/2),
                            (0,     1,  ed['Ga'].isotopes[71][0]/2),
                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                            (3,     0,  ed['N'].isotopes[14][0]*3),
                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                            (0,     1,  ed['Ga'].isotopes[69][0]),
                            (0,     1,  ed['Ga'].isotopes[71][0])],
                            dtype=[('N_at_ct','i4'), 
                                   ('Ga_at_ct','i4'),
                                   ('m2q','f4')] )

# Initialize a peak paramter array
pk_params = np.full(pk_data.size,-1,dtype=[('x0','f4'),
                                            ('std_fit','f4'),
                                            ('std_smooth','f4'),
                                            ('off','f4'),
                                            ('amp','f4'),
                                            ('pre_rng','f4'),
                                            ('post_rng','f4'),
                                            ('pre_bg','f4'),
                                            ('post_bg','f4')])
pk_params['x0'] = pk_data['m2q']

full_roi = np.array([0, 100])
xs_full_1mDa, ys_full_1mDa = bin_dat(m2q_corr,user_roi=full_roi,isBinAligned=True)
ys_full_5mDa_sm = ppd.do_smooth_with_gaussian(ys_full_1mDa, std=5)

N_kern = 250;
ys_full_fwd_sm = ppd.forward_moving_average(ys_full_1mDa,n=N_kern)
ys_full_bwd_sm = ppd.forward_moving_average(ys_full_1mDa,n=N_kern,reverse=True)

#ppd.do_smooth_with_gaussian(ys_full_1mDa, std=100)
#
#ys_fwd = np.roll(ys_full_50mDa_sm,50)
#ys_bwd = np.roll(ys_full_50mDa_sm,-50)


fig = plt.figure(num=222)
fig.clear()
ax = plt.axes()

ax.plot(xs_full_1mDa,ys_full_1mDa,label='1 mDa bin')    
ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='5 mDa smooth')    

ax.grid('major')

# Get estiamtes for x0, amp, std_fit
for idx,pk_param in enumerate(pk_params):
    # Select peak roi
    roi_half_wid = 0.25
    pk_roi = np.array([-roi_half_wid, roi_half_wid])+pk_param['x0']
    pk_dat = m2q_corr[(m2q_corr>pk_roi[0]) & (m2q_corr<pk_roi[1])]
    
    # Fit to gaussian
    smooth_param = 5
    popt = ppd.fit_to_g_off(pk_dat,user_std=smooth_param)
    pk_param['amp'] = popt[0]
    pk_param['x0'] = popt[1]
    pk_param['std_fit'] = popt[2]
    pk_param['off'] = popt[3]
    
    pk_param['x0'] = ppd.mean_shift_peak_location(pk_dat,user_std=pk_param['std_fit'],user_x0=pk_param['x0'])
    
    ax = plt.gca()
    ax.plot(np.array([1,1])*pk_param['x0'],np.array([0, pk_param['amp']+pk_param['off']]),'k--')
    
    plt.pause(0.001)
#    plt.pause(2)

    # select starting locations
    pk_idx = np.argmin(np.abs(pk_param['x0']-xs_full_1mDa))
    pk_lhs_idx = np.argmin(np.abs((pk_param['x0']-pk_param['std_fit'])-xs_full_1mDa))    
    pk_rhs_idx = np.argmin(np.abs((pk_param['x0']+pk_param['std_fit'])-xs_full_1mDa))
    
    pk_amp = (pk_param['amp']+ys_full_5mDa_sm[pk_idx])/2
    
    curr_val = ys_full_5mDa_sm[pk_idx]
    
    for i in np.arange(pk_lhs_idx,-1,-1):
        prev_val = curr_val
        curr_val = ys_full_5mDa_sm[i]        
            
        if (curr_val<0.1*pk_amp) and (pk_param['pre_rng']<0):
            # This is a range limit
            pk_param['pre_rng'] = xs_full_1mDa[i]
        
        if curr_val<ys_full_bwd_sm[i]:
            # Assume we are at the prepeak baseline noise
            if pk_param['pre_rng']<0:
                pk_param['pre_rng'] = xs_full_1mDa[i]
            pk_param['pre_bg'] = xs_full_1mDa[i]
            break
        
    curr_val = ys_full_5mDa_sm[pk_idx]
    for i in np.arange(pk_rhs_idx,xs_full_1mDa.size):
        prev_val = curr_val
        curr_val = ys_full_5mDa_sm[i]        
            
        if (curr_val<0.1*pk_amp) and (pk_param['post_rng']<0):
            # This is a range limit
            pk_param['post_rng'] = xs_full_1mDa[i]
        
        if curr_val<ys_full_fwd_sm[i]:
            # Assume we are at the prepeak baseline noise
            if pk_param['post_rng']<0:
                pk_param['post_rng'] = xs_full_1mDa[i]
            pk_param['post_bg'] = xs_full_1mDa[i]
            break      
    
    ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
    ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+5*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
    plt.pause(2)

ax.clear()
ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='5 mDa smooth')    
for idx,pk_param in enumerate(pk_params):
    ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
    ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
    ax.plot(np.array([1,1])*pk_param['pre_bg'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
    ax.plot(np.array([1,1])*pk_param['post_bg'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')

ax.set_yscale('log')
ax.set(ylim=[0.1,1000])
    
    
    

glob_bg_param = ppd.fit_uncorr_bg(m2q_corr)
glob_bg = ppd.physics_bg(xs_full_1mDa,glob_bg_param)    
    
#ax.clear()
#ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='5 mDa smooth')    
ax.plot(xs_full_1mDa,ys_full_fwd_sm,label='fwd')    
ax.plot(xs_full_1mDa,ys_full_bwd_sm,label='bwd')    
ax.plot(xs_full_1mDa,glob_bg,label='glob_bg')    

ax.legend()    
    
stoich = np.c_[pk_data['N_at_ct'][pk_data['is_anal_pk'],None],pk_data['Ga_at_ct'][pk_data['is_anal_pk'],None]]
cts = np.zeros((np.sum(pk_data['is_anal_pk']),3))-1
    
for idx,pk_param in enumerate(pk_params):
    pre_pk_rng = [pk_param['pre_bg']-0.15,pk_param['pre_bg']-0.05]
    pk_rng = [pk_param['pre_rng'],pk_param['post_rng']]
    
    local_bg_cts = np.sum((m2q_corr>=pre_pk_rng[0]) & (m2q_corr<=pre_pk_rng[1]))*(pk_rng[1]-pk_rng[0])/(pre_pk_rng[1]-pre_pk_rng[0])
    global_bg_cts = np.sum(glob_bg[(xs_full_1mDa>=pre_pk_rng[0]) & (xs_full_1mDa<=pre_pk_rng[1])])
    tot_cts = np.sum((m2q_corr>=pk_rng[0]) & (m2q_corr<=pk_rng[1]))
    
    cts[idx,0] = tot_cts
    cts[idx,1] = local_bg_cts
    cts[idx,2] = global_bg_cts
    

tot_cts = cts[:,0][:,None]
pk_cts_loc = cts[:,0][:,None]-cts[:,1][:,None]
pk_cts_glob = cts[:,0][:,None]-cts[:,2][:,None]

glob_res = np.sum(stoich*pk_cts_glob,axis=0)
glob_stoich = glob_res/np.sum(glob_res)
    
loc_res = np.sum(stoich*pk_cts_loc,axis=0)
loc_stoich = loc_res/np.sum(loc_res)

tot_res = np.sum(stoich*tot_cts,axis=0)
tot_stoich = tot_res/np.sum(tot_res)




print(tot_stoich,loc_stoich,glob_stoich)
    
