# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import apt_importers as apt
from voltage_and_bowl import do_voltage_and_bowl
import numpy as np
import time
import m2q_calib

import peak_param_determination as ppd
from helper_functions import bin_dat

import matplotlib.pyplot as plt

from scipy.optimize import least_squares
import initElements_P3




def moving_average(a, n=3) :    
    # Moving average with reflection at the boundaries
    if 2*n+1>a.size:
        raise Exception('The kernel is too big!')
    kern = np.ones(2*n+1)/(2*n+1)
    return np.convolve(np.r_[a[n:0:-1],a,a[-2:-n-2:-1]],kern,'valid')

def forward_moving_average(a, n=3, reverse=False) :    
    # Moving average with reflection at the boundaries
    if n>a.size:
        raise Exception('The kernel is too big!')    
    kern = np.ones(n)/n
    if not reverse:
        ret = np.convolve(np.r_[a,a[-2:-n-1:-1]],kern,'valid')
    else:
        ret = np.convolve(np.r_[a[n-1:0:-1],a],kern,'valid')
    return ret



# Read in template spectrum
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
ref_epos = apt.read_epos_numpy(fn)
ref_epos = ref_epos[0:ref_epos.size//2]

# Read in data
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
epos = apt.read_epos_numpy(fn)
epos = epos[epos.size//2:-1]

wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)

# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")


# Find c and t0 for ToF data based on reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)

full_roi = np.array([0, 100])
xs_full_1mDa, ys_full_1mDa = bin_dat(m2q_corr,user_roi=full_roi,isBinAligned=True)
ys_full_5mDa_sm = ppd.do_smooth_with_gaussian(ys_full_1mDa, std=5)

xs_full_1mDa_ref, ys_full_1mDa_ref = bin_dat(ref_epos['m2q'],user_roi=full_roi,isBinAligned=True)
ys_full_5mDa_sm_ref = ppd.do_smooth_with_gaussian(ys_full_1mDa_ref, std=5)


fig = plt.figure(num=101)
fig.clear()
ax = plt.axes()
ax.plot(xs_full_1mDa_ref,ys_full_5mDa_sm_ref,label='ref')
ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='dat')

ed = initElements_P3.initElements()




# Perform linear calibration over known peaks
#                            calib anal  N      Ga  Da
pk_data =   np.array(    [  (1,    0,    0,     0,  ed['H'].isotopes[1][0]),
                            (0,    1,    1,     0,  ed['N'].isotopes[14][0]/2),
                            (1,    1,    1,     0,  ed['N'].isotopes[14][0]/1),
                            (0,    1,    1,     0,  ed['N'].isotopes[15][0]/1),
                            (0,    1,    0,     1,  ed['Ga'].isotopes[69][0]/3),
                            (0,    1,    0,     1,  ed['Ga'].isotopes[71][0]/3),
                            (1,    1,    2,     0,  ed['N'].isotopes[14][0]*2),
                            (0,    1,    2,     0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                            (1,    1,    0,     1,  ed['Ga'].isotopes[69][0]/2),
                            (1,    1,    0,     1,  ed['Ga'].isotopes[71][0]/2),
                            (0,    1,    1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                            (0,    1,    3,     0,  ed['N'].isotopes[14][0]*3),
                            (0,    1,    1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                            (1,    1,    0,     1,  ed['Ga'].isotopes[69][0]),
                            (1,    1,    0,     1,  ed['Ga'].isotopes[71][0])],
                            dtype=[('is_cal_pk','?'), 
                                   ('is_anal_pk','?'),
                                   ('N_at_ct','i4'), 
                                   ('Ga_at_ct','i4'),
                                   ('m2q','f4')] )



ref_pk_m2qs = pk_data['m2q'][pk_data['is_cal_pk']]
m2q_corr = m2q_calib.calibrate_m2q_by_peak_location(m2q_corr,ref_pk_m2qs)
          
pk_params = np.zeros(np.sum(pk_data['is_anal_pk']),dtype=[('x0','f4'),
                                                  ('std_fit','f4'),
                                                  ('std_smooth','f4'),
                                                  ('off','f4'),
                                                  ('amp','f4'),
                                                  ('pre_rng','f4'),
                                                  ('post_rng','f4'),
                                                  ('pre_bg','f4'),
                                                  ('post_bg','f4')])
pk_params['x0'] = pk_data['m2q'][pk_data['is_anal_pk']]

pk_params['pre_rng'] = -1
pk_params['post_rng'] = -1

full_roi = np.array([0, 100])
xs_full_1mDa, ys_full_1mDa = bin_dat(m2q_corr,user_roi=full_roi,isBinAligned=True)
ys_full_5mDa_sm = ppd.do_smooth_with_gaussian(ys_full_1mDa, std=5)

N_kern = 250;
ys_full_fwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern)
ys_full_bwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern,reverse=True)

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
    

raise SystemExit(0)

  

stoich = np.c_[pk_data['N_at_ct'][pk_data['is_anal_pk'],None],pk_data['Ga_at_ct'][pk_data['is_anal_pk'],None]]
cts = np.zeros(np.sum(pk_data['is_anal_pk']))-1

singles = m2q_corr[epos['ipp']==1];
multiples = m2q_corr[epos['ipp']!=1];


dat_set = m2q_corr
dat_set = singles
dat_set = multiples

for idx,pk_param in enumerate(pk_params):
    pre_pk_rng = [pk_param['pre_bg']-0.15,pk_param['pre_bg']-0.05]
    pk_rng = [pk_param['pre_rng'],pk_param['post_rng']]
    
    tot_cts = np.sum((dat_set>=pk_rng[0]) & (dat_set<=pk_rng[1]))
    
    cts[idx] = tot_cts
    

tot_cts = cts[:,None]

tot_res = np.sum(stoich*tot_cts,axis=0)
tot_stoich = tot_res/np.sum(tot_res)


Ga3p = np.sum(tot_cts[3:4+1])
Ga2p = np.sum(tot_cts[7:8+1])
Ga1p = np.sum(tot_cts[12:13+1])

print(Ga1p/Ga2p)

print(tot_stoich)
    
    
    

full_roi = np.array([0, 100])
xs_full_1mDa, ys_full_1mDa = bin_dat(m2q_corr,user_roi=full_roi,isBinAligned=True)
ys_full_20mDa_sm = ppd.do_smooth_with_gaussian(ys_full_1mDa, std=50)

_, ys_singles_1mDa = bin_dat(singles,user_roi=full_roi,isBinAligned=True)
ys_singles_20mDa_sm = ppd.do_smooth_with_gaussian(ys_singles_1mDa, std=50)

_, ys_multiples_1mDa = bin_dat(multiples,user_roi=full_roi,isBinAligned=True)
ys_multiples_20mDa_sm = ppd.do_smooth_with_gaussian(ys_multiples_1mDa, std=50)


ax.clear()
ax.plot(xs_full_1mDa,ys_full_20mDa_sm,label='all')    
ax.plot(xs_full_1mDa,ys_singles_20mDa_sm,label='singles')    
ax.plot(xs_full_1mDa,ys_multiples_20mDa_sm,label='multiples')    



fig = plt.figure(num=222)
fig.clear()
ax = plt.axes()

ax.clear()
ax.plot(xs_full_1mDa,(ys_multiples_20mDa_sm-ys_singles_20mDa_sm)/(ys_multiples_20mDa_sm+ys_singles_20mDa_sm),label='MULT-SING/MULT+SING')    
#ax.set_yscale('log')
ax.set(ylim=[-0.5,0.5])
ax.legend()
    
    
    
    
    
    
    
    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    # re-Select peak roi
#    roi = np.array([-pk_param['std_fit']*10, pk_param['std_fit']*10])+pk_param['x0']    
#    pk_dat = m2q_corr[(m2q_corr>roi[0]) & (m2q_corr<roi[1])]
#    
#    # re-Fit to gaussian
#    smooth_param = 2
#    popt = ppd.fit_to_g_off(pk_dat,user_std=smooth_param,user_p0=popt)
#    pk_param['amp'] = popt[0]
#    pk_param['std_fit'] = popt[2]
#    pk_param['off'] = popt[3]
#    pk_param['x0'] = ppd.mean_shift_peak_location(pk_dat,user_std=pk_param['std_fit'],user_x0=popt[1])
#    
#    plt.pause(3)
#    
#    ax = plt.gca()
#    
#    ys_smoothed = ppd.do_smooth_with_gaussian(ys,pk_param['std_smooth']/5)
#    
#    ax.plot(np.array([1,1])*(pk_param['x0']-1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']-3*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+5*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    
#    ax.plot(np.array([-3*pk_param['std_fit'],5*pk_param['std_fit']])+pk_param['x0'],
#            0.1*(pk_param['amp']-pk_param['off'])+pk_param['off']+np.array([0,0]),'k--')
#    
#    plt.pause(4)
#    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#for idx,pk_param in enumerate(pk_params):
#    # Select peak roi
#    roi_half_wid = 0.25
#    roi = np.array([-roi_half_wid, roi_half_wid])+pk_param['x0']
#    pk_dat = m2q_corr[(m2q_corr>roi[0]) & (m2q_corr<roi[1])]
#    
#    # Determine optimal smoothing parameter
#    _,_,pk_param['std_smooth'] = ppd.smooth_with_gaussian_CV(pk_dat)
#    xs, ys = bin_dat(pk_dat)
#    smooth_param = np.max([pk_param['std_smooth']/2,5])
#    smooth_param = 10
#    # Fit to gaussian
#    popt = ppd.fit_to_g_off(pk_dat,user_std=smooth_param)
#    pk_param['amp'] = popt[0]
#    pk_param['x0'] = popt[1]
#    pk_param['std_fit'] = popt[2]
#    pk_param['off'] = popt[3]
#    
#    # re-Select peak roi
#    roi = np.array([-pk_param['std_fit']*10, pk_param['std_fit']*10])+pk_param['x0']
#    
#    pk_dat = m2q_corr[(m2q_corr>roi[0]) & (m2q_corr<roi[1])]
#    
#    # re-Fit to gaussian
#    smooth_param = np.max([pk_param['std_smooth']/5,5])
#    popt = ppd.fit_to_g_off(pk_dat,user_std=smooth_param,user_p0=popt)
#
#    pk_param['amp'] = popt[0]
#    pk_param['std_fit'] = popt[2]
#    pk_param['off'] = popt[3]
#    pk_param['x0'] = ppd.mean_shift_peak_location(pk_dat,user_std=pk_param['std_fit'],user_x0=popt[1])
#    
#    ax = plt.gca()
#    
#    ys_smoothed = ppd.do_smooth_with_gaussian(ys,pk_param['std_smooth']/5)
#    
#    ax.plot(np.array([1,1])*(pk_param['x0']-1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']-3*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    ax.plot(np.array([1,1])*(pk_param['x0']+5*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    
#    ax.plot(np.array([-3*pk_param['std_fit'],5*pk_param['std_fit']])+pk_param['x0'],
#            0.1*(pk_param['amp']-pk_param['off'])+pk_param['off']+np.array([0,0]),'k--')
#    
#    plt.pause(4)
#    
#    break
#
#
#
#
#
#for idx,pk_param in enumerate(pk_params):
#    # Select peak roi
#    roi_half_wid = 0.25
#    roi = np.array([-roi_half_wid, roi_half_wid])+pk_param['x0']
#    pk_dat = m2q_corr[(m2q_corr>roi[0]) & (m2q_corr<roi[1])]
#    
#    xs, ys = bin_dat(pk_dat,bin_width=0.001,isBinAligned=True)
#    
#    N = 5
#    ys_smoothed = moving_average(ys,n=N)
#    
#    ys_fwd = forward_moving_average(ys_smoothed, n=10*(2*N+1))
#    ys_bwd = forward_moving_average(ys_smoothed, n=10*(2*N+1), reverse=True)
#    
#    
#
#    fig = plt.figure(num=222)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys_smoothed,label='raw')    
##    ax.plot(xs,ys_fwd,label='fwd')    
##    ax.plot(xs,ys_bwd,label='bwd')    
#    
#    ax.plot(xs,ys_smoothed-ys_fwd,label='sub')    
#    ax.grid('major')
#    
#    
#    
#    plt.pause(4)
#
#
#
#
#
#
#
#
#
#from scipy.interpolate import CubicSpline
#
#knot_xs = 
#knot_ys = 
#
#CubicSpline(knot_xs, knot_ys, bc_type='not-a-knot', extrapolate=None)
#
##    
##    
##
##    ax.plot(xs,ys_smoothed,'--')
##
##    rng = ppd.get_range_empirical(xs,ys_smoothed)
##    
##    
##    ax.plot(np.array([1,1])*rng[0],np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
##    ax.plot(np.array([1,1])*rng[1],np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
##    
##    ax.set_yscale('linear')    
##    
##    print(rng)
##    
##    
#    
#    plt.pause(4)
#    # Find global background
#    
#    # Subtract global background from smoothed histogram
#
#    
#    
#    
#
#
#
#m2q_roi = [0, 100]
#m2q_bin_width = .01
#m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#
##ref_histo = np.histogram(epos['m2q'],range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#new_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#
#centers = (new_histo[1][1:]+new_histo[1][0:-1])/2
#
#
#fig = plt.figure(num=128)
#fig.clear()
#ax = plt.axes()
#
##ax.plot(centers,new_histo[0],label='ref')
#ax.plot(centers,new_histo[0],label='new')
#
##ax.hist(epos['m2q'],bins=ref_histo[1],log=True,histtype='step',label='ref')
##ax.hist(m2q_corr,bins=ref_histo[1],log=True,histtype='step',label='new')
#
##ax.legend()
#ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 100],
#       title='My First Plot')
#ax.set_yscale('log')
#    
#    
#    
#
#for idx,pk_param in enumerate(pk_params):
#    ax.plot(np.array([1,1])*pk_param['x0'],np.array([1,1e4]),'k--')
#
#    
#    
#    
#    
#    
#
## Find optimal smoothing parameter
#
## Fit smoothed data to gaussian with fixed width window
#
## Refit less smoothed data to gaussian with sigma width based window
#
## Somehow use the smoothing and gaussian fit parameters to make a range
#
#
#
#stoich = peak_data[:,0:2]
#quant_peak_m2qs = peak_data[:,2][:,None]
# 
#N_param = 1
#res_param = np.zeros((quant_peak_m2qs.size,N_param))
#cts = np.zeros((quant_peak_m2qs.size,2))
#
#
#for idx,m2q_loc in enumerate(quant_peak_m2qs):
#    
#    
#    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]   
#    
#    window_half_wid = 0.25
#    m2q_roi = np.array([-window_half_wid,window_half_wid])+quant_peak_m2qs[idx]   
#    
#    pk_dat = m2q_corr[(m2q_corr>=m2q_roi[0]) & (m2q_corr<=m2q_roi[1])]
#    
#    ppd.est_hwhm2(pk_dat)
#    plt.pause(4)
#    
#    
#    xs, ys, res_param[idx] = ppd.smooth_with_gaussian_CV(pk_dat)
#    
#    est_bl = ppd.est_baseline_empirical(xs,ys)
#    
#    ax = plt.gca()
#    ax.plot(np.array([np.min(xs),np.max(xs)]),np.array([1,1])*est_bl,'k--')
#    
#    rng = ppd.get_range_empirical(xs,ys)
#    
#    ax.plot(np.array([1,1])*rng[0],np.array([np.min(ys),np.max(ys)]),'k--')
#    ax.plot(np.array([1,1])*rng[1],np.array([np.min(ys),np.max(ys)]),'k--')
#    
#    ax.set_yscale('linear')    
#
#    cts[idx,:] = ppd.get_peak_count(m2q_corr,rng,est_bl)
#    
#    plt.pause(3)
#
#
#
#
#tot_cts = np.sum(cts,axis=1)[:,None]
#pk_cts = cts[:,0][:,None]
#    
#pk_res = np.sum(stoich*pk_cts,axis=0)
#pk_res/np.sum(pk_res)
#
#tot_res = np.sum(stoich*tot_cts,axis=0)
#tot_res/np.sum(tot_res)
#
#diff = tot_res/np.sum(tot_res)-pk_res/np.sum(pk_res)
#
#
#    
#
#raise SystemExit(0)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
## For each peak
## divide data in half
## Use one half to make a histo with a series of bin widths
## Use the other half to test which histo best predicts the other
#    
# 
#N_param = 1
#res_param = np.zeros((quant_peak_m2qs.size,N_param))
#
#
##for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
#for idx,m2q_loc in enumerate(quant_peak_m2qs):
#    
#    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
#    
#    pk_dat = m2q_corr[(m2q_corr>=m2q_roi[0]) & (m2q_corr<=m2q_roi[1])]
#    
#    choices = np.random.rand(pk_dat.size)>0.5
#    
#    pk_dat1 = pk_dat[np.nonzero(choices)]
#    pk_dat2 = pk_dat[np.nonzero(~choices)]
#    
#    m2q_bin_width = .001
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    histo1 = np.histogram(pk_dat1,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    histo2 = np.histogram(pk_dat2,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    
#    xs = (histo1[1][1:]+histo1[1][0:-1])/2
#    
#    stds = np.logspace(0,np.log10(50),32)
#    errs = np.zeros_like(stds)
#    
#    for std_idx,std in enumerate(stds):
#
#        if int(3*std)>xs.size:
#            raise Exception('Whoa buddy!')
#        kern = gaussian(int(3*std),std)
#        kern /= np.sum(kern)
#        
#        ys1 = histo1[0]
#        ys1 = np.convolve(ys1,kern,'same')
#        ys2 = histo2[0]
#
#        errs[std_idx] = np.sum(np.square(ys2-ys1))
#
#    best_idx = np.argmin(errs)
#    
#    fig = plt.figure(num=127)
#    fig.clear()
#    ax = plt.axes()
#    ax.plot(stds,errs)
#    ax.set_xscale('log')    
#    ax.set_yscale('log')    
#    
#    plt.pause(0.002)
#    
#    histo = np.histogram(pk_dat,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    kern = gaussian(int(3*stds[best_idx]),stds[best_idx])
#    kern /= np.sum(kern)
#    
#    ys1 = histo[0]
#    ys1 = np.convolve(ys1,kern,'same')
#    ys2 = histo[0]
#
#    fig = plt.figure(num=128)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys2)
#    ax.plot(xs,ys1)
#    
#    
#    ax.set_yscale('log')    
#    
#    res_param[idx] = stds[best_idx]
#    
#    plt.pause(1)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#def G(x,F,E):
#    return np.exp(-4*np.log(2)*np.square(x-E)/np.square(F))
#
#def GL(x,F,E,m):
#    num = np.exp(-4*np.log(2)*(1-m)*np.square(x-E)/np.square(F))
#    denom = 1+4*m*np.square(x-E)/np.square(F)
#    return num/denom
#
#def AW(x,a,F,E):
#    exp_num = 2*np.sqrt(np.log(2))*(x-E)
#    exp_denom = F-a*2*np.sqrt(np.log(2))*(x-E)
#    exp2 = np.square(exp_num/exp_denom)
#    return np.exp(-exp2)
#
#def UG(x,a,b,F,E,m):
#    # assume x is sorted ascending
#    lhs_idxs = np.nonzero(x>E)
#    rhs_idxs = np.nonzero(x<=E)
#    
#    w = b*0.7+0.3/(a+0.01)
#    
#    res = np.zeros_like(x)
#    
#    res[lhs_idxs] = GL(x[lhs_idxs],F,E,m)+w*(AW(x[lhs_idxs],a,F,E)-G(x[lhs_idxs],F,E))
#    res[rhs_idxs] = GL(x[rhs_idxs],F,E,m)
#    
#    return res
#
#
#def b_G(x,sigma,x0):
#    return np.exp(-np.square(x-x0)/(2*np.square(sigma)))
#
#def b_L(x,omega,x0):
#    return omega**2/(omega**2+(x-x0)**2)
#
#def b_Ln(x,omega,x0,n):
#    return np.power(b_L(x,omega,x0),n)
#
#def b_bg(x,m,b):
#    return m*x+b
#
#def b_model(x,x0,omega,sigma,n,amp,l_frac,m,b):
#    lhs_idxs = np.nonzero(x<=x0)
#    rhs_idxs = np.nonzero(x>x0)
#    
#    y = np.zeros_like(x)
#    
#    y[lhs_idxs] = b_G(x[lhs_idxs],sigma,x0)
#    y[rhs_idxs] = (1-l_frac)*b_G(x[rhs_idxs],sigma,x0)+l_frac*b_Ln(x[rhs_idxs],omega,x0,n)
#    
#    y *= amp
#    y += b_bg(x,m,b)
#    
#    return y
#
#def b_e(x,tau,x0):
#    return np.exp(-(x-x0)/tau)*(x>=x0)
#
#def b_model2(x,amp_g,amp_e,x0,sigma,tau,m,b):
#    
#    y = np.zeros_like(x)
#    
#    t1 = amp_g*b_G(x,sigma,x0)
#    kern = t1/np.sum(t1)
#    
#    t2 = b_e(x,tau,x0)
#    t2 = amp_e*np.convolve(t2,kern,'same')
#    
#    
#    y = t1+t2
#    
#    y += b_bg(x,m,b)
#    
#    return y
#
#
#
#
#
#
#quant_peak_m2qs = np.array([14.003074/2, 
#                            14.003074, 
#                            14.003074+1, 
#                            23,
#                            23.66,
#                            2*14.003074,
#                            29,
#                            68.925581/2, 
#                            70.924701/2, 
#                            41.5,
#                            42,
#                            42.5,
#                            68.925581,
#                            70.924701])
#
#N_param = 7
#res_param = np.zeros((quant_peak_m2qs.size,N_param))
#
#
##for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
#for idx,m2q_loc in enumerate(quant_peak_m2qs):
#    
#    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
#    m2q_bin_width = .001
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    
#    xs = (histo[1][1:]+histo[1][0:-1])/2
#    ys = histo[0]
#    
##    if idx==quant_peak_m2qs.size-1:
#    if idx==0:
#        p0 = [1, 1, 1, 0.025, 0.1, 0, 1]
#    else:
#        p0 = popt.x
##        p0[1:2] = popt.x[1:2]*m2q_loc/popt.x[0]
#        
#        
#        
#    p0[2] = m2q_loc
#    p0[0] = (np.max(ys)-np.mean(ys[0:10]))/2
#    p0[1] = (np.max(ys)-np.mean(ys[0:10]))/2
#    
#    p0[5] = np.min([(np.mean(ys[-10:])-np.mean(ys[0:10]))/(np.mean(xs[-10:])-np.mean(xs[0:-10])),0])
#    p0[6] = np.mean(ys[0:10])-p0[5]*np.mean(xs[0:10])
#    
##    p0[4] = np.mean(ys[0:20])*np.sqrt(np.mean(xs[0:20]))
#    
#    resid_func = lambda p: b_model2(xs,*p)-ys
## b_model2(x,amp_g,amp_e,x0,sigma,tau,m,b):
#    lbs = [0,       0,      p0[2]-0.1,   0.001,   0.002,    -np.inf,    0]
#    ubs = [np.inf,  np.inf, p0[2]+0.1,   0.2,     1,        0,          np.inf]
#    
#    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
##    popt = least_squares(resid_func, p0, ftol=1e-12, max_nfev=2048)
#    
#    fig = plt.figure(num=128)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.')
#
#    mod0_ys = b_model2(xs,*p0)
#    ax.plot(xs,mod0_ys)
#    
#    
#    plt.pause(1)
#    
#    
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.')
##    p0 = [0.35,0.5,0.05,28,0.5,800,10]
#    
#    mod_ys = b_model2(xs,*popt.x)
#    ax.plot(xs,mod_ys)
#    
##    ax.plot(xs,mod_ys-ys,'r--')
#    
#    ax.set_yscale('log')    
#    
#    res_param[idx,:] = popt.x
#    
#    plt.pause(2)
#    
##    plt.ginput(1)
#    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#quant_peak_m2qs = np.array([14.003074/2, 
#                            14.003074, 
#                            14.003074+1, 
#                            23,
#                            23.66,
#                            2*14.003074,
#                            29,
#                            68.925581/2, 
#                            70.924701/2, 
#                            41.5,
#                            42,
#                            42.5,
#                            68.925581,
#                            70.924701])
#
#N_param = 8
#res_param = np.zeros((quant_peak_m2qs.size,N_param))
#
#
##for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
#for idx,m2q_loc in enumerate(quant_peak_m2qs):
#    
#    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
#    m2q_bin_width = .001
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    
#    xs = (histo[1][1:]+histo[1][0:-1])/2
#    ys = histo[0]
#    
##    if idx==quant_peak_m2qs.size-1:
#    if idx==0:
#        p0 = [0, 0.025, 0.025, 1, 1, 0.5, -0.1, 1]
#    else:
#        p0 = popt.x
##        p0[1:2] = popt.x[1:2]*m2q_loc/popt.x[0]
#        
#        
#    p0[0] = m2q_loc
#    p0[4] = np.max(ys)-np.mean(ys[0:10])
#    
#    p0[6] = np.min([(np.mean(ys[-10:])-np.mean(ys[0:10]))/(np.mean(xs[-10:])-np.mean(xs[0:-10])),0])
#    p0[7] = np.mean(ys[0:10])-p0[6]*np.mean(xs[0:10])
#    
##    p0[4] = np.mean(ys[0:20])*np.sqrt(np.mean(xs[0:20]))
#    
#    resid_func = lambda p: b_model(xs,*p)-ys
## b_model(x,x0,omega,sigma,n,amp,l_frac,m,b):
#    lbs = [p0[0]-0.1,   0.001,    0.001,   0.5,  0,      0, -np.inf,    0]
#    ubs = [p0[0]+0.1,   0.2,      0.2,     2,    np.inf, 1, 0,          np.inf]
#    
#    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    
#    fig = plt.figure(num=128)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys)
##    p0 = [0.35,0.5,0.05,28,0.5,800,10]
#    mod0_ys = b_model(xs,*p0)
#    ax.plot(xs,mod0_ys)
#    
#    
#    plt.pause(1)
#    
#    
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys)
##    p0 = [0.35,0.5,0.05,28,0.5,800,10]
#    
#    mod_ys = b_model(xs,*popt.x)
#    ax.plot(xs,mod_ys)
#    
##    ax.plot(xs,mod_ys-ys,'r--')
#    
#    ax.set_yscale('log')    
#    
#    res_param[idx,:] = popt.x
#    
#    plt.pause(1)
#    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#def fit_func(x,a,b,F,E,m,amp,fudge):
#    return amp*UG(x,a,b,F,E,m)+fudge*np.reciprocal(np.sqrt(x))
#
#
#quant_peak_m2qs = np.array([14.003074/2, 
#                            14.003074, 
#                            14.003074+1, 
#                            23,
#                            23.66,
#                            2*14.003074,
#                            29,
#                            68.925581/2, 
#                            70.924701/2, 
#                            41.5,
#                            42,
#                            42.5,
#                            68.925581,
#                            70.924701])
#
#N_param = 7
#res_param = np.zeros((quant_peak_m2qs.size,N_param))
#
#
#
#
#
##for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
#for idx,m2q_loc in enumerate(quant_peak_m2qs):
#    
#    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
#    m2q_bin_width = .005
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    
#    xs = (histo[1][1:]+histo[1][0:-1])/2
#    ys = histo[0]
#    
##    if idx==quant_peak_m2qs.size-1:
#    if idx==0:
#        p0 = [0.35,0.5,0.005,-1,0.5,-1,70]
#    else:
#        p0 = popt.x
#        p0[2] = popt.x[2]*m2q_loc/popt.x[3]
#        
#        
#    p0[3] = m2q_loc
#    p0[5] = np.max(ys)        
#    
#    resid_func = lambda p: fit_func(xs,*p)-ys
#         # a, b, F ,E, m, amp, off
#    lbs = [0,-1,0, p0[3]-0.1 ,0,0,1]
#    ubs = [1,1,0.2,p0[3]+0.1 ,1,np.inf,1e6]
#    
#    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=0, ftol=1e-12, max_nfev=2048)
#    
#    fig = plt.figure(num=128)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys)
##    p0 = [0.35,0.5,0.05,28,0.5,800,10]
#    mod0_ys = fit_func(xs,*p0)
#    ax.plot(xs,mod0_ys)
#    
#    
#    plt.pause(.01)
#    
#    
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.')
##    p0 = [0.35,0.5,0.05,28,0.5,800,10]
#    
#    mod_ys = fit_func(xs,*popt.x)
#    ax.plot(xs,mod_ys)
#    
##    ax.plot(xs,mod_ys-ys,'r--')
#    
#    ax.set_yscale('log')    
#    
#    res_param[idx,:] = popt.x
#    
#    plt.pause(1)
#
#
#
#
#
#
#    
#    
#    
#   
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#m2q_roi = [0, 100]
#m2q_bin_width = .001
#m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#
#ref_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=True)
#
#std = 30
#kern = gaussian(int(3*std),std)
#kern /= np.sum(kern)
#
#ys = ref_histo[0]
#ys = np.convolve(ys,kern,'same')
#
#
#
#centers = (ref_histo[1][1:]+ref_histo[1][0:-1])/2
#
#
#
#fig = plt.figure(num=128)
#fig.clear()
#ax = plt.axes()
#
#ax.plot(centers,ref_histo[0],label='ref')
#ax.plot(centers,ys,label='new')
##
##ax.hist(epos['m2q'],bins=ref_histo[1],log=True,histtype='step',label='ref')
##ax.hist(m2q_corr,bins=ref_histo[1],log=True,histtype='step',label='new')
#
#ax.legend()
#ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 100],
#       title='My First Plot')
#ax.set_yscale('log')    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#found_peak_m2q = np.zeros_like(ref_peak_m2qs)
#
#for idx in range(ref_peak_m2qs.size):
#    m2q_roi = ref_peak_m2qs[idx]*np.array([0.98,1.02])
#    x = m2q_corr[(m2q_corr>m2q_roi[0]) & (m2q_corr<m2q_roi[1])]        
##    x = epos['m2q'][(epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1])]        
#    found_peak_m2q[idx] = mean_shift_peak_location(x)
#
#fig = plt.figure(num=256)
#fig.clear()
#ax = plt.axes()
#ax.plot(ref_peak_m2qs,found_peak_m2q/ref_peak_m2qs,'-o')
##ax.plot(exp_peak_m2q,exp_peak_m2q/exp_peak_m2q,'--k')
#
#
#f = interp1d(found_peak_m2q,exp_peak_m2q,
#             kind='linear',
#             copy=True,
#             fill_value='extrapolate',
#             assume_sorted=False)
#m2q_corr2 = f(m2q_corr)
#
#
#exp_peak_m2q = np.array([1.007825, 14.003074/2, 14.003074, 2*14.003074, 68.925581/2, 70.924701/2, 68.925581, 70.924701])
#found_peak_m2q = np.zeros_like(exp_peak_m2q)
#
#for idx in range(exp_peak_m2q.size):
#    m2q_roi = exp_peak_m2q[idx]*np.array([0.98,1.02])
#    x = m2q_corr2[(m2q_corr2>m2q_roi[0]) & (m2q_corr2<m2q_roi[1])]        
##    x = epos['m2q'][(epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1])]        
#    print(est_hwhm(x))
#    found_peak_m2q[idx] = mean_shift_peak_location(x)
#
#fig = plt.figure(num=256)
#fig.clear()
#ax = plt.axes()
#ax.plot(exp_peak_m2q,(found_peak_m2q-exp_peak_m2q)/exp_peak_m2q,'-o')
##ax.plot(exp_peak_m2q,exp_peak_m2q,'--k')
#
#
#
#for i in range(exp_peak_m2q.size):
#    
#    m2q_roi = exp_peak_m2q[i]*np.array([0.98,1.02])
#    x = m2q_corr[(m2q_corr>m2q_roi[0]) & (m2q_corr<m2q_roi[1])]
#        
#    hwhm_est = est_hwhm(x)
#    pk_loc_est = mean_shift_peak_location(x)
#
#    
#    m2q_bin_width = hwhm_est/5
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    new_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
#    
#    
#    fig = plt.figure(num=128)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.hist(x,bins=new_histo[1],log=False,histtype='step',label='ref')
#    
#    
#    ax.plot([pk_loc_est,pk_loc_est],[0,np.max(new_histo[0])],'k--')
#    ax.plot([pk_loc_est,pk_loc_est]-hwhm_est,[0,np.max(new_histo[0])],'k--')
#    ax.plot([pk_loc_est,pk_loc_est]+hwhm_est,[0,np.max(new_histo[0])],'k--')
##    ax.plot([pcts[1],pcts[1]],[0,500],'k--')
##    ax.plot([pcts[2],pcts[2]],[0,500],'k--')
#    
##    ax.plot(np.diff(new_histo[0])/np.max(new_histo[0]))
##    ax.set(xlabel='m2q', ylabel='counts', ylim=[-1, 1])
#    ax.grid()
#    plt.pause(1.5)
#    
#
#
##pcts = np.percentile(x,[25,50,75])
##
##ax.plot([pcts[0],pcts[0]],[0,500],'k--')
##ax.plot([pcts[1],pcts[1]],[0,500],'k--')
##ax.plot([pcts[2],pcts[2]],[0,500],'k--')
##
#
## Bin until I get 1000 cts in peak
#
## Find mode
#
## Find an estimate of the FWHM
#
## Use mean shift to find peak
#
#
#
#
#
#
#
#
#
#
#
#
#
#kde = gaussian_kde(x,bw_method=0.001)
#
#
##grid = GridSearchCV(KernelDensity(kernel='gaussian'),
##                    {'bandwidth': np.logspace(-5, 0, 8)},
##                    cv=2) # 20-fold cross-validation
##grid.fit(x[:, None])
##print(grid.best_params_)
#
#
#
##kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
##kde.fit(x[:, None])
#
## score_samples returns the log of the probability density
#
#m2q_roi = [0.95, 1.75]
#m2q_bin_width = 0.001
#m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#
#x_d = np.linspace(m2q_roi[0], m2q_roi[1], m2q_num_bins)
#
#
#fig = plt.figure(num=128)
#fig.clear()
#ax = plt.axes()
#
#ax.fill_between(x_d, 10*kde(x_d), alpha=0.5)
##ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
##ax.ylim(-0.02, 0.22)
#
#
#
#
#for idx, pk_loc in enumerate(exp_peak_m2q):
#    found_peak_m2q[idx] = np.mean(m2q_corr[(m2q_corr<pk_loc*1.0015) & (m2q_corr>pk_loc*0.9985)])
#    
#
#
#m2q_corr_tmp = m2q_corr[(m2q_corr>0.95) & (m2q_corr<1.75)]
#
#
#nbins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20])
#pk_est = np.zeros(nbins.shape)
#for idx, nbin in enumerate(nbins):
#    tmp_histo = np.histogram(m2q_corr_tmp,range=(0, 2),bins=2*nbin,density=False)
#    max_idx = np.argmax(tmp_histo[0])
#    pk_est[idx] = tmp_histo[1][max_idx]
#    
#    
#
#
#fig = plt.figure(num=64)
#fig.clear()
#ax = plt.axes()
#
#ax.plot(pk_est,'-o')
#
##ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 200],
##       title='My First Plot')
#
#    
#    
#
#
#
#
#
#
#
#
#
#
#
