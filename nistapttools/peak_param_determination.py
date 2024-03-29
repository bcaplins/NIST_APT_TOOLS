# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:40:36 2019

@author: bwc
"""

import numpy as np

from scipy.signal.windows import gaussian
from histogram_functions import bin_dat
from scipy.optimize import least_squares
from scipy.optimize import minimize
from constrNMPy import constrNM

#from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Prints the composition data structure in a 
def pretty_print_compositions(compositions,pk_data):
    keys = list(pk_data.dtype.fields.keys())
    keys.remove('m2q')
    
    col_wid = 12
    
    out = "bg method".ljust(col_wid)
    for key in keys:
        out += (key+" %").ljust(col_wid)
        out += ("d_"+key+" %").ljust(col_wid)
    
    bg_labels = ["none","local","global"]
    
    for bg_idx,label in enumerate(bg_labels):
        out += "\n"
        out += label.ljust(col_wid)

        for key_idx,key in enumerate(keys):
            out += '{:.4f}'.format(100*compositions[bg_idx][0][key_idx]).ljust(col_wid)
            out += '{:.4f}'.format(100*compositions[bg_idx][1][key_idx]).ljust(col_wid)
    
    print(out)
    return None

def compute_comp_and_error(cts,cts_std):
    dfdn = np.zeros((cts.size,cts.size))
    Ntot = np.sum(cts)
    
    # Compute dfj/dNi matrix
    for i in np.arange(cts.size):
        for j in np.arange(cts.size):
            if i==j:
                dfdn[i,j] = (Ntot-cts[i])/Ntot**2
            else:
                dfdn[i,j] = -cts[i]/Ntot**2
                
    # Compute atomic fraction
    atom_frac = cts/Ntot
    
    atom_frac_std = np.zeros(atom_frac.shape)
    # Compute uncertainty of atomic fraction
    for i in np.arange(cts.size):
        atom_frac_std[i] = np.sqrt(np.sum(np.square(dfdn[i,:]*cts_std)))
    
    return (atom_frac,atom_frac_std)

def do_composition(pk_data,cts):
    
    # Create an array that is the stoichiometries for each peak
    # the number of rows is the number of peaks
    # the number of cols is the number of unique atoms
    keys = list(pk_data.dtype.fields.keys())
    keys.remove('m2q')
    
    stoich = None
    for key in keys:
        if stoich is None:
            stoich = pk_data[key][:,None]
        else:
            stoich = np.c_[stoich,pk_data[key][:,None]]
    
    tot_cts = cts['total'][:,None]
    tot_cts_std = np.sqrt(cts['total'][:,None])
    
    if len(list(pk_data.dtype.fields.keys()))>1:
        pk_cts_loc = cts['total'][:,None]-cts['local_bg'][:,None]
        pk_cts_loc_std = np.sqrt(cts['total'][:,None]+cts['local_bg'][:,None])
    
        pk_cts_glob = cts['total'][:,None]-cts['global_bg'][:,None]
        pk_cts_glob_std = np.sqrt(cts['total'][:,None]+cts['global_bg'][:,None])
    

    tot_res = np.sum(stoich*tot_cts,axis=0)
    tot_res_std = np.sum(stoich*tot_cts_std,axis=0)    
    tot_stoich = compute_comp_and_error(tot_res,tot_res_std)

    if len(list(pk_data.dtype.fields.keys()))>1:    
        loc_res = np.sum(stoich*pk_cts_loc,axis=0)
        loc_res_std = np.sum(stoich*pk_cts_loc_std,axis=0)        
        loc_stoich = compute_comp_and_error(loc_res,loc_res_std)
               
        glob_res = np.sum(stoich*pk_cts_glob,axis=0)
        glob_res_std = np.sum(stoich*pk_cts_glob_std,axis=0)
        glob_stoich = compute_comp_and_error(glob_res,glob_res_std)
        
    
    if len(list(pk_data.dtype.fields.keys()))>1:
        ret_val = (tot_stoich,loc_stoich,glob_stoich)
    else:
        ret_val = (tot_stoich)
    
    return ret_val


def do_counting(epos, pk_params, glob_bg_param):
    
    xs, _ = bin_dat(epos['m2q'],user_roi=[0,150],isBinAligned=True)
    
    glob_bg = physics_bg(xs,glob_bg_param)    

    
    cts = np.full(pk_params.size,-1,dtype=[('total','f4'),
                                            ('local_bg','f4'),
                                            ('global_bg','f4')])
    
    for idx,pk_param in enumerate(pk_params):
        
        pk_rng = [pk_param['pre_rng'],pk_param['post_rng']]
        
#        print('pk rng',pk_rng)
        local_bg_cts = pk_param['loc_bg']*(pk_rng[1]-pk_rng[0])/0.001
        global_bg_cts = np.sum(glob_bg[(xs>=pk_rng[0]) & (xs<=pk_rng[1])])
        tot_cts = np.sum((epos['m2q']>=pk_rng[0]) & (epos['m2q']<=pk_rng[1]))
#        print('tot counts',tot_cts)
        
        cts['total'][idx] = tot_cts
        cts['local_bg'][idx] = local_bg_cts
        cts['global_bg'][idx] = global_bg_cts
        
    return cts 


def do_counting_csv_simple(xs, ys, pk_params):
    
    
    cts = np.full(pk_params.size,-1,dtype=[('total','f4'),
                                            ('local_bg','f4'),
                                            ('global_bg','f4')])
    
    for idx,pk_param in enumerate(pk_params):
        
        pk_rng = [pk_param['pre_rng'],pk_param['post_rng']]
        
        tot_cts = ys[(xs>=pk_rng[0]) & (xs<=pk_rng[1])].sum()
        
        cts['total'][idx] = tot_cts
        cts['local_bg'][idx] = 0
        cts['global_bg'][idx] = 0
        
    return cts 

def do_counting_csv(xs, ys, pk_params, glob_bg_param):
    
    
    
    glob_bg = physics_bg(xs,glob_bg_param)    

    
    cts = np.full(pk_params.size,-1,dtype=[('total','f4'),
                                            ('local_bg','f4'),
                                            ('global_bg','f4')])
    
    for idx,pk_param in enumerate(pk_params):
        
        pk_rng = [pk_param['pre_rng'],pk_param['post_rng']]
        
#        print('pk rng',pk_rng)
        local_bg_cts = pk_param['loc_bg']*(pk_rng[1]-pk_rng[0])/0.001
        global_bg_cts = glob_bg[(xs>=pk_rng[0]) & (xs<=pk_rng[1])].sum()
        tot_cts = ys[(xs>=pk_rng[0]) & (xs<=pk_rng[1])].sum()
        
        
#        print('tot counts',tot_cts)
        
        cts['total'][idx] = tot_cts
        cts['local_bg'][idx] = local_bg_cts
        cts['global_bg'][idx] = global_bg_cts
        
    return cts 



def get_peak_ranges_csv(xs_full_1mDa, ys_full_1mDa, peak_m2qs,peak_height_fraction=0.01, glob_bg_param=0):

    # Initialize a peak paramter array
    pk_params = np.full(peak_m2qs.size,-1,dtype=[('x0_nominal','f4'),
                                                 ('x0_g_fit','f4'),
                                                 ('x0_mean_shift','f4'),
                                                ('std_fit','f4'),
                                                ('off','f4'),
                                                ('amp','f4'),
                                                ('pre_rng','f4'),
                                                ('post_rng','f4'),
                                                ('pre_bg_rng','f4'),
                                                ('post_bg_rng','f4'),
                                                ('loc_bg','f4')])
    pk_params['x0_nominal'] = peak_m2qs
    
        
    ys_full_5mDa_sm = do_smooth_with_gaussian(ys_full_1mDa, std=5)
    
    N_kern = 250;
    ys_full_fwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern)
    ys_full_bwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern,reverse=True)
    
    ys_glob_bg_1mDa = physics_bg(xs_full_1mDa,glob_bg_param)
    
    # Get estiamtes for x0, amp, std_fit
    for idx,pk_param in enumerate(pk_params):
        # Select peak roi
        roi_half_wid = 0.5
        pk_roi = np.array([-roi_half_wid, roi_half_wid])+pk_param['x0_nominal']
        
        
        pk_idxs = np.where((xs_full_1mDa>pk_roi[0]) & (xs_full_1mDa<pk_roi[1]))[0]
        
        
        # Fit to gaussian
        smooth_param = 5
        popt = fit_to_g_off_csv(xs_full_1mDa[pk_idxs],ys_full_1mDa[pk_idxs],user_std=smooth_param)
        
        pk_param['amp'] = popt[0]
        pk_param['x0_g_fit'] = popt[1]
        pk_param['std_fit'] = popt[2]
        pk_param['off'] = popt[3]
#        print('gaussian peak loc',pk_param['x0_g_fit'])
        
        
        pk_idxs = np.where((xs_full_1mDa>(pk_param['x0_g_fit']-0.5*pk_param['std_fit'])) & (xs_full_1mDa<(pk_param['x0_g_fit']+0.5*pk_param['std_fit'])))[0]
        popt = fit_to_g_off_csv(xs_full_1mDa[pk_idxs],ys_full_1mDa[pk_idxs],user_std=1)

        pk_param['x0_mean_shift'] = popt[1]
        
                
#        ax = plt.gca()
#        ax.plot(np.array([1,1])*pk_param['x0_mean_shift'],np.array([0, pk_param['amp']+pk_param['off']]),'k--')
        
#        plt.pause(0.001)
#        plt.pause(2)
    
        # select starting locations
        pk_idx = np.argmin(np.abs(pk_param['x0_mean_shift']-xs_full_1mDa))
        pk_lhs_idx = np.argmin(np.abs((pk_param['x0_mean_shift']-pk_param['std_fit'])-xs_full_1mDa))    
        pk_rhs_idx = np.argmin(np.abs((pk_param['x0_mean_shift']+pk_param['std_fit'])-xs_full_1mDa))
        
        # Create a peak amplitude estimate.  I use the average of the gaussian amplitude and the 5 mDa smoothed data.
        # Notice the gaussian amplitude has the baseline removed, and therefor I subtract the global background from the smoothed data.
        pk_amp = 0.5*(pk_param['amp'] + ys_full_5mDa_sm[pk_idx]-ys_glob_bg_1mDa[pk_idx])
        
        curr_val = ys_full_5mDa_sm[pk_idx]
        
        for i in np.arange(pk_lhs_idx,-1,-1):
            curr_val = ys_full_5mDa_sm[i]        
            
            # Note that the global background is subtracted off the current value.  One day I should make this more general...
            if ((curr_val-ys_glob_bg_1mDa[i]) < peak_height_fraction*pk_amp) and (pk_param['pre_rng']<0):
                # This is a range limit
                pk_param['pre_rng'] = xs_full_1mDa[i]
            
            if curr_val<ys_full_bwd_sm[i]:
                # Assume we are at the prepeak baseline noise
                if pk_param['pre_rng']<0:
                    pk_param['pre_rng'] = xs_full_1mDa[i]
                pk_param['pre_bg_rng'] = xs_full_1mDa[i]
                break
            
        curr_val = ys_full_5mDa_sm[pk_idx]
        for i in np.arange(pk_rhs_idx,xs_full_1mDa.size):
            curr_val = ys_full_5mDa_sm[i]        
                
            if ((curr_val-ys_glob_bg_1mDa[i]) < peak_height_fraction*pk_amp) and (pk_param['post_rng']<0):
                # This is a range limit
                pk_param['post_rng'] = xs_full_1mDa[i]
            
            if curr_val<ys_full_fwd_sm[i]:
                # Assume we are at the prepeak baseline noise
                if pk_param['post_rng']<0:
                    pk_param['post_rng'] = xs_full_1mDa[i]
                pk_param['post_bg_rng'] = xs_full_1mDa[i]
                break      
        
        pre_pk_rng = [pk_param['pre_bg_rng']-0.22,pk_param['pre_bg_rng']-0.02]
        
        
        bg_idxs = np.where((xs_full_1mDa>=pre_pk_rng[0]) & (xs_full_1mDa<pre_pk_rng[1]))[0]
        
        
        pk_param['loc_bg'] = np.sum(ys_full_1mDa[bg_idxs].sum())*0.001/(pre_pk_rng[1]-pre_pk_rng[0])

        
#        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
#        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
#    #    ax.plot(np.array([1,1])*(pk_param['x0']+1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    #    ax.plot(np.array([1,1])*(pk_param['x0']+5*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#        plt.pause(0.1)
    
#    ax.clear()
#    ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='5 mDa smooth')    
#    for idx,pk_param in enumerate(pk_params):
#        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
#        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
#        ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
#        ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
#    
#    ax.set_yscale('log')
#    ax.set(ylim=[0.1,1000])
    
    return pk_params









def get_peak_ranges(epos, peak_m2qs,peak_height_fraction=0.01, glob_bg_param=0):

    # Initialize a peak paramter array
    pk_params = np.full(peak_m2qs.size,-1,dtype=[('x0_nominal','f4'),
                                                 ('x0_g_fit','f4'),
                                                 ('x0_mean_shift','f4'),
                                                ('std_fit','f4'),
                                                ('off','f4'),
                                                ('amp','f4'),
                                                ('pre_rng','f4'),
                                                ('post_rng','f4'),
                                                ('pre_bg_rng','f4'),
                                                ('post_bg_rng','f4'),
                                                ('loc_bg','f4')])
    pk_params['x0_nominal'] = peak_m2qs
    
    
    
    
    
    
    full_roi = np.array([0, 150])
    xs_full_1mDa, ys_full_1mDa = bin_dat(epos['m2q'],user_roi=full_roi,isBinAligned=True)
    ys_full_5mDa_sm = do_smooth_with_gaussian(ys_full_1mDa, std=5)
    
    N_kern = 250;
    ys_full_fwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern)
    ys_full_bwd_sm = forward_moving_average(ys_full_1mDa,n=N_kern,reverse=True)
    
    ys_glob_bg_1mDa = physics_bg(xs_full_1mDa,glob_bg_param)
    
    # Get estiamtes for x0, amp, std_fit
    for idx,pk_param in enumerate(pk_params):
        # Select peak roi
        roi_half_wid = 0.5
        pk_roi = np.array([-roi_half_wid, roi_half_wid])+pk_param['x0_nominal']
        pk_dat = epos['m2q'][(epos['m2q']>pk_roi[0]) & (epos['m2q']<pk_roi[1])]
        
        # Fit to gaussian
        smooth_param = 5
        popt = fit_to_g_off(pk_dat,user_std=smooth_param)
        pk_param['amp'] = popt[0]
        pk_param['x0_g_fit'] = popt[1]
        pk_param['std_fit'] = popt[2]
        pk_param['off'] = popt[3]
#        print('gaussian peak loc',pk_param['x0_g_fit'])
        
        
        pk_param['x0_mean_shift'] = mean_shift_peak_location(pk_dat,user_std=pk_param['std_fit'],user_x0=pk_param['x0_g_fit'])
        
#        ax = plt.gca()
#        ax.plot(np.array([1,1])*pk_param['x0_mean_shift'],np.array([0, pk_param['amp']+pk_param['off']]),'k--')
        
#        plt.pause(0.001)
#        plt.pause(2)
    
        # select starting locations
        pk_idx = np.argmin(np.abs(pk_param['x0_mean_shift']-xs_full_1mDa))
        pk_lhs_idx = np.argmin(np.abs((pk_param['x0_mean_shift']-pk_param['std_fit'])-xs_full_1mDa))    
        pk_rhs_idx = np.argmin(np.abs((pk_param['x0_mean_shift']+pk_param['std_fit'])-xs_full_1mDa))
        
        # Create a peak amplitude estimate.  I use the average of the gaussian amplitude and the 5 mDa smoothed data.
        # Notice the gaussian amplitude has the baseline removed, and therefor I subtract the global background from the smoothed data.
        pk_amp = 0.5*(pk_param['amp'] + ys_full_5mDa_sm[pk_idx]-ys_glob_bg_1mDa[pk_idx])
        
        curr_val = ys_full_5mDa_sm[pk_idx]
        
        for i in np.arange(pk_lhs_idx,-1,-1):
            curr_val = ys_full_5mDa_sm[i]        
            
            # Note that the global background is subtracted off the current value.  One day I should make this more general...
            if ((curr_val-ys_glob_bg_1mDa[i]) < peak_height_fraction*pk_amp) and (pk_param['pre_rng']<0):
                # This is a range limit
                pk_param['pre_rng'] = xs_full_1mDa[i]
            
            if curr_val<ys_full_bwd_sm[i]:
                # Assume we are at the prepeak baseline noise
                if pk_param['pre_rng']<0:
                    pk_param['pre_rng'] = xs_full_1mDa[i]
                pk_param['pre_bg_rng'] = xs_full_1mDa[i]
                break
            
        curr_val = ys_full_5mDa_sm[pk_idx]
        for i in np.arange(pk_rhs_idx,xs_full_1mDa.size):
            curr_val = ys_full_5mDa_sm[i]        
                
            if ((curr_val-ys_glob_bg_1mDa[i]) < peak_height_fraction*pk_amp) and (pk_param['post_rng']<0):
                # This is a range limit
                pk_param['post_rng'] = xs_full_1mDa[i]
            
            if curr_val<ys_full_fwd_sm[i]:
                # Assume we are at the prepeak baseline noise
                if pk_param['post_rng']<0:
                    pk_param['post_rng'] = xs_full_1mDa[i]
                pk_param['post_bg_rng'] = xs_full_1mDa[i]
                break      
        
        pre_pk_rng = [pk_param['pre_bg_rng']-0.22,pk_param['pre_bg_rng']-0.02]
        pk_param['loc_bg'] = np.sum((epos['m2q']>=pre_pk_rng[0]) & (epos['m2q']<=pre_pk_rng[1]))*0.001/(pre_pk_rng[1]-pre_pk_rng[0])

        
#        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
#        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0,1])*(pk_param['amp']+pk_param['off']),'k--')
#    #    ax.plot(np.array([1,1])*(pk_param['x0']+1*pk_param['std_fit'])   ,np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#    #    ax.plot(np.array([1,1])*(pk_param['x0']+5*pk_param['std_fit']),np.array([np.min(ys_smoothed),np.max(ys_smoothed)]),'k--')
#        plt.pause(0.1)
    
#    ax.clear()
#    ax.plot(xs_full_1mDa,ys_full_5mDa_sm,label='5 mDa smooth')    
#    for idx,pk_param in enumerate(pk_params):
#        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
#        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
#        ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
#        ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
#    
#    ax.set_yscale('log')
#    ax.set(ylim=[0.1,1000])
    
    return pk_params

def b_G(x,sigma,x0):
    return np.exp(-np.square(x-x0)/(2*np.square(sigma)))

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

# Assumes that the xs are linearly spaced
def physics_bg(xs,alpha):
    bin_size = np.mean(np.diff(xs))    
    EPS = 0.01
    return bin_size*alpha*np.reciprocal(np.sqrt(xs+EPS))

def get_glob_bg(m2qs,rois=None):
    if rois==None:
        rois = [[3.5,6.5]]
    
    # Check to see if just one roi. Nested lists are what the code expects
    if len(rois)==2:
        if type(rois[0]) is not list:
            rois = [rois]
    
    EPS = 0.01
    
    numerator = 0.0
    denominator = 0.0
    for roi in rois:
        upp = np.max(roi)
        low = np.min(roi)    
        numerator += m2qs[(m2qs>=low) & (m2qs<=upp)].size
        denominator += 2*(np.sqrt(upp+EPS)-np.sqrt(low+EPS))
    
    print('# of counts used to define global bg: ', numerator)
    print('global bg parameter error: ',100*np.sqrt(numerator)/numerator,' %')
    bg_param = numerator/denominator
    
#    bg_param_per_dalton = (epos['m2q'][(epos['m2q']>=roi[0]) & (epos['m2q']<=roi[1])].size) \
#                            /(2*(np.sqrt(roi[1]+EPS)-np.sqrt(roi[0]+EPS)))     
#    def bg_func(m2qs,m2q_bin_size):
#        return m2q_bin_size*bg_param/np.sqrt(m2qs+EPS)
        
    return bg_param

def get_glob_bg_csv(xs, ys,rois=None):
    if rois==None:
        rois = [[3.5,6.5]]
    
    # Check to see if just one roi. Nested lists are what the code expects
    if len(rois)==2:
        if type(rois[0]) is not list:
            rois = [rois]
    
    EPS = 0.01
    
    numerator = 0.0
    denominator = 0.0
    for roi in rois:
        upp = np.max(roi)
        low = np.min(roi)    
        numerator += ys[(xs>=low) & (xs<upp)].sum()
        denominator += 2*(np.sqrt(upp+EPS)-np.sqrt(low+EPS))
    
    print('# of counts used to define global bg: ', numerator)
    print('global bg parameter error: ',100*np.sqrt(numerator)/numerator,' %')
    bg_param = numerator/denominator
    
#    bg_param_per_dalton = (epos['m2q'][(epos['m2q']>=roi[0]) & (epos['m2q']<=roi[1])].size) \
#                            /(2*(np.sqrt(roi[1]+EPS)-np.sqrt(roi[0]+EPS)))     
#    def bg_func(m2qs,m2q_bin_size):
#        return m2q_bin_size*bg_param/np.sqrt(m2qs+EPS)
        
    return bg_param


#
#def fit_uncorr_bg(dat,fit_roi=[3.5,6.5]):
#
#    xs_full, ys_full = bin_dat(dat,user_roi=[0,100],isBinAligned=True)
#    
#    xs = xs_full[(xs_full>fit_roi[0]) & (xs_full<fit_roi[1])]
#    ys = ys_full[(xs_full>fit_roi[0]) & (xs_full<fit_roi[1])]
#    
#    opt_fun = lambda p: np.sum(np.square(physics_bg(xs,*p)-ys))
#    
#    p_guess = np.array([100])
#    
#    
#    
#    opts = {'xatol' : 1e-5,
#            'fatol' : 1e-12,
#             'maxiter' : 512,
#             'maxfev' : 512,
#             'disp' : True}
#    res = minimize(opt_fun, 
#                   p_guess,
#                   options=opts,
#                   method='Nelder-Mead')  
#    
#    mod_bg = physics_bg(xs_full,res.x)
#        
#    fig = plt.figure(num=1299)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs_full,ys_full)
#    ax.plot(xs_full,mod_bg)
##    
#    return res.x
    
    

def pk_mod_fun(x,amp,x0,sigma,b):
    return amp*b_G(x,sigma,x0)+b
        






def fit_to_g_off_csv(xs, ys, user_std, user_p0=np.array([])):
    
    if ys.sum()<32:
        raise Exception('NEED MORE COUNTS IN PEAK')
        
    std = user_std
        
    
    ys_smoothed = do_smooth_with_gaussian(ys,std)
    
    
    opt_fun = lambda p: np.sum(np.square(pk_mod_fun(xs, *p)-ys_smoothed))

    

#    def resid_func(p): 
#        return pk_mod_fun(xs,*p)-ys_smoothed
    
    N4 = ys_smoothed.size//4
    mx_idx = np.argmax(ys_smoothed[N4:(3*N4)])+N4
    
    if(user_p0.size == 0):
        p0 = np.array([ys_smoothed[mx_idx]-np.min(ys_smoothed), xs[mx_idx], 0.015,  np.percentile(ys_smoothed,20)])
    else:
        p0 = user_p0    
        
    # b_model2(x,amp_g,x0,sigma,b):
    lbs = np.array([0,       np.percentile(xs,10),  0.005,   0])
    ubs = np.array([2*p0[0],  np.percentile(xs,90), 0.50,   p0[0]])

    # Force in bounds
    p_guess = np.sort(np.c_[lbs,p0,ubs])[:,1]
    
    
#    popt2, pcov =  = curve_fit(pk_mod_fun, xs, ys_smoothed, p0=p_guess, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    popt2, pcov =  = curve_fit(pk_mod_fun, xs, ys_smoothed)
    
    
#    bnds = ((0,2*p0[0]),
#            (np.percentile(xs,10),np.percentile(xs,90)),
#             (0.007,0.1),
#             (0,p0[0]))
#    

#    opts = {'xatol' : 1e-5,
#            'fatol' : 1e-12,
#             'maxiter' : 1024,
#             'maxfev' : 1024,
#             'disp' : True}
    
    
    ret_dict = constrNM(opt_fun,p_guess,lbs,ubs,xtol=1e-5, ftol=1e-12, maxiter=1024, maxfun=1024, full_output=1, disp=0)
    
#    print(p_guess)
#    print(ret_dict['xopt'])
    
#    res = minimize(opt_fun, 
#                   p_guess,
#                   options=opts,
##                   bounds=bnds,
#                   method='Nelder-Mead')  
#
#    if res.x[1]<np.percentile(xs,10) or res.x[1]>np.percentile(xs,90):
#        res.x = p_guess


#    print(np.abs(res.x))
    
    
#    popt2 = least_squares(resid_func, x0=p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    curve_fit(pk_mod_fun, xs, ys_smoothed, p0=p0)
    
#    popt = least_squares(resid_func, p0, verbose=0, ftol=1e-12, max_nfev=2048)
#
#    fig = plt.figure(num=999)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.',label='raw')
#    ax.plot(xs,ys_smoothed,label='smoothed')
#    
#    mod_y = pk_mod_fun(xs,*p_guess)
#    ax.plot(xs,mod_y,label='guess')
#    
#    
#    mod_y = pk_mod_fun(xs,*res.x)
#    
#    ax.plot(xs,mod_y,label='fit')
#    
#    ax.legend()
#    
#    
#    
#    plt.pause(.001)
#
#    # Find halfway down and up each side
#    max_idx = np.argmax(N)
#    max_val = N[max_idx]
#    
#    lhs_dat = N[0:max_idx]
#    rhs_dat = N[max_idx:]
#    
#    lhs_idx = np.argmin(np.abs(lhs_dat-max_val/2))
#    rhs_idx = np.argmin(np.abs(rhs_dat-max_val/2))+max_idx
##    
#    hwhm_est = (x[rhs_idx]-x[lhs_idx])/2
#
#
    return np.abs(ret_dict['xopt'])




def fit_to_g_off(dat, user_std, user_p0=np.array([])):
    
    if dat.size<32:
        raise Exception('NEED MORE COUNTS IN PEAK')
        
    std = user_std
        
    xs,ys = bin_dat(dat)
    
    ys_smoothed = do_smooth_with_gaussian(ys,std)
    
    
    opt_fun = lambda p: np.sum(np.square(pk_mod_fun(xs, *p)-ys_smoothed))

    

#    def resid_func(p): 
#        return pk_mod_fun(xs,*p)-ys_smoothed
    
    N4 = ys_smoothed.size//4
    mx_idx = np.argmax(ys_smoothed[N4:(3*N4)])+N4
    
    if(user_p0.size == 0):
        p0 = np.array([ys_smoothed[mx_idx]-np.min(ys_smoothed), xs[mx_idx], 0.015,  np.percentile(ys_smoothed,20)])
    else:
        p0 = user_p0    
        
    # b_model2(x,amp_g,x0,sigma,b):
    lbs = np.array([0,       np.percentile(xs,10),  0.005,   0])
    ubs = np.array([2*p0[0],  np.percentile(xs,90), 0.50,   p0[0]])

    # Force in bounds
    p_guess = np.sort(np.c_[lbs,p0,ubs])[:,1]
    
    
#    popt2, pcov =  = curve_fit(pk_mod_fun, xs, ys_smoothed, p0=p_guess, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    popt2, pcov =  = curve_fit(pk_mod_fun, xs, ys_smoothed)
    
    
#    bnds = ((0,2*p0[0]),
#            (np.percentile(xs,10),np.percentile(xs,90)),
#             (0.007,0.1),
#             (0,p0[0]))
#    

#    opts = {'xatol' : 1e-5,
#            'fatol' : 1e-12,
#             'maxiter' : 1024,
#             'maxfev' : 1024,
#             'disp' : True}
    
    
    ret_dict = constrNM(opt_fun,p_guess,lbs,ubs,xtol=1e-5, ftol=1e-12, maxiter=1024, maxfun=1024, full_output=1, disp=0)
    
#    print(p_guess)
#    print(ret_dict['xopt'])
    
#    res = minimize(opt_fun, 
#                   p_guess,
#                   options=opts,
##                   bounds=bnds,
#                   method='Nelder-Mead')  
#
#    if res.x[1]<np.percentile(xs,10) or res.x[1]>np.percentile(xs,90):
#        res.x = p_guess


#    print(np.abs(res.x))
    
    
#    popt2 = least_squares(resid_func, x0=p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    curve_fit(pk_mod_fun, xs, ys_smoothed, p0=p0)
    
#    popt = least_squares(resid_func, p0, verbose=0, ftol=1e-12, max_nfev=2048)
#
#    fig = plt.figure(num=999)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.',label='raw')
#    ax.plot(xs,ys_smoothed,label='smoothed')
#    
#    mod_y = pk_mod_fun(xs,*p_guess)
#    ax.plot(xs,mod_y,label='guess')
#    
#    
#    mod_y = pk_mod_fun(xs,*res.x)
#    
#    ax.plot(xs,mod_y,label='fit')
#    
#    ax.legend()
#    
#    
#    
#    plt.pause(.001)
#
#    # Find halfway down and up each side
#    max_idx = np.argmax(N)
#    max_val = N[max_idx]
#    
#    lhs_dat = N[0:max_idx]
#    rhs_dat = N[max_idx:]
#    
#    lhs_idx = np.argmin(np.abs(lhs_dat-max_val/2))
#    rhs_idx = np.argmin(np.abs(rhs_dat-max_val/2))+max_idx
##    
#    hwhm_est = (x[rhs_idx]-x[lhs_idx])/2
#
#
    return np.abs(ret_dict['xopt'])
#    return np.abs(res.x)





def do_smooth_with_gaussian(hist_dat, std):
     
    window_len = int(4*std)+1
    
    if window_len>2*hist_dat.size-1:
        raise Exception('Whoa buddy!')
    kern = gaussian(window_len,std)
    kern /= np.sum(kern)
    
    ys1 = hist_dat
    ys1 = np.r_[ys1[(window_len-1)//2:0:-1],ys1,ys1[-2:(-window_len-3)//2:-1]]

    ys1 = np.convolve(ys1,kern,'valid')
    return ys1


def mean_shift_peak_location(dat,user_std=-1,user_x0=-1):
    if user_std>0:
        hwhm = np.sqrt(2*np.log(2))*user_std
    else:
        popt = fit_to_g_off(dat,user_std=5)
        #amp_g,x0,sigma,b
        hwhm = np.max([np.sqrt(2*np.log(2))*popt[2], 0.005])
#        hwhm = est_hwhm2(dat)
    
    prev_est = np.inf
    
    if user_x0>0:
        curr_est = user_x0
    else:
        curr_est = np.mean(dat)
        
            

    
    prev_est = 1e9
    loop_count = 0
    while np.abs(curr_est-prev_est)/curr_est >= 1e-6:
        loop_count += 1
        prev_est = curr_est
#        print('curr_est: '+str(curr_est))
#        print('hwhm: '+str(hwhm))
        idxs = np.where((dat>curr_est-hwhm) & (dat<curr_est+hwhm))[0]
        if(idxs.size==0):
            print('ERRROOR')
            print('curr_est = '+str(curr_est))
            print('prev_est = '+str(prev_est))
            print('hwhm = '+str(hwhm))
            print(dat)
        curr_est = np.mean(dat[idxs])
        
        
        
        
        if loop_count>64:
            print('Exiting mean shift after 64 iterations (before convergence)')
            break
        
    return curr_est

#
#def est_baseline_empirical(xs,ys):
#    return np.percentile(ys,20)

#
#def get_range_empirical(xs,ys_smoothed):
#    N = xs.size
#    N4 = N//4
#    
#    max_idx = np.argmax(ys_smoothed[N4:3*N4])+N4
#    max_val = ys_smoothed[max_idx]
#    
#    ax = plt.gca()
#    ax.plot(xs[max_idx],ys_smoothed[max_idx],'o')
#    
#    bl_est = est_baseline_empirical(xs,ys_smoothed)
#    
#    search_val = 0.5*(max_val-bl_est)+bl_est
#    
#    lims = np.zeros(2)
#    # Search right
#    for idx in np.arange(max_idx+10,N-1):
#        if (ys_smoothed[idx]>search_val) & (ys_smoothed[idx+1]<search_val):
#            lims[1] = np.interp(search_val,ys_smoothed[idx:idx+2],xs[idx:idx+2])
#            break
#        if ys_smoothed[idx] < ys_smoothed[idx+1]:
#            lims[1] = xs[idx]
#            break
#        
#    # Search left
#    for idx in np.arange(max_idx-1-10,-1,-1):
#        if (ys_smoothed[idx]>search_val) & (ys_smoothed[idx-1]<search_val):
#            lims[0] = np.interp(search_val,ys_smoothed[idx:idx-2:-1],xs[idx:idx-2:-1])
#            break
#        if ys_smoothed[idx] < ys_smoothed[idx-1]:
#            lims[0] = xs[idx]
#            break
#    
#    return lims


#
#def get_peak_count(dat,rng):
#    cts = dat[(dat>=rng[0]) & (dat<=rng[1])].size
#    return cts

#def get_peak_count(dat,rng,bl_est_per_mDa):
#    bg_cts = 1000*(rng[1]-rng[0])*bl_est_per_mDa
#    tot_cts = dat[(dat>=rng[0]) & (dat<=rng[1])].size
#    pk_cts = tot_cts-bg_cts
#    return np.array([pk_cts, bg_cts])
    



#
#def est_hwhm(dat):
#    
#    if dat.size<1024:
#        raise Exception('NEED MORE COUNTS IN PEAK')
#    
#    roi = np.array([np.min(dat), np.max(dat)])
#    bin_width = roi[1]-roi[0]
#    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
#
#    while True:
#        histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
#        max_val = np.max(histo[0])
#        if(max_val>=128):
#            bin_width = bin_width/2
#            num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
#        else:
#            break
#        if bin_width<1e-6:
#            raise Exception('Not sure what is going on now...')
#    
#    bin_width = bin_width*2
#    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
#    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
#    
#    N = histo[0]
#    x = (histo[1][1:]+histo[1][0:-1])/2
#    
#    # Find halfway down and up each side
#    max_idx = np.argmax(N)
#    max_val = N[max_idx]
#    
#    lhs_dat = N[0:max_idx]
#    rhs_dat = N[max_idx:]
#    
#    lhs_idx = np.argmin(np.abs(lhs_dat-max_val/2))
#    rhs_idx = np.argmin(np.abs(rhs_dat-max_val/2))+max_idx
##    
#    hwhm_est = (x[rhs_idx]-x[lhs_idx])/2
#
#    return hwhm_est


#
#def smooth_with_gaussian_CV(dat):
#    N_tries = 32
#    
#    roi = [np.min(dat),np.max(dat)]
#    bin_width = .001
#    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
#    
#    stds = np.logspace(np.log10(5),np.log10(50),16)
#    errs = np.zeros_like(stds)
#    
#    idxs = np.arange(dat.size)
#    for try_idx in np.arange(N_tries):
#        
#        np.random.shuffle(idxs)
#        
#        split_size = dat.size // 2
#        
#        dat1 = dat[0:split_size]
#        dat2 = dat[split_size:2*split_size]
#        
#        histo1 = np.histogram(dat1,range=(roi[0], roi[1]),bins=num_bins,density=False)
#        histo2 = np.histogram(dat2,range=(roi[0], roi[1]),bins=num_bins,density=False)
#        
#        xs = (histo1[1][1:]+histo1[1][0:-1])/2
#        
#        ys2 = histo2[0]    
#        
#        for std_idx,std in enumerate(stds):
#            ys1 = do_smooth_with_gaussian(histo1[0],std)
#            errs[std_idx] += np.sum(np.square(ys2-ys1))
#    
#    best_idx = np.argmin(errs)
#    best_std = stds[best_idx]/np.sqrt(2)
#    
#    fig = plt.figure(num=100)
#    fig.clear()
#    ax = plt.axes()
#    ax.plot(stds,errs)
#    ax.set_xscale('log')    
#    ax.set_yscale('log')    
#    
#    plt.pause(0.001)
#    
#    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
#    
#    ys1 = do_smooth_with_gaussian(histo[0],best_std)
#    ys2 = histo[0]
#
#    fig = plt.figure(num=101)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys2,'.')
#    ax.plot(xs,ys1)
#    
#    
##    ax.set_yscale('log')    
#    
#    plt.pause(0.001)
#    
#    return (xs, ys1, best_std)
#

#
#def est_hwhm2(dat):
#    
#    if dat.size<256:
#        raise Exception('NEED MORE COUNTS IN PEAK')
#
#    xs, ys1, best_std = smooth_with_gaussian_CV(dat)    
#    xs,ys = bin_dat(dat)
#    
#    ys_smoothed = do_smooth_with_gaussian(ys,best_std)
#    
#    pk_mod_fun = lambda x,amp,x0,sigma,b: amp*b_G(x,sigma,x0)+b
#    resid_func = lambda p: pk_mod_fun(xs,*p)-ys_smoothed
#    
#    p0 = np.array([np.max(ys_smoothed)-np.min(ys_smoothed), np.percentile(xs,50), 0.025,  np.min(ys_smoothed)])
#        
#    # b_model2(x,amp_g,x0,sigma,b):
#    lbs = np.array([0,       np.percentile(xs,25),  0.01,   0])
#    ubs = np.array([2*p0[0],  np.percentile(xs,75), 0.10,   p0[0]])
#
#    # Force in bounds
#    p0 = np.sort(np.c_[lbs,p0,ubs])[:,1]
#    
#    
#    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
##    popt = least_squares(resid_func, p0, verbose=0, ftol=1e-12, max_nfev=2048)
#
#    fig = plt.figure(num=999)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.plot(xs,ys,'.',label='1 mDa')
#    ax.plot(xs,ys_smoothed,label='smooth')
#    
#    mod_y = pk_mod_fun(xs,*popt.x)
#    ax.plot(xs,mod_y,label='optimized')
#    
#    mod_y = pk_mod_fun(xs,*p0)
#    ax.plot(xs,mod_y,label='guess')
#    
#    ax.legend()
#    plt.pause(0.001)
##
##    # Find halfway down and up each side
##    max_idx = np.argmax(N)
##    max_val = N[max_idx]
##    
##    lhs_dat = N[0:max_idx]
##    rhs_dat = N[max_idx:]
##    
##    lhs_idx = np.argmin(np.abs(lhs_dat-max_val/2))
##    rhs_idx = np.argmin(np.abs(rhs_dat-max_val/2))+max_idx
###    
##    hwhm_est = (x[rhs_idx]-x[lhs_idx])/2
##
##
#
#    return popt.x[2]*np.sqrt(2*np.log(2))
