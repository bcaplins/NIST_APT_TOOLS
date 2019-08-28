# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:40:36 2019

@author: bwc
"""

import numpy as np

from scipy.signal.windows import gaussian
from helper_functions import bin_dat
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



def b_G(x,sigma,x0):
    return np.exp(-np.square(x-x0)/(2*np.square(sigma)))



def est_hwhm(dat):
    
    if dat.size<1024:
        raise Exception('NEED MORE COUNTS IN PEAK')
    
    roi = np.array([np.min(dat), np.max(dat)])
    bin_width = roi[1]-roi[0]
    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))

    while True:
        histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
        max_val = np.max(histo[0])
        if(max_val>=128):
            bin_width = bin_width/2
            num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
        else:
            break
        if bin_width<1e-6:
            raise Exception('Not sure what is going on now...')
    
    bin_width = bin_width*2
    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
    
    N = histo[0]
    x = (histo[1][1:]+histo[1][0:-1])/2
    
    # Find halfway down and up each side
    max_idx = np.argmax(N)
    max_val = N[max_idx]
    
    lhs_dat = N[0:max_idx]
    rhs_dat = N[max_idx:]
    
    lhs_idx = np.argmin(np.abs(lhs_dat-max_val/2))
    rhs_idx = np.argmin(np.abs(rhs_dat-max_val/2))+max_idx
#    
    hwhm_est = (x[rhs_idx]-x[lhs_idx])/2

    return hwhm_est




def fit_to_g_off(dat, user_std=-1, user_p0=np.array([])):
    
    if dat.size<256:
        raise Exception('NEED MORE COUNTS IN PEAK')

    
    
    if user_std>0:
        std = user_std
    else:
        _, _, std = smooth_with_gaussian_CV(dat)    
        
    xs,ys = bin_dat(dat)
    
    ys_smoothed = do_smooth_with_gaussian(ys,std)
    
    pk_mod_fun = lambda x,amp,x0,sigma,b: amp*b_G(x,sigma,x0)+b
    resid_func = lambda p: pk_mod_fun(xs,*p)-ys_smoothed
    
    if(user_p0.size == 0):
        p0 = np.array([np.max(ys_smoothed)-np.min(ys_smoothed), np.percentile(xs,50), 0.025,  np.min(ys_smoothed)])
    else:
        p0 = user_p0    
        
    # b_model2(x,amp_g,x0,sigma,b):
    lbs = np.array([0,       np.percentile(xs,10),  0.01,   0])
    ubs = np.array([2*p0[0],  np.percentile(xs,90), 0.10,   p0[0]])

    # Force in bounds
    p0 = np.sort(np.c_[lbs,p0,ubs])[:,1]
    
    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    popt = least_squares(resid_func, p0, verbose=0, ftol=1e-12, max_nfev=2048)

    fig = plt.figure(num=999)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys,'.')
    ax.plot(xs,ys_smoothed)
    
    mod_y = pk_mod_fun(xs,*popt.x)
    ax.plot(xs,mod_y)
    
    plt.pause(.001)
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

    return popt.x


def est_hwhm2(dat):
    
    if dat.size<256:
        raise Exception('NEED MORE COUNTS IN PEAK')

    xs, ys1, best_std = smooth_with_gaussian_CV(dat)    
    xs,ys = bin_dat(dat)
    
    ys_smoothed = do_smooth_with_gaussian(ys,best_std)
    
    pk_mod_fun = lambda x,amp,x0,sigma,b: amp*b_G(x,sigma,x0)+b
    resid_func = lambda p: pk_mod_fun(xs,*p)-ys_smoothed
    
    p0 = np.array([np.max(ys_smoothed)-np.min(ys_smoothed), np.percentile(xs,50), 0.025,  np.min(ys_smoothed)])
        
    # b_model2(x,amp_g,x0,sigma,b):
    lbs = np.array([0,       np.percentile(xs,25),  0.01,   0])
    ubs = np.array([2*p0[0],  np.percentile(xs,75), 0.10,   p0[0]])

    # Force in bounds
    p0 = np.sort(np.c_[lbs,p0,ubs])[:,1]
    
    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    popt = least_squares(resid_func, p0, verbose=0, ftol=1e-12, max_nfev=2048)

    fig = plt.figure(num=999)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys,'.')
    ax.plot(xs,ys_smoothed)
    
    mod_y = pk_mod_fun(xs,*popt.x)
    ax.plot(xs,mod_y)
    
    plt.pause(0.001)
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

    return popt.x[2]*np.sqrt(2*np.log(2))




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

def smooth_with_gaussian_CV(dat):
    N_tries = 32
    
    roi = [np.min(dat),np.max(dat)]
    bin_width = .001
    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
    
    stds = np.logspace(np.log10(5),np.log10(50),16)
    errs = np.zeros_like(stds)
    
    idxs = np.arange(dat.size)
    for try_idx in np.arange(N_tries):
        
        np.random.shuffle(idxs)
        
        split_size = dat.size // 2
        
        dat1 = dat[0:split_size]
        dat2 = dat[split_size:2*split_size]
        
        histo1 = np.histogram(dat1,range=(roi[0], roi[1]),bins=num_bins,density=False)
        histo2 = np.histogram(dat2,range=(roi[0], roi[1]),bins=num_bins,density=False)
        
        xs = (histo1[1][1:]+histo1[1][0:-1])/2
        
        ys2 = histo2[0]    
        
        for std_idx,std in enumerate(stds):
            ys1 = do_smooth_with_gaussian(histo1[0],std)
            errs[std_idx] += np.sum(np.square(ys2-ys1))
    
    best_idx = np.argmin(errs)
    best_std = stds[best_idx]/np.sqrt(2)
    
    fig = plt.figure(num=100)
    fig.clear()
    ax = plt.axes()
    ax.plot(stds,errs)
    ax.set_xscale('log')    
    ax.set_yscale('log')    
    
    plt.pause(0.001)
    
    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
    
    ys1 = do_smooth_with_gaussian(histo[0],best_std)
    ys2 = histo[0]

    fig = plt.figure(num=101)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys2,'.')
    ax.plot(xs,ys1)
    
    
#    ax.set_yscale('log')    
    
    plt.pause(0.001)
    
    return (xs, ys1, best_std)



def mean_shift_peak_location(dat,user_std=-1,user_x0=-1):
    if user_std>0:
        hwhm = np.sqrt(2*np.log(2))*user_std
    else:
        hwhm = est_hwhm2(dat)
    
    prev_est = np.inf
    
    if user_x0>0:
        curr_est = user_x0
    else:
        curr_est = np.mean(dat)
    
    loop_count = 0
    while np.abs(curr_est-prev_est)/curr_est >= 1e-6:
        loop_count += 1
        prev_est = curr_est
        curr_est = np.mean(dat[(dat>curr_est-hwhm) & (dat<curr_est+hwhm)])
        
        if loop_count>64:
            print('Exiting mean shift after 64 iterations (before convergence)')
            break
        
    return curr_est


def est_baseline_empirical(xs,ys):
    return np.percentile(ys,20)


def get_range_empirical(xs,ys_smoothed):
    N = xs.size
    N4 = N//4
    
    max_idx = np.argmax(ys_smoothed[N4:3*N4])+N4
    max_val = ys_smoothed[max_idx]
    
    ax = plt.gca()
    ax.plot(xs[max_idx],ys_smoothed[max_idx],'o')
    
    bl_est = est_baseline_empirical(xs,ys_smoothed)
    
    search_val = 0.5*(max_val-bl_est)+bl_est
    
    lims = np.zeros(2)
    # Search right
    for idx in np.arange(max_idx+10,N-1):
        if (ys_smoothed[idx]>search_val) & (ys_smoothed[idx+1]<search_val):
            lims[1] = np.interp(search_val,ys_smoothed[idx:idx+2],xs[idx:idx+2])
            break
        if ys_smoothed[idx] < ys_smoothed[idx+1]:
            lims[1] = xs[idx]
            break
        
    # Search left
    for idx in np.arange(max_idx-1-10,-1,-1):
        if (ys_smoothed[idx]>search_val) & (ys_smoothed[idx-1]<search_val):
            lims[0] = np.interp(search_val,ys_smoothed[idx:idx-2:-1],xs[idx:idx-2:-1])
            break
        if ys_smoothed[idx] < ys_smoothed[idx-1]:
            lims[0] = xs[idx]
            break
    
    return lims




def get_peak_count(dat,rng,bl_est_per_mDa):
    bg_cts = 1000*(rng[1]-rng[0])*bl_est_per_mDa
    tot_cts = dat[(dat>=rng[0]) & (dat<=rng[1])].size
    pk_cts = tot_cts-bg_cts
    return np.array([pk_cts, bg_cts])
    


