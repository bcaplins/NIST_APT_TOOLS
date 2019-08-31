# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:38:00 2019

@author: bwc
"""

import numpy as np
from scipy.optimize import brute
from scipy.optimize import minimize

import peak_param_determination as ppd

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def align_m2q_to_ref_m2q(ref_m2q,tof):
    
    
    
    
    
    ROI = [0.5, 120]
    BIN_SIZE = 0.01
    
    N_BIN = int(np.rint((ROI[1]-ROI[0])/BIN_SIZE))
    

    SCALES = np.array([1e-4, 1])

#    com_ref = np.mean(ref_m2q)
#    com = np.mean(tof**2)

    t0 = 0
#    c = com_ref/com

    com_ref = np.median(ref_m2q)
    com = np.median(tof**2)
    c = com_ref/com    
#    c = 1.39325423*1e-4


    h1, e1 = np.histogram(ref_m2q,bins=N_BIN,density=True)
    h2, e2 = np.histogram(tof**2,bins=N_BIN,density=True)
    
    h1 = h1*np.diff(e1)[0]
    h2 = h2*np.diff(e2)[0]
    
        
#    fig = plt.figure(num=105)
#
#    ax = plt.gca()
#    ax.clear()
#    ax.plot(e1[1:],h1,label='ref')
#    ax.plot(e2[1:],h2,label='curr')
#    
#    plt.pause(5)
    
    
    NP = 20;
    com_ref = np.mean(np.power(h1,NP)*e1[1:])/np.sum(np.power(h1,NP))
    com = np.mean(np.power(h2,NP)*e2[1:])/np.sum(np.power(h2,NP))
    c = com_ref/com    
    
    print("c_guess = "+str(c))
#    c = 1.39325423*1e-4



    
    # Brute force search to find correct minimum
    ref_histo = np.histogram(ref_m2q,range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]
    curr_histo = lambda p: np.histogram(mod_physics_m2q_calibration(np.array([p[0],t0]),tof),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]
    
    
#    fig = plt.figure(num=103)
#
#    ax = plt.gca()
#    ax.clear()
#    ax.plot(ref_histo,label='ref')
#    ax.plot(curr_histo(np.array([c/SCALES[0], 0])),label='curr')
#    
#    plt.pause(1)
    opt_fun = lambda p: -1*BIN_SIZE*np.sum(ref_histo*curr_histo(p))
    
    N_SLICE = 128
    fractional_range = 0.2
    
    p_range = ( ((1-fractional_range)*c/SCALES[0], (1+fractional_range)*c/SCALES[0]),)
    res = brute(opt_fun,
                p_range,
                Ns=N_SLICE,
                disp=True,
                full_output=True,
                finish=None)
    print('Optimization terminated!!!')
    print('     Current function value: '+str(res[1]))
    print('     Current parameter value: '+str(res[0]))


    # Minimize with simplex method 
    curr_histo = lambda p: np.histogram(mod_physics_m2q_calibration(p,tof),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]
    
    opt_fun = lambda p: -1*BIN_SIZE*np.sum(ref_histo*curr_histo(p))
    
#    cb = lambda xk: print([xk, opt_fun(xk)])
    cb = lambda xk: 0
 
    p_guess = np.array([res[0], t0/SCALES[1]])

    
    opts = {'xatol' : 1e-8,
            'fatol' : 1e-12,
            'maxiter' : 512,
            'maxfev' : 512,
            'disp' : True}
    res = minimize(opt_fun, 
                   p_guess,
                   options=opts,
                   method='Nelder-Mead', 
                   callback=cb)  
    
    new_m2q = mod_physics_m2q_calibration(res.x,tof)
    return (new_m2q, res.x)

def mod_physics_m2q_calibration(p_in,tof):
    SCALES = np.array([1e-4, 1])
    p_in = p_in*SCALES
    new_m2q = p_in[0]*((tof-p_in[1])**2)
    return new_m2q





def calibrate_m2q_by_peak_location(m2q, ref_peak_m2qs):
    
    found_peak_m2qs = np.zeros_like(ref_peak_m2qs)
    
#    roi_fraction = 0.02
    roi_half_wid = 0.25
    
    for idx in range(ref_peak_m2qs.size):
        m2q_roi = np.array([ref_peak_m2qs[idx]-roi_half_wid ,ref_peak_m2qs[idx]+roi_half_wid])
#        m2q_roi = ref_peak_m2qs[idx]*np.array([1-roi_fraction,1+roi_fraction])
        x = m2q[(m2q>m2q_roi[0]) & (m2q<m2q_roi[1])]
        found_peak_m2qs[idx] = ppd.mean_shift_peak_location(x)
    
    f = interp1d(found_peak_m2qs,ref_peak_m2qs,
                 kind='linear',
                 copy=True,
                 fill_value='extrapolate',
                 assume_sorted=False)
    m2q_corr = f(m2q)

    return m2q_corr
