# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:46:43 2019

@author: bwc
"""

import numpy as np
from scipy.optimize import minimize

def mod_full_voltage_correction(p_in,tof,v_dc):
    SCALES = np.array([1e3, 1e-5])
    
    p_in = p_in*SCALES
    
    new_tof = tof*np.sqrt(p_in[0]+v_dc+p_in[1]*v_dc**2)
    
    com0 = np.mean(tof)
    com = np.mean(new_tof)
    new_tof = (com0/com)*new_tof
    
    return new_tof

def full_voltage_correction(tof,v_dc,p_guess,ROI,TOF_BIN_SIZE):
    SCALES = np.array([1e3, 1e-5])
    
    N_BIN = int(np.rint((ROI[1]-ROI[0])/TOF_BIN_SIZE))
    opt_fun = lambda p: -1*TOF_BIN_SIZE*np.sum(np.histogram(mod_full_voltage_correction(p,tof,v_dc),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]**2)
    
    cb = lambda xk: print(xk)
    cb = lambda xk: 0

    if not p_guess.size:
        raise Exception('START WITH A REASONABLE GUESS!!!')
    else:
        if(np.size(p_guess) == 1):
            p_guess = np.array([p_guess[0], -5e-6/SCALES[1]])
        
    opts = {'xatol' : 1e-5,
            'fatol' : 1e-12,
            'maxiter' : 512,
            'maxfev' : 512,
            'disp' : True}
    res = minimize(opt_fun, 
                   p_guess,
                   options=opts,
                   method='Nelder-Mead', 
                   callback=cb)  

    return res.x
    

def mod_basic_voltage_correction(p_in,tof,v_dc):
    SCALES = np.array([1e3])
    
    p_in = p_in*SCALES
        
    new_tof = tof*np.sqrt(p_in[0]+v_dc)
    
    com0 = np.mean(tof)
    com = np.mean(new_tof)
    new_tof = (com0/com)*new_tof
    
    return new_tof
     
def basic_voltage_correction(tof,v_dc,p_guess,ROI,TOF_BIN_SIZE):
    SCALES = np.array([1e3])
    
    N_BIN = int(np.rint((ROI[1]-ROI[0])/TOF_BIN_SIZE))
    opt_fun = lambda p: -1*TOF_BIN_SIZE*np.sum(np.histogram(mod_basic_voltage_correction(p,tof,v_dc),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]**2)
    
    cb = lambda xk: print(xk)
    cb = lambda xk: 0

    if not p_guess.size:
        p_guess = np.mean([-500, np.max(v_dc)])/SCALES
        
    opts = {'xatol' : 1e-5,
            'fatol' : 1e-12,
            'maxiter' : 512,
            'maxfev' : 512,
            'disp' : True}
    res = minimize(opt_fun, 
                   p_guess,
                   options=opts,
                   method='Nelder-Mead', 
                   callback=cb)  

    return res.x