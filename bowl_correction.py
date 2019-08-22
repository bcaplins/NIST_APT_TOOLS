# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:50:01 2019

@author: bwc
"""

import numpy as np
from scipy.optimize import minimize

def mod_geometric_bowl_correction(p_in,tof,x_det,y_det):
    SCALES = np.array([1e2, 1e-4, 1e-4, 1e-3])
    
    p_in = p_in*SCALES

    r2 = x_det**2+y_det**2
    new_tof = tof/     \
        (np.sqrt(1+r2/p_in[0]**2)   \
        *(1+p_in[1]*x_det+p_in[2]*y_det+p_in[3]*(r2/30**2)**2))
    
    com0 = np.mean(tof)
    com = np.mean(new_tof)
    new_tof = (com0/com)*new_tof   
            
    return new_tof
 
def geometric_bowl_correction(tof,x_det,y_det,p_guess,ROI,TOF_BIN_SIZE):    
    SCALES = np.array([1e2, 1e-4, 1e-4, 1e-3])

    N_BIN = int(np.rint((ROI[1]-ROI[0])/TOF_BIN_SIZE))

    opt_fun = lambda p: -1*TOF_BIN_SIZE*np.sum(np.histogram(mod_geometric_bowl_correction(p,tof,x_det,y_det),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]**2)
#    cb = lambda xk: print(xk)
    cb = lambda xk: 0

    if not p_guess.size:
        p_guess = np.array([89, 5e-5, -8e-5, -3e-3])/SCALES

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
 