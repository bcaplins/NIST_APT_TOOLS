# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute

def mod_full_vb_correction(epos, p_volt, p_bowl):
    return epos['tof']\
                *mod_full_voltage_correction(p_volt,
                                              np.ones_like(epos['tof']),
                                              epos['v_dc'])\
                *mod_geometric_bowl_correction(p_bowl,
                                                np.ones_like(epos['tof']),
                                                epos['x_det'],
                                                epos['y_det'])


def do_voltage_and_bowl(epos,p_volt,p_bowl, skip_voltage=False):

    TOF_BIN_SIZE = 1.0
    ROI = np.array([150, 1000])

    
    if skip_voltage:
        TOF_BIN_SIZE = 0.125
        p_bowl = geometric_bowl_correction(epos['tof'],epos['x_det'],epos['y_det'],p_bowl,ROI, TOF_BIN_SIZE)
        tof_corr = mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
        p_volt = np.array([0.0, 0.0])
    else:
            
        if p_volt.size<1:
            p_volt = basic_voltage_correction(epos['tof'],epos['v_dc'],p_volt,ROI, TOF_BIN_SIZE)
            
        tof_vcorr = mod_basic_voltage_correction(p_volt[0],epos['tof'],epos['v_dc'])
        
        
        p_bowl = geometric_bowl_correction(tof_vcorr,epos['x_det'],epos['y_det'],p_bowl,ROI, TOF_BIN_SIZE)
        tof_bcorr = mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
        
        TOF_BIN_SIZES = [0.125]
        
        for tof_bin_size in TOF_BIN_SIZES:
            p_volt = full_voltage_correction(tof_bcorr,epos['v_dc'],p_volt,ROI,tof_bin_size)
            tof_vcorr = mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
            
        #    plot_vs_time()
        #    plot_vs_time_kde()
        #    plot_histos()
        
            p_bowl = geometric_bowl_correction(tof_vcorr,epos['x_det'],epos['y_det'],p_bowl,ROI, tof_bin_size)
            tof_bcorr = mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
    
        #    plot_vs_radius()
        #    plot_vs_time_kde()
        #    plot_histos()
            
        tof_vcorr = mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
        tof_corr = mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])
    
        #plot_vs_time()
        #plot_vs_radius()
        #plot_histos()
        #plot_vs_time_kde()
        
    return (tof_corr, p_volt, p_bowl)


def mod_full_voltage_correction(p_in,tof,v_dc, nom_vdc=5500):
    SCALES = np.array([1e3, 1e-5])
    
    p_in = p_in*SCALES
    
    new_tof = tof*np.sqrt(p_in[0]+v_dc+p_in[1]*v_dc**2)\
                /np.sqrt(p_in[0]+nom_vdc+p_in[1]*nom_vdc**2)
    
#    com0 = np.mean(tof)
#    com = np.mean(new_tof)
#    new_tof = (com0/com)*new_tof
    
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
    

def mod_basic_voltage_correction(p_in,tof,v_dc, nom_vdc=5500):
    SCALES = np.array([1e3])
    
    p_in = p_in*SCALES
        
    new_tof = tof*np.sqrt(p_in[0]+v_dc)/np.sqrt(p_in[0]+nom_vdc)
    
#    com0 = np.mean(tof)
#    com = np.mean(new_tof)
#    new_tof = (com0/com)*new_tof
    
    return new_tof
     
def basic_voltage_correction(tof,v_dc,p_guess,ROI,TOF_BIN_SIZE):
    SCALES = np.array([1e3])
    
    N_BIN = int(np.rint((ROI[1]-ROI[0])/TOF_BIN_SIZE))
    opt_fun = lambda p: -1*TOF_BIN_SIZE*np.sum(np.histogram(mod_basic_voltage_correction(p,tof,v_dc),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]**2)
       
    N_SLICE = 32;
    p_range = ( (-(np.min([np.min(v_dc)-100,500]))/SCALES[0], (np.max(v_dc)*2000/6500)/SCALES[0]),)
    res, _,x,y = brute(opt_fun,
                p_range,
                Ns=N_SLICE,
                disp=True,
                full_output=True,
                finish=None)
#    print('Optimization terminated!!!')
#    print('     Current function value: '+str(res[1]))
#    print('     Current parameter value: '+str(res[0]))
#    
    import matplotlib.pyplot as plt

    print(p_range)
    plt.plot(x,y)
#    opts = {'xatol' : 1e-6,
#            'maxiter' : 512,
#            'disp' : 3}    
#    res = minimize_scalar(opt_fun,
#                          bounds=np.array([-500, np.max(v_dc)])/SCALES,
#                          options=opts,
#                          method='bounded')
##        
        
    p_guess = np.atleast_1d(res)

#    cb = lambda xk: print(xk)
    cb = lambda xk: 0
    
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

    return np.atleast_1d(res.x)



def mod_geometric_bowl_correction(p_in,tof,x_det,y_det):
    SCALES = np.array([1e2, 1e-4, 1e-4, 1e-3])
    
    p_in = p_in*SCALES

    r2 = x_det**2+y_det**2
    new_tof = tof/     \
        (np.sqrt(1+r2/p_in[0]**2)   \
        *(1+p_in[1]*x_det+p_in[2]*y_det+p_in[3]*(r2/30**2)**2))
    
#    new_tof = new_tof/np.sqrt(1+p_in[0]**2)
    
#    com0 = np.mean(tof)
#    com = np.mean(new_tof)
#    new_tof = (com0/com)*new_tof   
            
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
 