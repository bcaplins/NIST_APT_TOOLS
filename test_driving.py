# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import apt_importers as apt
from voltage_and_bowl import do_voltage_and_bowl
import numpy as np
import time

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def align_m2q_to_ref_m2q(ref_m2q,tof):
    ROI = [0.5, 120]
    BIN_SIZE = 0.01
    
    N_BIN = int(np.rint((ROI[1]-ROI[0])/BIN_SIZE))
    

    SCALES = np.array([1e-4, 1])

    com_ref = np.mean(ref_m2q)
    com = np.mean(tof**2)
    
    t0 = -5.0
    c = com_ref/com
    
    
    # Brute force search to find correct minimum
    ref_histo = np.histogram(ref_m2q,range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]
    curr_histo = lambda p: np.histogram(mod_physics_m2q_calibration(np.array([p[0],t0]),tof),range=(ROI[0], ROI[1]),bins=N_BIN,density=True)[0]
    
    opt_fun = lambda p: -1*BIN_SIZE*np.sum(ref_histo*curr_histo(p))
    
    N_SLICE = 128
    fractional_range = 0.1
    
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

def mean_shift_peak_location(dat):
    hwhm = est_hwhm(dat)

    prev_est = np.inf
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


def calibrate_m2q_by_peak_location(m2q, ref_peak_m2qs):
    
    found_peak_m2qs = np.zeros_like(ref_peak_m2qs)
    
    roi_fraction = 0.02
    for idx in range(ref_peak_m2qs.size):
        m2q_roi = ref_peak_m2qs[idx]*np.array([1-roi_fraction,1+roi_fraction])
        x = m2q[(m2q>m2q_roi[0]) & (m2q<m2q_roi[1])]
        found_peak_m2qs[idx] = mean_shift_peak_location(x)
    
    f = interp1d(found_peak_m2qs,ref_peak_m2qs,
                 kind='linear',
                 copy=True,
                 fill_value='extrapolate',
                 assume_sorted=False)
    m2q_corr = f(m2q)

    return m2q_corr
    



from scipy.signal.windows import gaussian

def smooth_with_gaussian_CV(dat):
    N_tries = 32
    
    roi = [np.min(dat),np.max(dat)]
    bin_width = .001
    num_bins = int(np.rint((roi[1]-roi[0])/bin_width))
    
    stds = np.logspace(0,np.log10(75),32)
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
            
        for std_idx,std in enumerate(stds):
            window_len = int(4*std)+1
            
            if window_len>2*xs.size-1:
                raise Exception('Whoa buddy!')
            kern = gaussian(window_len,std)
            kern /= np.sum(kern)
            
            ys1 = histo1[0]
            ys1 = np.r_[ys1[(window_len-1)//2:0:-1],ys1,ys1[-2:(-window_len-3)//2:-1]]

            ys1 = np.convolve(ys1,kern,'valid')
            ys2 = histo2[0]
    
            errs[std_idx] += np.sum(np.square(ys2-ys1))
    
    best_idx = np.argmin(errs)
    
    fig = plt.figure(num=100)
    fig.clear()
    ax = plt.axes()
    ax.plot(stds,errs)
    ax.set_xscale('log')    
    ax.set_yscale('log')    
    
    plt.pause(0.001)
    
    window_len = int(4*stds[best_idx])+1
    
    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=False)
    kern = gaussian(window_len,stds[best_idx])
    kern /= np.sum(kern)
    
    ys1 = histo[0]    

    ys1 = np.r_[ys1[(window_len-1)//2:0:-1],ys1,ys1[-2:(-window_len-3)//2:-1]]

    ys1 = np.convolve(kern,ys1,mode='valid')
    ys2 = histo[0]

    fig = plt.figure(num=101)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys2,'.')
    ax.plot(xs,ys1)
    
    
    ax.set_yscale('log')    
    
    plt.pause(0.001)
    
    return (xs, ys1, stds[best_idx])




def est_baseline_empirical(xs,ys):
    return np.percentile(ys,20)


def get_range_empirical(xs,ys_smoothed):
    N = xs.size
    N4 = N//4
    
    max_idx = np.argmax(ys_smoothed[N4:2*N4])+N4
    max_val = ys_smoothed[max_idx]
    
    ax = plt.gca()
    ax.plot(xs[max_idx],ys_smoothed[max_idx],'o')
    
    bl_est = est_baseline_empirical(xs,ys_smoothed)
    
    search_val = 0.1*(max_val-bl_est)+bl_est
    
    lims = np.zeros(2)
    # Search right
    for idx in np.arange(max_idx+20,N-1):
        if (ys_smoothed[idx]>search_val) & (ys_smoothed[idx+1]<search_val):
            lims[1] = np.interp(search_val,ys_smoothed[idx:idx+2],xs[idx:idx+2])
            break
        if ys_smoothed[idx] < ys_smoothed[idx+1]:
            lims[1] = xs[idx]
            break
        
    # Search left
    for idx in np.arange(max_idx-1-20,-1,-1):
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
    



# Read in template spectrum
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
ref_epos = apt.read_epos_numpy(fn)


# Read in data
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
epos = apt.read_epos_numpy(fn)
wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")


# Find c and t0 for ToF data based on reference spectrum
m2q_corr, p_m2q = align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)

# Perform linear calibration over known peaks
ref_peak_m2qs = np.array([1.007825, 14.003074/2, 14.003074, 2*14.003074, 68.925581/2, 70.924701/2, 68.925581, 70.924701])
m2q_corr = calibrate_m2q_by_peak_location(m2q_corr,ref_peak_m2qs)



quant_peak_m2qs = np.array([14.003074/2, 
                            14.003074, 
                            14.003074+1, 
                            23,
                            23.66,
                            2*14.003074,
                            29,
                            68.925581/2, 
                            70.924701/2, 
                            41.5,
                            42,
                            42.5,
                            68.925581,
                            70.924701])

 
N_param = 1
res_param = np.zeros((quant_peak_m2qs.size,N_param))
cts = np.zeros((quant_peak_m2qs.size,2))


#for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
for idx,m2q_loc in enumerate(quant_peak_m2qs):
    
    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]   
    pk_dat = m2q_corr[(m2q_corr>=m2q_roi[0]) & (m2q_corr<=m2q_roi[1])]
    
    xs, ys, res_param[idx] = smooth_with_gaussian_CV(pk_dat)
    
    est_bl = est_baseline_empirical(xs,ys)
    
    ax = plt.gca()
    ax.plot(np.array([np.min(xs),np.max(xs)]),np.array([1,1])*est_bl,'k--')
    
    rng = get_range_empirical(xs,ys)
    
    ax.plot(np.array([1,1])*rng[0],np.array([np.min(ys),np.max(ys)]),'k--')
    ax.plot(np.array([1,1])*rng[1],np.array([np.min(ys),np.max(ys)]),'k--')
    
    ax.set_yscale('linear')    

    cts[idx,:] = get_peak_count(m2q_corr,rng,est_bl)
    
    plt.pause(1)

stoich = np.array([ [1, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [2, 0],
                    [2, 0],
                    [0, 1],
                    [0, 1],
                    [1, 1],
                    [3, 0],
                    [1, 1],
                    [0, 1],
                    [0, 1]])
    
res = np.sum(stoich*cts[:,0][:,None],axis=0)

res/np.sum(res)


    

raise SystemExit(0)























































# For each peak
# divide data in half
# Use one half to make a histo with a series of bin widths
# Use the other half to test which histo best predicts the other
    
 
N_param = 1
res_param = np.zeros((quant_peak_m2qs.size,N_param))


#for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
for idx,m2q_loc in enumerate(quant_peak_m2qs):
    
    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
    
    pk_dat = m2q_corr[(m2q_corr>=m2q_roi[0]) & (m2q_corr<=m2q_roi[1])]
    
    choices = np.random.rand(pk_dat.size)>0.5
    
    pk_dat1 = pk_dat[np.nonzero(choices)]
    pk_dat2 = pk_dat[np.nonzero(~choices)]
    
    m2q_bin_width = .001
    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
    
    histo1 = np.histogram(pk_dat1,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    histo2 = np.histogram(pk_dat2,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    
    xs = (histo1[1][1:]+histo1[1][0:-1])/2
    
    stds = np.logspace(0,np.log10(50),32)
    errs = np.zeros_like(stds)
    
    for std_idx,std in enumerate(stds):

        if int(3*std)>xs.size:
            raise Exception('Whoa buddy!')
        kern = gaussian(int(3*std),std)
        kern /= np.sum(kern)
        
        ys1 = histo1[0]
        ys1 = np.convolve(ys1,kern,'same')
        ys2 = histo2[0]

        errs[std_idx] = np.sum(np.square(ys2-ys1))

    best_idx = np.argmin(errs)
    
    fig = plt.figure(num=127)
    fig.clear()
    ax = plt.axes()
    ax.plot(stds,errs)
    ax.set_xscale('log')    
    ax.set_yscale('log')    
    
    plt.pause(0.002)
    
    histo = np.histogram(pk_dat,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    kern = gaussian(int(3*stds[best_idx]),stds[best_idx])
    kern /= np.sum(kern)
    
    ys1 = histo[0]
    ys1 = np.convolve(ys1,kern,'same')
    ys2 = histo[0]

    fig = plt.figure(num=128)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys2)
    ax.plot(xs,ys1)
    
    
    ax.set_yscale('log')    
    
    res_param[idx] = stds[best_idx]
    
    plt.pause(1)








































def G(x,F,E):
    return np.exp(-4*np.log(2)*np.square(x-E)/np.square(F))

def GL(x,F,E,m):
    num = np.exp(-4*np.log(2)*(1-m)*np.square(x-E)/np.square(F))
    denom = 1+4*m*np.square(x-E)/np.square(F)
    return num/denom

def AW(x,a,F,E):
    exp_num = 2*np.sqrt(np.log(2))*(x-E)
    exp_denom = F-a*2*np.sqrt(np.log(2))*(x-E)
    exp2 = np.square(exp_num/exp_denom)
    return np.exp(-exp2)

def UG(x,a,b,F,E,m):
    # assume x is sorted ascending
    lhs_idxs = np.nonzero(x>E)
    rhs_idxs = np.nonzero(x<=E)
    
    w = b*0.7+0.3/(a+0.01)
    
    res = np.zeros_like(x)
    
    res[lhs_idxs] = GL(x[lhs_idxs],F,E,m)+w*(AW(x[lhs_idxs],a,F,E)-G(x[lhs_idxs],F,E))
    res[rhs_idxs] = GL(x[rhs_idxs],F,E,m)
    
    return res


def b_G(x,sigma,x0):
    return np.exp(-np.square(x-x0)/(2*np.square(sigma)))

def b_L(x,omega,x0):
    return omega**2/(omega**2+(x-x0)**2)

def b_Ln(x,omega,x0,n):
    return np.power(b_L(x,omega,x0),n)

def b_bg(x,m,b):
    return m*x+b

def b_model(x,x0,omega,sigma,n,amp,l_frac,m,b):
    lhs_idxs = np.nonzero(x<=x0)
    rhs_idxs = np.nonzero(x>x0)
    
    y = np.zeros_like(x)
    
    y[lhs_idxs] = b_G(x[lhs_idxs],sigma,x0)
    y[rhs_idxs] = (1-l_frac)*b_G(x[rhs_idxs],sigma,x0)+l_frac*b_Ln(x[rhs_idxs],omega,x0,n)
    
    y *= amp
    y += b_bg(x,m,b)
    
    return y

def b_e(x,tau,x0):
    return np.exp(-(x-x0)/tau)*(x>=x0)

def b_model2(x,amp_g,amp_e,x0,sigma,tau,m,b):
    
    y = np.zeros_like(x)
    
    t1 = amp_g*b_G(x,sigma,x0)
    kern = t1/np.sum(t1)
    
    t2 = b_e(x,tau,x0)
    t2 = amp_e*np.convolve(t2,kern,'same')
    
    
    y = t1+t2
    
    y += b_bg(x,m,b)
    
    return y






quant_peak_m2qs = np.array([14.003074/2, 
                            14.003074, 
                            14.003074+1, 
                            23,
                            23.66,
                            2*14.003074,
                            29,
                            68.925581/2, 
                            70.924701/2, 
                            41.5,
                            42,
                            42.5,
                            68.925581,
                            70.924701])

N_param = 7
res_param = np.zeros((quant_peak_m2qs.size,N_param))


#for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
for idx,m2q_loc in enumerate(quant_peak_m2qs):
    
    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
    m2q_bin_width = .001
    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
    
    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    
    xs = (histo[1][1:]+histo[1][0:-1])/2
    ys = histo[0]
    
#    if idx==quant_peak_m2qs.size-1:
    if idx==0:
        p0 = [1, 1, 1, 0.025, 0.1, 0, 1]
    else:
        p0 = popt.x
#        p0[1:2] = popt.x[1:2]*m2q_loc/popt.x[0]
        
        
        
    p0[2] = m2q_loc
    p0[0] = (np.max(ys)-np.mean(ys[0:10]))/2
    p0[1] = (np.max(ys)-np.mean(ys[0:10]))/2
    
    p0[5] = np.min([(np.mean(ys[-10:])-np.mean(ys[0:10]))/(np.mean(xs[-10:])-np.mean(xs[0:-10])),0])
    p0[6] = np.mean(ys[0:10])-p0[5]*np.mean(xs[0:10])
    
#    p0[4] = np.mean(ys[0:20])*np.sqrt(np.mean(xs[0:20]))
    
    resid_func = lambda p: b_model2(xs,*p)-ys
# b_model2(x,amp_g,amp_e,x0,sigma,tau,m,b):
    lbs = [0,       0,      p0[2]-0.1,   0.001,   0.002,    -np.inf,    0]
    ubs = [np.inf,  np.inf, p0[2]+0.1,   0.2,     1,        0,          np.inf]
    
    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
#    popt = least_squares(resid_func, p0, ftol=1e-12, max_nfev=2048)
    
    fig = plt.figure(num=128)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys,'.')

    mod0_ys = b_model2(xs,*p0)
    ax.plot(xs,mod0_ys)
    
    
    plt.pause(1)
    
    
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys,'.')
#    p0 = [0.35,0.5,0.05,28,0.5,800,10]
    
    mod_ys = b_model2(xs,*popt.x)
    ax.plot(xs,mod_ys)
    
#    ax.plot(xs,mod_ys-ys,'r--')
    
    ax.set_yscale('log')    
    
    res_param[idx,:] = popt.x
    
    plt.pause(2)
    
#    plt.ginput(1)
    














































quant_peak_m2qs = np.array([14.003074/2, 
                            14.003074, 
                            14.003074+1, 
                            23,
                            23.66,
                            2*14.003074,
                            29,
                            68.925581/2, 
                            70.924701/2, 
                            41.5,
                            42,
                            42.5,
                            68.925581,
                            70.924701])

N_param = 8
res_param = np.zeros((quant_peak_m2qs.size,N_param))


#for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
for idx,m2q_loc in enumerate(quant_peak_m2qs):
    
    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
    m2q_bin_width = .001
    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
    
    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    
    xs = (histo[1][1:]+histo[1][0:-1])/2
    ys = histo[0]
    
#    if idx==quant_peak_m2qs.size-1:
    if idx==0:
        p0 = [0, 0.025, 0.025, 1, 1, 0.5, -0.1, 1]
    else:
        p0 = popt.x
#        p0[1:2] = popt.x[1:2]*m2q_loc/popt.x[0]
        
        
    p0[0] = m2q_loc
    p0[4] = np.max(ys)-np.mean(ys[0:10])
    
    p0[6] = np.min([(np.mean(ys[-10:])-np.mean(ys[0:10]))/(np.mean(xs[-10:])-np.mean(xs[0:-10])),0])
    p0[7] = np.mean(ys[0:10])-p0[6]*np.mean(xs[0:10])
    
#    p0[4] = np.mean(ys[0:20])*np.sqrt(np.mean(xs[0:20]))
    
    resid_func = lambda p: b_model(xs,*p)-ys
# b_model(x,x0,omega,sigma,n,amp,l_frac,m,b):
    lbs = [p0[0]-0.1,   0.001,    0.001,   0.5,  0,      0, -np.inf,    0]
    ubs = [p0[0]+0.1,   0.2,      0.2,     2,    np.inf, 1, 0,          np.inf]
    
    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=2, ftol=1e-12, max_nfev=2048)
    
    fig = plt.figure(num=128)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys)
#    p0 = [0.35,0.5,0.05,28,0.5,800,10]
    mod0_ys = b_model(xs,*p0)
    ax.plot(xs,mod0_ys)
    
    
    plt.pause(1)
    
    
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys)
#    p0 = [0.35,0.5,0.05,28,0.5,800,10]
    
    mod_ys = b_model(xs,*popt.x)
    ax.plot(xs,mod_ys)
    
#    ax.plot(xs,mod_ys-ys,'r--')
    
    ax.set_yscale('log')    
    
    res_param[idx,:] = popt.x
    
    plt.pause(1)
    




























def fit_func(x,a,b,F,E,m,amp,fudge):
    return amp*UG(x,a,b,F,E,m)+fudge*np.reciprocal(np.sqrt(x))


quant_peak_m2qs = np.array([14.003074/2, 
                            14.003074, 
                            14.003074+1, 
                            23,
                            23.66,
                            2*14.003074,
                            29,
                            68.925581/2, 
                            70.924701/2, 
                            41.5,
                            42,
                            42.5,
                            68.925581,
                            70.924701])

N_param = 7
res_param = np.zeros((quant_peak_m2qs.size,N_param))





#for idx,m2q_loc in reversed(list(enumerate(ref_peak_m2qs))):
for idx,m2q_loc in enumerate(quant_peak_m2qs):
    
    m2q_roi = np.array([0.985,1.015])*quant_peak_m2qs[idx]
    m2q_bin_width = .005
    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
    
    histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    
    xs = (histo[1][1:]+histo[1][0:-1])/2
    ys = histo[0]
    
#    if idx==quant_peak_m2qs.size-1:
    if idx==0:
        p0 = [0.35,0.5,0.005,-1,0.5,-1,70]
    else:
        p0 = popt.x
        p0[2] = popt.x[2]*m2q_loc/popt.x[3]
        
        
    p0[3] = m2q_loc
    p0[5] = np.max(ys)        
    
    resid_func = lambda p: fit_func(xs,*p)-ys
         # a, b, F ,E, m, amp, off
    lbs = [0,-1,0, p0[3]-0.1 ,0,0,1]
    ubs = [1,1,0.2,p0[3]+0.1 ,1,np.inf,1e6]
    
    popt = least_squares(resid_func, p0, bounds=(lbs,ubs), verbose=0, ftol=1e-12, max_nfev=2048)
    
    fig = plt.figure(num=128)
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys)
#    p0 = [0.35,0.5,0.05,28,0.5,800,10]
    mod0_ys = fit_func(xs,*p0)
    ax.plot(xs,mod0_ys)
    
    
    plt.pause(.01)
    
    
    fig.clear()
    ax = plt.axes()
    
    ax.plot(xs,ys,'.')
#    p0 = [0.35,0.5,0.05,28,0.5,800,10]
    
    mod_ys = fit_func(xs,*popt.x)
    ax.plot(xs,mod_ys)
    
#    ax.plot(xs,mod_ys-ys,'r--')
    
    ax.set_yscale('log')    
    
    res_param[idx,:] = popt.x
    
    plt.pause(1)






    
    
    
   















m2q_roi = [0, 100]
m2q_bin_width = .001
m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))

ref_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=True)

std = 30
kern = gaussian(int(3*std),std)
kern /= np.sum(kern)

ys = ref_histo[0]
ys = np.convolve(ys,kern,'same')



centers = (ref_histo[1][1:]+ref_histo[1][0:-1])/2



fig = plt.figure(num=128)
fig.clear()
ax = plt.axes()

ax.plot(centers,ref_histo[0],label='ref')
ax.plot(centers,ys,label='new')
#
#ax.hist(epos['m2q'],bins=ref_histo[1],log=True,histtype='step',label='ref')
#ax.hist(m2q_corr,bins=ref_histo[1],log=True,histtype='step',label='new')

ax.legend()
ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 100],
       title='My First Plot')
ax.set_yscale('log')    






















found_peak_m2q = np.zeros_like(ref_peak_m2qs)

for idx in range(ref_peak_m2qs.size):
    m2q_roi = ref_peak_m2qs[idx]*np.array([0.98,1.02])
    x = m2q_corr[(m2q_corr>m2q_roi[0]) & (m2q_corr<m2q_roi[1])]        
#    x = epos['m2q'][(epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1])]        
    found_peak_m2q[idx] = mean_shift_peak_location(x)

fig = plt.figure(num=256)
fig.clear()
ax = plt.axes()
ax.plot(ref_peak_m2qs,found_peak_m2q/ref_peak_m2qs,'-o')
#ax.plot(exp_peak_m2q,exp_peak_m2q/exp_peak_m2q,'--k')


f = interp1d(found_peak_m2q,exp_peak_m2q,
             kind='linear',
             copy=True,
             fill_value='extrapolate',
             assume_sorted=False)
m2q_corr2 = f(m2q_corr)


exp_peak_m2q = np.array([1.007825, 14.003074/2, 14.003074, 2*14.003074, 68.925581/2, 70.924701/2, 68.925581, 70.924701])
found_peak_m2q = np.zeros_like(exp_peak_m2q)

for idx in range(exp_peak_m2q.size):
    m2q_roi = exp_peak_m2q[idx]*np.array([0.98,1.02])
    x = m2q_corr2[(m2q_corr2>m2q_roi[0]) & (m2q_corr2<m2q_roi[1])]        
#    x = epos['m2q'][(epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1])]        
    print(est_hwhm(x))
    found_peak_m2q[idx] = mean_shift_peak_location(x)

fig = plt.figure(num=256)
fig.clear()
ax = plt.axes()
ax.plot(exp_peak_m2q,(found_peak_m2q-exp_peak_m2q)/exp_peak_m2q,'-o')
#ax.plot(exp_peak_m2q,exp_peak_m2q,'--k')



for i in range(exp_peak_m2q.size):
    
    m2q_roi = exp_peak_m2q[i]*np.array([0.98,1.02])
    x = m2q_corr[(m2q_corr>m2q_roi[0]) & (m2q_corr<m2q_roi[1])]
        
    hwhm_est = est_hwhm(x)
    pk_loc_est = mean_shift_peak_location(x)

    
    m2q_bin_width = hwhm_est/5
    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
    
    new_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=False)
    
    
    fig = plt.figure(num=128)
    fig.clear()
    ax = plt.axes()
    
    ax.hist(x,bins=new_histo[1],log=False,histtype='step',label='ref')
    
    
    ax.plot([pk_loc_est,pk_loc_est],[0,np.max(new_histo[0])],'k--')
    ax.plot([pk_loc_est,pk_loc_est]-hwhm_est,[0,np.max(new_histo[0])],'k--')
    ax.plot([pk_loc_est,pk_loc_est]+hwhm_est,[0,np.max(new_histo[0])],'k--')
#    ax.plot([pcts[1],pcts[1]],[0,500],'k--')
#    ax.plot([pcts[2],pcts[2]],[0,500],'k--')
    
#    ax.plot(np.diff(new_histo[0])/np.max(new_histo[0]))
#    ax.set(xlabel='m2q', ylabel='counts', ylim=[-1, 1])
    ax.grid()
    plt.pause(1.5)
    


#pcts = np.percentile(x,[25,50,75])
#
#ax.plot([pcts[0],pcts[0]],[0,500],'k--')
#ax.plot([pcts[1],pcts[1]],[0,500],'k--')
#ax.plot([pcts[2],pcts[2]],[0,500],'k--')
#

# Bin until I get 1000 cts in peak

# Find mode

# Find an estimate of the FWHM

# Use mean shift to find peak













kde = gaussian_kde(x,bw_method=0.001)


#grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                    {'bandwidth': np.logspace(-5, 0, 8)},
#                    cv=2) # 20-fold cross-validation
#grid.fit(x[:, None])
#print(grid.best_params_)



#kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
#kde.fit(x[:, None])

# score_samples returns the log of the probability density

m2q_roi = [0.95, 1.75]
m2q_bin_width = 0.001
m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))

x_d = np.linspace(m2q_roi[0], m2q_roi[1], m2q_num_bins)


fig = plt.figure(num=128)
fig.clear()
ax = plt.axes()

ax.fill_between(x_d, 10*kde(x_d), alpha=0.5)
#ax.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
#ax.ylim(-0.02, 0.22)




for idx, pk_loc in enumerate(exp_peak_m2q):
    found_peak_m2q[idx] = np.mean(m2q_corr[(m2q_corr<pk_loc*1.0015) & (m2q_corr>pk_loc*0.9985)])
    


m2q_corr_tmp = m2q_corr[(m2q_corr>0.95) & (m2q_corr<1.75)]


nbins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20])
pk_est = np.zeros(nbins.shape)
for idx, nbin in enumerate(nbins):
    tmp_histo = np.histogram(m2q_corr_tmp,range=(0, 2),bins=2*nbin,density=False)
    max_idx = np.argmax(tmp_histo[0])
    pk_est[idx] = tmp_histo[1][max_idx]
    
    


fig = plt.figure(num=64)
fig.clear()
ax = plt.axes()

ax.plot(pk_est,'-o')

#ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 200],
#       title='My First Plot')

    
    




m2q_roi = [0, 100]
m2q_bin_width = .03
m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))

ref_histo = np.histogram(epos['m2q'],range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=True)
new_histo = np.histogram(m2q_corr,range=(m2q_roi[0], m2q_roi[1]),bins=m2q_num_bins,density=True)

centers = (ref_histo[1][1:]+ref_histo[1][0:-1])/2

-1*m2q_bin_width*np.sum(ref_histo[0]*new_histo[0])


fig = plt.figure(num=128)
fig.clear()
ax = plt.axes()

#ax.plot(centers,ref_histo[0],label='ref')
#ax.plot(centers,new_histo[0],label='new')

#ax.hist(epos['m2q'],bins=ref_histo[1],log=True,histtype='step',label='ref')
ax.hist(m2q_corr,bins=ref_histo[1],log=True,histtype='step',label='new')

ax.legend()
ax.set(xlabel='m2q', ylabel='counts', xlim=[0, 100],
       title='My First Plot')










    











