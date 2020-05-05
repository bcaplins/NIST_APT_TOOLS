# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:53:25 2019

@author: bwc
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff

from histogram_functions import bin_dat
import scipy.interpolate



def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def get_shifts(ref,N,max_shift=150):
    
    rfft_ref = np.fft.rfft(ref,axis=1)
    rfft_N = np.fft.rfft(N,axis=1)
    xc = np.fft.irfft(rfft_N*np.conj(rfft_ref),axis=1)    

    xc[:,max_shift:xc.shape[1]-max_shift] = 0

    max_idxs = np.argmax(xc, axis=1)
    max_idxs[max_idxs>xc.shape[1]//2] = max_idxs[max_idxs>xc.shape[1]//2]-xc.shape[1]

#    shifts = max_idxs-np.mean(max_idxs)
    shifts = max_idxs-np.median(max_idxs)
    
    return shifts

def create_histogram(lys,cts_per_slice=2**10,y_roi=None,delta_ly=1e-4):
    if y_roi is None:
        y_roi = [0.5, 200]
    ly_roi = np.log(y_roi)

#    num_ly = int(np.ceil(np.abs(np.diff(ly_roi))/delta_ly)) # integer number
    num_ly = int(np.ceil(np.abs(np.diff(ly_roi))/delta_ly/2)*2) # even number
#    num_ly = int(2**np.round(np.log2(np.abs(np.diff(ly_roi))/delta_ly)))-1 # closest power of 2
    print('number of points in ly = ',num_ly)
    num_x = int(lys.size/cts_per_slice)
    
    xs = np.arange(lys.size)
    print([num_x,num_ly])
        
    N,x_edges,ly_edges = np.histogram2d(xs,lys,bins=[num_x,num_ly],range=[[1,lys.size],ly_roi],density=False)
    return (N,x_edges,ly_edges)

def edges_to_centers(*edges):
    centers = []
    for es in edges:
        centers.append((es[0:-1]+es[1:])/2)
    return centers

def get_all_scale_coeffs(m2q,max_scale = 1.1,m2q_roi=None,cts_per_slice=2**10):
    if m2q_roi is None:        
        m2q_roi = [0.5,200]
    
    # Take the log of m2q data
    lys = np.log(m2q)
    
    # Create the histgram.  Compute centers and delta y
    N,x_edges,ly_edges = create_histogram(lys,y_roi=m2q_roi,cts_per_slice=cts_per_slice)
    x_centers,ly_centers = edges_to_centers(x_edges,ly_edges)
    delta_ly = ly_edges[1]-ly_edges[0]
    
    # The total pointwise log(y) shift
    pointwise_ly_shifts = np.zeros(m2q.size)
    
    # Do one iteration with the center of the data as a reference (log) spectrum
    # Note: ensure it is 2d to keep image_registration happy
    ref = np.mean(N[N.shape[0]//4:3*N.shape[0]//4,:],axis=0)[None,:]
    
    # Get the piecewise shifts
    max_pixel_shift = int(np.ceil(np.log(max_scale)/delta_ly))
    shifts0 = get_shifts(ref,N,max_shift=max_pixel_shift)*delta_ly
    
    # Interpolate to pointwise shifts
    f = scipy.interpolate.interp1d(x_centers,shifts0,fill_value='extrapolate')
    pointwise_ly_shifts += f(np.arange(m2q.size))
    
    # Correct the log m2q data
    lys_corr = lys - pointwise_ly_shifts

    # Recompute the histogram 
    N,x_edges,ly_edges = create_histogram(lys_corr,y_roi=m2q_roi,cts_per_slice=cts_per_slice)
    x_centers,ly_centers = edges_to_centers(x_edges,ly_edges)
    delta_ly = ly_edges[1]-ly_edges[0]

    # Use the center half of the data as the new reference (log) spectrum
    ref = np.mean(N[N.shape[0]//4:3*N.shape[0]//4,:],axis=0)[None,:]

    # Get the piecewise shifts
    max_pixel_shift = int(np.ceil(np.log(max_scale)/delta_ly))
    shifts1 = get_shifts(ref,N,max_shift=max_pixel_shift)*delta_ly
    
    # Interpolate to get pointwise shifts
    f = scipy.interpolate.interp1d(x_centers,shifts1,fill_value='extrapolate')
    pointwise_ly_shifts += f(np.arange(m2q.size))
    
    # Print total pointwise and piecewise shifts for output
    pointwise_scales = np.exp(pointwise_ly_shifts)
    piecewise_scales = np.exp(shifts0+shifts1)

    return (pointwise_scales,piecewise_scales)



def __main__():
    
    
    plt.close('all')
    #
    # Load data
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
    fn = r"\\cfs2w.campus.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"

    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbm_corr.epos"
    #fn = r"D:\Users\clifford\Documents\Python Scripts\NIST_DATA\R20_07094-v03.epos"
    #fn = r"D:\Users\clifford\Documents\Python Scripts\NIST_DATA\R45_04472-v03.epos"
    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos"
    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01.epos"
    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07086-v01.epos"
    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07276-v03.epos"
#    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
    #fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02.epos"
#    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg doped
#    fn = fn[:-5]+'_vbm_corr.epos'
    
    
    epos = apt_fileio.read_epos_numpy(fn)
        
#    plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,1,clearFigure=True,user_ylim=[0,150])
#    epos = epos[0:2**20]
    
    cts_per_slice=2**10
    m2q_roi = [0.8,75]
    
    import time
    t_start = time.time()
    pointwise_scales,piecewise_scales = get_all_scale_coeffs(epos['m2q'],
                                                             m2q_roi=m2q_roi,
                                                             cts_per_slice=cts_per_slice,
                                                             max_scale=1.15)
    t_end = time.time()
    print('Total Time = ',t_end-t_start)
    
    
    
    
    
    # Plot histogram in log space
    lys_corr = np.log(epos['m2q'])-np.log(pointwise_scales)
    N,x_edges,ly_edges = create_histogram(lys_corr,y_roi=m2q_roi,cts_per_slice=cts_per_slice)
    
#    fig = plt.figure(figsize=(8,8))
#    plt.imshow(np.log1p(np.transpose(N)), aspect='auto', interpolation='none',
#               extent=extents(x_edges) + extents(ly_edges), origin='lower')
#    
    # Compute corrected data
    m2q_corr = epos['m2q']/pointwise_scales
    
    # Plot data uncorrected and corrected
    TEST_PEAK = 32
    ax = plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,111,clearFigure=True,user_ylim=[0,75])
    ax.plot(pointwise_scales*TEST_PEAK)
    plotting_stuff.plot_TOF_vs_time(m2q_corr,epos,222,clearFigure=True,user_ylim=[0,75])
    
    # Plot histograms uncorrected and corrected
    plotting_stuff.plot_histo(m2q_corr,333,user_xlim=[0, 75],user_bin_width=0.01)
    plotting_stuff.plot_histo(epos['m2q'],333,user_xlim=[0, 75],clearFigure=False,user_bin_width=0.01)
    
#    epos['m2q'] = m2q_corr
#    apt_fileio.write_epos_numpy(epos,'Q:\\NIST_Projects\\EUV_APT_IMS\\BWC\\GaN epos files\\R20_07148-v01_vbmq_corr.epos')
    
    
    _, ys = bin_dat(m2q_corr,isBinAligned=True,bin_width=0.01,user_roi=[0,75])

    print(np.sum(np.square(ys)))
    

    
    return 0



def test_1d_vec(X):
    vals = X.copy()
    vals.sort()
    cs = np.r_[0,np.cumsum(vals)[0:-1]]
    idxs = np.arange(vals.size)
    tot_dist_1d = np.sum(vals*idxs-cs)
    return tot_dist_1d

def chi2(dat):
    n = dat.size
    f = np.sum(dat)    
    f_n = f/n    
    chi2 = np.sum(np.square(dat-f_n)/f_n)
    return chi2

 
def testing():
    
    # Load data
#    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
#    fn = r"\\cfs2w.campus.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"
    fn = r"C:\Users\bwc\Documents\NetBeansProjects\R44_03115\recons\recon-v02\default\R44_03115-v02.epos"
    
    epos = apt_fileio.read_epos_numpy(fn)
#    epos = epos[0:1000000]    
    
    epos1 = epos[0:-1:2]
    epos2 = epos[1::2]


    
    
    N_lower = 8
    N_upper = 16
    N = N_upper-N_lower+1
    slicings = np.logspace(N_lower,N_upper,N,base=2)
    
    
    opt_res = np.zeros(N)
    time_res = np.zeros(N)

    for idx,cts_per_slice in enumerate(slicings):
        m2q_roi = [0.9,180]
        
        import time
        t_start = time.time()
        pointwise_scales,piecewise_scales = get_all_scale_coeffs(epos1['m2q'],
                                                                 m2q_roi=m2q_roi,
                                                                 cts_per_slice=cts_per_slice,
                                                                 max_scale=1.15)
        t_end = time.time()
        time_res[idx] = t_end-t_start
        print('Total Time = ',time_res[idx])
        
        
        # Compute corrected data
        m2q_corr = epos2['m2q']/pointwise_scales
            
        _, ys = bin_dat(m2q_corr,isBinAligned=True,bin_width=0.01,user_roi=[0,250])
    
        opt_res[idx] = chi2(ys)
#        ys = ys/np.sum(ys)
#        opt_res[idx] = np.sum(np.square(ys))
#        opt_res[idx] = test_1d_vec(m2q_corr[(m2q_corr>0.8) & (m2q_corr<80)])
        print(opt_res[idx])
        
    print(slicings)
    print(opt_res/np.max(opt_res))
    print(time_res)
    
    
    
    fig = plt.figure(num=666)
    fig.clear()
    ax = fig.gca()

    ax.plot(tmpx,tmpy,'s-', 
            markersize=8,label='SiO2')

    ax.plot(slicings,opt_res/np.max(opt_res),'o-', 
            markersize=8,label='ceria')
    ax.set(xlabel='N (events per chunk)', ylabel='compactness metric (normalized)')
    ax.set_xscale('log')    

    
    ax.legend()
       
    ax.set_xlim(5,1e5)
    ax.set_ylim(0.15, 1.05)


    fig.tight_layout()
    
    
    return 0   




#ceria_chi2 = [50100017.77823232, 54953866.6417411 , 56968470.41426052,
#                57832991.31751654, 58136713.37802257, 58103886.08055325,
#                57387594.45685758, 56278878.21237884, 52715317.92279702,
#                48064845.44202947, 42888989.38802697, 34852375.17765743,
#                30543492.44201695]
#ceria_slic = [1.6000e+01, 3.2000e+01, 6.4000e+01, 1.2800e+02, 2.5600e+02,
#              5.1200e+02, 1.0240e+03, 2.0480e+03, 4.0960e+03, 8.1920e+03,
#              1.6384e+04, 3.2768e+04, 6.5536e+04]
#
#sio2_slic = [1.6000e+01, 3.2000e+01, 6.4000e+01, 1.2800e+02, 2.5600e+02,
#           5.1200e+02, 1.0240e+03, 2.0480e+03, 4.0960e+03, 8.1920e+03,
#           1.6384e+04, 3.2768e+04, 6.5536e+04]
#
#sio2_chi2 = [1.14778821e+08, 1.47490976e+08, 1.52686129e+08, 1.51663402e+08,
#           1.45270347e+08, 1.34437550e+08, 1.18551040e+08, 1.01481358e+08,
#           8.62360167e+07, 7.45989701e+07, 6.50088595e+07, 4.22995630e+07,
#           3.71045091e+07]
#
#
#fig = plt.figure(num=666)
#fig.clear()
#ax = fig.gca()
#
#ax.plot(sio2_slic,sio2_chi2/np.max(sio2_chi2),'s-', 
#        markersize=8,label='SiO2')
#
#ax.plot(ceria_slic,ceria_chi2/np.max(ceria_chi2),'o-', 
#        markersize=8,label='ceria')
#ax.set(xlabel='N (events per chunk)', ylabel='compactness metric (normalized)')
#ax.set_xscale('log')    
#
#
#ax.legend()
#   
#ax.set_xlim(5,1e5)
#ax.set_ylim(0.15, 1.05)
#
#
#fig.tight_layout()
#
#
#
#
#  
## Load data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
##fn = r"\\cfs2w.campus.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"
#epos = apt_fileio.read_epos_numpy(fn)
##epos = epos[0:400000]    
#m2q_roi = [0.9,200]
#
#import time
#t_start = time.time()
#pointwise_scales,piecewise_scales = get_all_scale_coeffs(epos['m2q'],
#                                                         m2q_roi=m2q_roi,
#                                                         cts_per_slice=512,
#                                                         max_scale=1.15)
#t_end = time.time()
#print('Total Time = ',t_end-t_start)
#
#
#
#
#
#
#
#
#













