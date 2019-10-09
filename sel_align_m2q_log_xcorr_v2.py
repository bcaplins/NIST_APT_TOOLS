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

    shifts = max_idxs-np.mean(max_idxs)
    
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
    
    cts_per_slice=640
    m2q_roi = [0.8,150]
    
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
    ax = plotting_stuff.plot_TOF_vs_time(epos['m2q'],epos,111,clearFigure=True,user_ylim=[0,150])
    ax.plot(pointwise_scales*TEST_PEAK)
    plotting_stuff.plot_TOF_vs_time(m2q_corr,epos,222,clearFigure=True,user_ylim=[0,150])
    
    # Plot histograms uncorrected and corrected
    plotting_stuff.plot_histo(m2q_corr,333,user_xlim=[0, 150],user_bin_width=0.05)
    plotting_stuff.plot_histo(epos['m2q'],333,user_xlim=[0, 150],clearFigure=False,user_bin_width=0.05)
    
#    epos['m2q'] = m2q_corr
#    apt_fileio.write_epos_numpy(epos,'Q:\\NIST_Projects\\EUV_APT_IMS\\BWC\\GaN epos files\\R20_07148-v01_vbmq_corr.epos')
    
    
    _, ys = bin_dat(m2q_corr,isBinAligned=True,bin_width=0.01,user_roi=[0,150])

    print(np.sum(np.square(ys)))
    

    
    return 0



 
def testing():
    
    # Load data
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
    
    epos = apt_fileio.read_epos_numpy(fn)
        
    epos1 = epos[0:-1:2]
    epos2 = epos[1::2]

    cts_per_slice=650
    m2q_roi = [0.8,150]
    
    import time
    t_start = time.time()
    pointwise_scales,piecewise_scales = get_all_scale_coeffs(epos1['m2q'],
                                                             m2q_roi=m2q_roi,
                                                             cts_per_slice=cts_per_slice,
                                                             max_scale=1.15)
    t_end = time.time()
    print('Total Time = ',t_end-t_start)
    
    
    # Compute corrected data
    m2q_corr = epos2['m2q']/pointwise_scales
        
    _, ys = bin_dat(m2q_corr,isBinAligned=True,bin_width=0.01,user_roi=[0,150])

    print(np.sum(np.square(ys)))
    
    return 0   







