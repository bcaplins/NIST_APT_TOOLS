# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:41:03 2019

@author: bwc
"""

import numpy as np

def bin_dat(dat,bin_width=0.001,user_roi=[],isBinAligned=False,isDensity=False):
    user_roi = np.asarray(user_roi)
    roi_supp = (user_roi.size == 2)
    
    # Get roi
    if isBinAligned and roi_supp:
        lower = np.floor(np.min(user_roi)/bin_width)*bin_width
        upper = np.ceil(np.max(user_roi)/bin_width)*bin_width
        roi = np.array([lower, upper])
    elif isBinAligned and (not roi_supp):
        lower = np.floor(np.min(dat)/bin_width)*bin_width
        upper = np.ceil(np.max(dat)/bin_width)*bin_width
        roi = np.array([lower, upper])        
    elif (not isBinAligned) and roi_supp:
        roi = user_roi
    else: # (not isBinAligned) and (not roi_supp):
        roi = np.array([np.min(dat), np.max(dat)])
        
    num_bins = int(np.rint((roi[1]/bin_width-roi[0]/bin_width)))
    histo = np.histogram(dat,range=(roi[0], roi[1]),bins=num_bins,density=isDensity)
    
    xs = (histo[1][1:]+histo[1][0:-1])/2
    ys = histo[0]
    
    return (xs,ys)


def edges_to_centers(*edges):
    """
    Convert bin edges to bin centers
    Parameters
    ----------
    *edges : bin edges
    Returns
    -------
    centers : list of bin centers
    """
    centers = []
    for es in edges:
        centers.append((es[0:-1]+es[1:])/2)
    return centers

def corrhist(epos):
    
    

    
    
    
        
    dat = epos['tof']
    roi = [0, 5000]
    delta = 1
 
#    dat = epos['m2q']
#    roi = [0, 100]
#    delta = .1
#    
#    MF = np.mean(epos['tof']/np.sqrt(epos['m2q']))
#    dat = np.sqrt(epos['m2q'])*MF
#    roi = [0, np.sqrt(250)*MF]
#    delta = .001*MF
##    
    
    N = int(np.ceil((roi[1]-roi[0])/delta))
    
    
    
    corrhist = np.zeros([N,N], dtype=int)
    
    multi_idxs = np.where(epos['ipp']>1)[0]
    
    for multi_idx in multi_idxs:
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        
        idx1 = -1
        idx2 = -1
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                idx1 = int(np.floor(cluster[i]/delta))
                idx2 = int(np.floor(cluster[j]/delta))
                if idx1 < N and idx2 < N:
                    corrhist[idx1,idx2] += 1
                
    return corrhist+corrhist.T-np.diag(np.diag(corrhist))
                
                
                
        
def dummy():        
    
    # Voltage and bowl correct ToF data
    from voltage_and_bowl import do_voltage_and_bowl

    p_volt = np.array([])
    p_bowl = np.array([])
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)        
        
    epos_vb = epos.copy()
    
    epos_vb['tof'] = tof_corr.copy()
    
    import voltage_and_bowl
    
    tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
    epos_v = epos.copy()
    epos_v['tof'] = tof_vcorr.copy()
    
    tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
    epos_b = epos.copy()
    epos_b['tof'] = tof_bcorr.copy()
    
    ROI = [0, None]
    
    ch = histogram_functions.corrhist(epos)
    fig1 = plt.figure(num=1)
    plt.clf()    
    plt.imshow(np.log2(1+ch))
    plt.title('raw')
    fig1.gca().set_xlim(ROI[0],ROI[1])
    fig1.gca().set_ylim(ROI[0],ROI[1])
    
    ch = histogram_functions.corrhist(epos_v)
    fig2 = plt.figure(num=2)
    plt.clf()    
    plt.imshow(np.log2(1+ch))
    plt.title('volt')
    fig2.gca().set_xlim(ROI[0],ROI[1])
    fig2.gca().set_ylim(ROI[0],ROI[1])
    
    ch = histogram_functions.corrhist(epos_b)
    fig3 = plt.figure(num=3)
    plt.clf()    
    plt.imshow(np.log2(1+ch))
    plt.title('bowl')
    fig3.gca().set_xlim(ROI[0],ROI[1])
    fig3.gca().set_ylim(ROI[0],ROI[1])
    
    ch = histogram_functions.corrhist(epos_vb)
    fig4 = plt.figure(num=4)
    plt.clf()    
    plt.imshow(np.log10(1+ch))
    plt.title('v+b')
#    fig4.gca().set_xlim(ROI[0],ROI[1])
#    fig4.gca().set_ylim(ROI[0],ROI[1])
    
    idxs = np.where(epos['ipp'] == 2)[0]
    
    
    fig5 = plt.figure(num=5)
    plt.clf()    
    dts = np.abs(tof_corr[idxs]-tof_corr[idxs+1])
    plt.hist(dts,bins=np.arange(0,2000,.5),label='deltaT')
    plt.hist(tof_corr[np.r_[idxs,idxs+1]],bins=np.arange(0,2000,.5),label='since t0')
    
    
    
    
    
    
    
    
    
    fig66 = plt.figure(num=66)
    plt.clf()    
    dts = np.abs(tof_corr[idxs]-tof_corr[idxs+1])
#    sus = np.sqrt(tof_corr[idxs]**2+tof_corr[idxs+1]**2)
#    sus = np.fmax(tof_corr[idxs],tof_corr[idxs+1])
    sus = (tof_corr[idxs]+tof_corr[idxs+1])/np.sqrt(2)
    plt.plot(sus,dts,'.',ms=1,alpha=1)
#    fig66.gca().axis('equal')
    fig66.gca().set_xlim(0,7000)
    fig66.gca().set_ylim(-100, 800)
    
    
    
    
    
    
    return
    
        
    
    
    
    
    
    
