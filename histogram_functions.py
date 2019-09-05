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
