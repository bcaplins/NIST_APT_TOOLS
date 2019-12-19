# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import time

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat
from voltage_and_bowl import do_voltage_and_bowl

plt.close('all')

# Read in template spectrum
ref_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos"
ref_epos = apt_fileio.read_epos_numpy(ref_fn)
ref_epos = ref_epos[130000:]

# Read in data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07263-v02.epos" # 25 K
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07080-v01.epos" # 50 K
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07086-v01.epos" # 125 K
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\Final EPOS for APL Mat Paper\R20_07276-v03.epos" # 150 K

epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[epos.size//2:-1]

# Plot TOF vs event index and show the current ROI selection
#roi_event_idxs = np.arange(1000,epos.size-1000)
roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_TOF_vs_time(epos['tof'],epos,1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)
#
import sel_align_m2q_log_xcorr
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr.get_all_scale_coeffs(m2q_corr,
                                                                                 m2q_roi=[0.5,75],
                                                                                 cts_per_slice=2**8,
                                                                                 max_scale=1.15)
# Compute corrected data
m2q_corr_q = m2q_corr/pointwise_scales

# Convert back to tof
tof_corr_q = np.sqrt(m2q_corr_q/(p_m2q[0]*1e-4))+p_m2q[1]

# Plot TOF vs event index with voltage overlaid to show if voltage corr went ok
ax = plotting_stuff.plot_TOF_vs_time(tof_corr_q,epos,2)
ax.set_title('voltage and bowl corrected')

# Plot slices from the detector to show if bowl corr went ok
plotting_stuff.plot_bowl_slices(tof_corr_q,epos,3,clearFigure=True,user_ylim=[0,1200])

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr_q, p_m2q_q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr_q)
   
# Define calibration peaks
ed = initElements_P3.initElements()
ref_pk_m2qs = np.array([    ed['H'].isotopes[1][0],
                        ed['Si'].isotopes[28][0]/3,
                        ed['Si'].isotopes[28][0]/2,
                        ed['O'].isotopes[16][0]/1,
                        (ed['Si'].isotopes[28][0]+ed['O'].isotopes[16][0])/2,
                        ed['O'].isotopes[16][0]*2,
                        (ed['Si'].isotopes[28][0]+ed['O'].isotopes[16][0])/1,
                        (ed['Si'].isotopes[28][0]+2*ed['O'].isotopes[16][0])/1])

# Perform 'linearization' m2q calibration
m2q_corr2 = m2q_calib.calibrate_m2q_by_peak_location(m2q_corr_q,ref_pk_m2qs)

# Plot the reference spectrum, (c, t0) corr spectrum and linearized spectrum
#     to confirm that mass calibration went ok
plotting_stuff.plot_histo(ref_epos['m2q'],fig_idx=4,user_label='ref')
plotting_stuff.plot_histo(m2q_corr_q,fig_idx=4,clearFigure=False,user_label='[c,t0] corr')
ax = plotting_stuff.plot_histo(m2q_corr2,4,clearFigure=False,user_label='linearized')
for ref_pk_m2q in ref_pk_m2qs:
    ax.plot(ref_pk_m2q*np.ones(2),np.array([1,1e4]),'k--')

# Save the data as a new epos file
epos['m2q'] = m2q_corr2
new_fn = fn[:-5]+'_vbmq_corr.epos'
apt_fileio.write_epos_numpy(epos,new_fn)






plotting_stuff.plot_bowl_slices(epos['tof'],epos,3,clearFigure=True,user_ylim=[0,1200])




