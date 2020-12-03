# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Need to put the functions in the path
# Probably not necessary if I understood Python/Git/modules better
import os 
import sys
parent_directory = os.getcwd().rsplit(sep='\\',maxsplit=1)[0]
if parent_directory not in sys.path:
    sys.path.insert(1, parent_directory)

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


# Read in template spectrum
#work_dir = r"C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript"
work_dir = r"C:\Users\lnm\Documents\Div 686\Data"

ref_fn = work_dir+"\\"+"180821_GaN_A71"+"\\"+"R20_07094-v03.epos"
ref_epos = apt_fileio.read_epos_numpy(ref_fn)
#ref_epos = ref_epos[0:ref_epos.size//2]

# Read in data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\190421_AlGaN50p7_A83\R20_07208-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07247.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07167-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07148-v02.epos"
#fn = r"\\cfs2w.campus.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181204_InGaNQW_A73\R20_07144-v02.epos"
fn = work_dir+"\\"+"190406_InGaNQW_A82"+"\\"+"R44_03146.epos"



epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[25000:-1]

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

# Plot TOF vs event index with voltage overlaid to show if voltage corr went ok
ax = plotting_stuff.plot_TOF_vs_time(tof_corr,epos,2)
ax.set_title('voltage and bowl corrected')

# Plot slices from the detector to show if bowl corr went ok
plotting_stuff.plot_bowl_slices(tof_corr,epos,3,clearFigure=True,user_ylim=[0,1200])

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)

# Define calibration peaks
ed = initElements_P3.initElements()
ref_pk_m2qs = np.array([    ed['H'].isotopes[1][0],
                        2*ed['H'].isotopes[1][0],
                        (1/2)*ed['N'].isotopes[14][0],
                        ed['N'].isotopes[14][0],
                    2*ed['N'].isotopes[14][0],
                    (1/2)*ed['Ga'].isotopes[69][0],
                    (1/2)*ed['Ga'].isotopes[71][0],
                    ed['Ga'].isotopes[69][0],
                    ed['Ga'].isotopes[71][0]])

# Perform 'linearization' m2q calibration
m2q_corr2 = m2q_calib.calibrate_m2q_by_peak_location(m2q_corr,ref_pk_m2qs)

# Plot the reference spectrum, (c, t0) corr spectrum and linearized spectrum
#     to confirm that mass calibration went ok
user_xlim=[0,150]
plotting_stuff.plot_histo(ref_epos['m2q'],4,user_label='ref',user_xlim=user_xlim)
plotting_stuff.plot_histo(m2q_corr,4,clearFigure=False,user_label='[c,t0] corr',user_xlim=user_xlim)
ax = plotting_stuff.plot_histo(m2q_corr2,4,clearFigure=False,user_label='linearized',user_xlim=user_xlim)
for ref_pk_m2q in ref_pk_m2qs:
    ax.plot(ref_pk_m2q*np.ones(2),np.array([1,1e4]),'k--')

# Save the data as a new epos file
epos['m2q'] = m2q_corr2
new_fn = fn[:-5]+'_vbm_corr.epos'
apt_fileio.write_epos_numpy(epos,new_fn)
