# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
import os
parent_path = '..\\nistapttools'
if parent_path not in sys.path:
    sys.path.append(os.path.abspath(parent_path))    
import time

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat
from voltage_and_bowl import do_voltage_and_bowl

sub_path = '.\\SEDcorr'
if sub_path not in sys.path:
    sys.path.append(os.path.abspath(sub_path))    
    
from SEDcorr import sed_corr



# Read in template spectrum
ref_fn = r"Ga2O3 epos files\R20_28215_3800pA.epos"
#ref_fn = r"Ga2O3 epos files\R44_03672.epos"
#ref_fn = r"Ga2O3 epos files\R44_03569.epos"
#ref_fn = r"Ga2O3 epos files\R44_03696_5pJ.epos"
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
#fn = r"Ga2O3 epos files\R20_18170.epos"
#fn = r"Ga2O3 epos files\R20_28219_500pA.epos"
#fn = r"Ga2O3 epos files\R20_28218_862pA.epos"
#fn = r"Ga2O3 epos files\R20_28224_1740pA.epos"
#fn = r"Ga2O3 epos files\R20_28216_4000pA.epos"
#fn = r"Ga2O3 epos files\R20_28220_12000pA.epos"
#fn = r"Ga2O3 epos files\R20_28221_21900pA.epos"
#fn = r"Ga2O3 epos files\R20_28222_69300pA.epos"
#fn = r"Ga2O3 epos files\R44_03569.epos"
#fn = r"Ga2O3 epos files\R20_28227_68600pA.epos"
#fn = r"Ga2O3 epos files\R20_28228_22300pA.epos"
#fn = r"Ga2O3 epos files\R20_28229_12100pA.epos"
#fn = r"Ga2O3 epos files\R20_28230_3990pA.epos"
#fn = r"Ga2O3 epos files\R20_28231_1720pA.epos"
#fn = r"Ga2O3 epos files\R20_28233_505pA.epos"
#fn = r"Ga2O3 epos files\R20_28234_1720pA.epos"
#fn = r"Ga2O3 epos files\R20_28235_12100pA.epos"
#fn = r"Ga2O3 epos files\R20_28236_68600pA.epos"
#fn = r"Ga2O3 epos files\R20_28250.epos"
#fn = r"Ga2O3 epos files\R20_28251.epos"
#fn = r"Ga2O3 epos files\R20_28253_4300pA.epos"
fn = r"Ga2O3 epos files\R20_28254_11500pA.epos"
#fn = r"Ga2O3 epos files\R20_28255_1770pA.epos"
#fn = r"Ga2O3 epos files\R20_28256_925pA.epos"
#fn = r"Ga2O3 epos files\R20_28257_500pA.epos"
#fn = r"Ga2O3 epos files\R20_28258_265pA.epos"
#fn = r"Ga2O3 epos files\R20_28259_925pA.epos"
#fn = r"Ga2O3 epos files\R20_28260_1770pA.epos"
#fn = r"Ga2O3 epos files\R20_28261_11500pA.epos"
#fn = r"Ga2O3 epos files\R20_28262_4300pA.epos"
#fn = r"Ga2O3 epos files\R44_03672.epos"
#fn = r"Ga2O3 epos files\R44_03695_200fJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_500fJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_5pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_1pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_10pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_20pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_40pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_80pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_120pJ.epos"
#fn = r"Ga2O3 epos files\R44_03695_160pJ.epos"
#fn = r"Ga2O3 epos files\R44_03696_500fJ.epos"
#fn = r"Ga2O3 epos files\R44_03696_1pJ.epos"
#fn = r"Ga2O3 epos files\R44_03696_5pJ.epos"
#fn = r"Ga2O3 epos files\R44_03697_10pJ.epos"
#fn = r"Ga2O3 epos files\R44_03697_20pJ.epos"
#fn = r"Ga2O3 epos files\R44_03697_40pJ.epos"
#fn = r"Ga2O3 epos files\R44_03697_80pJ.epos"
#fn = r"Ga2O3 epos files\R44_03698_120pJ.epos"
#fn = r"Ga2O3 epos files\R44_03698_160pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_1pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_5pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_10pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_20pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_40pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_80pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_81pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_120pJ.epos"
#fn = r"Ga2O3 epos files\R44_03699_160pJ.epos"


epos = apt_fileio.read_epos_numpy(fn)
#epos['tof'] = epos['tof']-13

#epos = epos[25000:-1]

# Plot TOF vs event index and show the current ROI selection
roi_event_idxs = np.arange(1000,epos.size-1000)
#roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_TOF_vs_time(epos['tof'],epos,1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
wall_time = np.cumsum(epos['pslep'])/25000.0
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()

# NOTE THE VALUE OF SKIP_VOLTAGE AND CHANGE ACCORDING TO DATA. IF sv = CONSTANT THEN USE TRUE!!
#if epos['v_dc'].std() != 0:
#    raise Exception('Voltage DID vary.')
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl, skip_voltage=False) 
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Plot TOF vs event index with voltage overlaid to show if voltage corr went ok
ax = plotting_stuff.plot_TOF_vs_time(tof_corr,epos,2)
ax.set_title('voltage and bowl corrected')

# Plot slices from the detector to show if bowl corr went ok
plotting_stuff.plot_bowl_slices(tof_corr,epos,3,clearFigure=True,user_ylim=[0,1200])

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)



pointwise_scales = sed_corr.get_all_scale_coeffs(m2q_corr,
                                                max_scale=1.15,
                                                  roi=[0.5,75],
                                                  cts_per_chunk=2**10,
                                                  delta_logdat=5e-4)

#CHANGE TO pointwise_scales=1 IF WANT TO AVOID USING CHARGING CORRECTION                             
#pointwise_scales = 1;                                                    
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
#ref_pk_m2qs = np.array([    ed['H'].isotopes[1][0],
#                            ed['H'].isotopes[2][0],
#                            ed['H'].isotopes[3][0],
#                        ed['Si'].isotopes[28][0]/2,
#                        ed['O'].isotopes[16][0]/1,
#                        ed['Ga'].isotopes[69][0]/3,
#                        ed['Ga'].isotopes[71][0]/3,
#                        ed['Si'].isotopes[28][0],
#                        ed['O'].isotopes[16][0]*2,
#                        ed['Ga'].isotopes[69][0]/2,
#                        ed['Ga'].isotopes[71][0]/2,
#                        (ed['Ga'].isotopes[69][0]+ed['O'].isotopes[16][0])/2,
#                        (ed['Ga'].isotopes[71][0]+ed['O'].isotopes[16][0])/2,
#                        ed['Ga'].isotopes[69][0]/1,
#                        ed['Ga'].isotopes[71][0]/1,
#                        (ed['Ga'].isotopes[69][0]*2+ed['O'].isotopes[16][0])/2,
#                        (ed['Ga'].isotopes[71][0]*2+ed['O'].isotopes[16][0])/2,
#                        (ed['Ga'].isotopes[69][0]*2+ed['O'].isotopes[16][0]*2)/2,
#                        (ed['Ga'].isotopes[71][0]*2+ed['O'].isotopes[16][0]*2)/2,
#                        (ed['Ga'].isotopes[69][0]*2+ed['O'].isotopes[16][0]*3)/2,
#                        (ed['Ga'].isotopes[71][0]*2+ed['O'].isotopes[16][0]*3)/2,
#                        (ed['Ga'].isotopes[69][0]+ed['O'].isotopes[16][0]*2),
#                        (ed['Ga'].isotopes[71][0]+ed['O'].isotopes[16][0]*2)])

ref_pk_m2qs = np.array([    ed['H'].isotopes[1][0],
                            ed['H'].isotopes[2][0],
                        ed['O'].isotopes[16][0]/1,
                        ed['O'].isotopes[16][0]*2,
                        ed['Ga'].isotopes[69][0]/2,
                        ed['Ga'].isotopes[71][0]/2,
                        ed['Ga'].isotopes[69][0]/1,
                        ed['Ga'].isotopes[71][0]/1])


# Perform 'linearization' m2q calibration
m2q_corr2 = m2q_calib.calibrate_m2q_by_peak_location(m2q_corr_q,ref_pk_m2qs)

# Plot the reference spectrum, (c, t0) corr spectrum and linearized spectrum
#     to confirm that mass calibration went ok
user_xlim=[0,120]
plotting_stuff.plot_histo(ref_epos['m2q'],fig_idx=4,user_label='ref',user_xlim=user_xlim)
plotting_stuff.plot_histo(m2q_corr_q,fig_idx=4,clearFigure=False,user_label='[c,t0] corr',user_xlim=user_xlim)
ax = plotting_stuff.plot_histo(m2q_corr2,4,clearFigure=False,user_label='linearized',user_xlim=user_xlim)
#for ref_pk_m2q in ref_pk_m2qs:
#    ax.plot(ref_pk_m2q*np.ones(2),np.array([1,1e4]),'k--')

# plotting_stuff.plot_bowl_slices(epos['tof'],epos,3,clearFigure=True,user_ylim=[0,1200])


# Save the data as a new epos file
epos['m2q'] = m2q_corr2
new_fn = fn[:-5]+'_vmbq_corr.epos'
apt_fileio.write_epos_numpy(epos,new_fn)







