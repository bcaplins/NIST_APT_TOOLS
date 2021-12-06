# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import GaN_fun
import GaN_type_peak_assignments

import peak_param_determination as ppd
from histogram_functions import bin_dat

plt.close('all')

# Read in data
epos = GaN_fun.load_epos(run_number='R20_07249', 
                         epos_trim=[5000, 5000],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.GaN()
#pk_data = GaN_type_peak_assignments.Mg_doped_GaN()
#pk_data = GaN_type_peak_assignments.AlGaN()

bg_rois=[[0.4,0.9]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.1, 
        bg_rois=bg_rois)

# all ions
#idxs = np.arange(epos.size)

# singles only
#idxs = np.where(epos['ipp']==1)[0]

# all multiples (no singles)
idxs = np.where(epos['ipp']!=1)[0]

# doubles only
#idxs = np.where(epos['ipp']==2)[0]
#idxs = np.sort(np.concatenate((idxs,idxs+1)))

# triples only
#idxs = np.where(epos['ipp']==3)[0]
#idxs = np.sort(np.concatenate((idxs,idxs+1,idxs+2)))


sub_epos = epos[idxs]

cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=2)

# Print out the composition of the full dataset
ppd.pretty_print_compositions(compositions,pk_data)