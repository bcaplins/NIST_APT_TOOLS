# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import apt_importers as apt
from voltage_and_bowl import do_voltage_and_bowl
import numpy as np
import time

t_i = time.time()

fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
epos = apt.read_epos_numpy(fn)
wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)

p_volt = np.array([])
p_bowl = np.array([])

print(str(time.time()-t_i))
t_i = time.time()

tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)

print(str(time.time()-t_i))