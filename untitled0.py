
# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats


# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat
from voltage_and_bowl import do_voltage_and_bowl


# Read in template spectrum
ref_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
#ref_fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg Doped
ref_epos = apt_fileio.read_epos_numpy(ref_fn)

#ref_epos = ref_epos[0:ref_epos.size//2]


# Read in data
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg Doped
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"

epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[1450000:-1]


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr,nom_voltage=np.mean(epos['v_dc']))
print(p_m2q)

plotting_stuff.plot_histo(ref_epos['m2q'],4,user_label='ref')
plotting_stuff.plot_histo(m2q_corr[m2q_corr.size//2:-1],4,clearFigure=False,user_label='[c,t0] corr')

xs,ys = bin_dat(tof_corr,isDensity=True,bin_width=0.25)

bg_lvl = np.mean(ys[(xs>1000) & (xs<3500)])

idxs = np.nonzero((xs>200) & (xs<1000))

xs = xs[idxs]
ys = ys[idxs]

fig = plt.figure(num=113)
fig.clear()
ax = fig.gca()
ax.plot(xs,np.cumsum(ys**2))
ax.plot(xs,np.cumsum((ys-bg_lvl)**2))




















# Read in data
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
epos = apt_fileio.read_epos_numpy(fn)
epos = epos[0:1000000]


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Find c and t0 for ToF data based on aligning to reference spectrum
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr,nom_voltage=np.mean(epos['v_dc']))
print(p_m2q)

plotting_stuff.plot_histo(ref_epos['m2q'],4,user_label='ref')
plotting_stuff.plot_histo(m2q_corr[m2q_corr.size//2:-1],4,clearFigure=False,user_label='[c,t0] corr')

xs,ys = bin_dat(tof_corr,isDensity=True,bin_width=0.25)

bg_lvl = np.mean(ys[(xs>1000) & (xs<3500)])

idxs = np.nonzero((xs>200) & (xs<1000))

xs = xs[idxs]
ys = ys[idxs]

fig = plt.figure(num=113)
fig.clear()
ax = fig.gca()
ax.plot(xs,np.cumsum(ys**2))
ax.plot(xs,np.cumsum((ys-bg_lvl)**2))





