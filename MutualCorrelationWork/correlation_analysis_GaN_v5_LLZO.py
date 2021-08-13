
# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
import os
parent_path = '..\\nistapttools'
if parent_path not in sys.path:
    sys.path.append(os.path.abspath(parent_path))    

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3
import histogram_functions 

import peak_param_determination as ppd

from histogram_functions import bin_dat
import voltage_and_bowl
from voltage_and_bowl import do_voltage_and_bowl
from voltage_and_bowl import mod_full_vb_correction

import colorcet as cc


def create_histogram(xs, ys, x_roi=None, delta_x=0.1, y_roi=None, delta_y=0.1):
    """Create a 2d histogram of the data, specifying the bin intensity, region
    of interest (on the y-axis), and the spacing of the y bins"""
    # even number
    num_x = int(np.ceil((x_roi[1]-x_roi[0])/delta_x))
    num_y = int(np.ceil((y_roi[1]-y_roi[0])/delta_y))

    return np.histogram2d(xs, ys, bins=[num_x, num_y],
                          range=[x_roi, y_roi],
                          density=False)

def _extents(f):
    """Helper function to determine axis extents based off of the bin edges"""
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


def plot_2d_histo(ax, N, x_edges, y_edges, scale='log'):
    if scale=='log':
        dat = np.log10(1+N)
    elif scale=='lin':
        dat = N
            
    """Helper function to plot a histogram on an axis"""
    ax.imshow(np.transpose(dat), aspect='auto',
              extent=_extents(x_edges) + _extents(y_edges),
              origin='lower', cmap=cc.cm.CET_L8,
              interpolation='nearest')


def corrhist(epos, delta=1, roi=None):
    dat = epos['tof']
    if roi is None:
        roi = [0, 1000]
    
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
                if idx1 < N and idx1>=0 and idx2 < N and idx2>=0:
                    corrhist[idx1,idx2] += 1
    
    edges = np.arange(roi[0],roi[1]+delta,delta)
    assert edges.size-1 == N
    
    return (edges, corrhist+corrhist.T-np.diag(np.diag(corrhist)))
            

       
def calc_t0(tof,tof_vcorr_fac,tof_bcorr_fac,sigma):
    BB = tof_bcorr_fac[0::2]+tof_bcorr_fac[1::2]       
    t0 = ((tof_bcorr_fac[0::2]*tof[0::2]+tof_bcorr_fac[1::2]*tof[1::2]) - sigma/(tof_vcorr_fac[0::2]))/BB    
    t0 = np.ravel(np.column_stack((t0,t0)))    
    return t0

def create_sigma_delta_histogram(raw_tof, tof_vcorr_fac, tof_bcorr_fac, sigmas=None, delta_range=None, delta_step=0.5):
    # Must be a doubles only epos...
    
    # scan through a range of sigmas and compute the corrected data
    if sigmas is None:
        sigmas = np.linspace(0,2000,2**7)
    
    if delta_range is None:
        delta_range = [0,700]
    
    delta_n_bins = int((delta_range[1]-delta_range [0])/delta_step)
#    print('delta_n_bins = '+str(delta_n_bins))
    
    res_dat = np.zeros((sigmas.size,delta_n_bins))        
    
    for sigma_idx in np.arange(sigmas.size):
        
        t0 = calc_t0(raw_tof, tof_vcorr_fac, tof_bcorr_fac, sigmas[sigma_idx])
        tof_corr = tof_vcorr_fac*tof_bcorr_fac*(raw_tof-t0)
    
        dts = np.abs(tof_corr[:-1:2]-tof_corr[1::2])
               
        N, delta_edges = np.histogram(dts, bins=delta_n_bins, range=delta_range)
        res_dat[sigma_idx,:] = N
        
        if np.mod(sigma_idx,10)==0:
            print("Loop index "+str(sigma_idx+1)+" of "+str(sigmas.size))
            
    delta_centers = 0.5*(delta_edges[:-1]+delta_edges[1:])

    return (res_dat, sigmas, delta_centers)

def interleave(a,b):
    return np.ravel(np.column_stack((a,b)))


def calc_slope_and_intercept(raw_tof, volt_coeff, bowl_coeff):
    A = volt_coeff[0::2]
    B_alpha = bowl_coeff[0::2]
    B_beta = bowl_coeff[1::2]
    tof_alpha = raw_tof[0::2]
    tof_beta = raw_tof[1::2]
    
    intercept = 2*A*B_alpha*B_beta*(tof_beta-tof_alpha)/(B_alpha+B_beta)
    slope = (B_beta-B_alpha)/(B_beta+B_alpha)
    
    return (slope, intercept)

# Note that x is sums and y is diffs 
def compute_dist_to_line(slope, intercept, x, y):    
    return np.abs(intercept+slope*x-y)/np.sqrt(1+slope**2)




def calc_parametric_line(raw_tof, volt_coeff, bowl_coeff, n=2):
    
    if n>0:
        t = raw_tof.reshape(-1,n)        
        v = volt_coeff.reshape(-1,n)
        b = bowl_coeff.reshape(-1,n)
    else:
        t = raw_tof
        v = volt_coeff
        b = bowl_coeff
        
         
    r0 = v*b*(t-np.sum(b*t,axis=1)[:,np.newaxis]/np.sum(b,axis=1)[:,np.newaxis])
    r1 = b/np.sum(b,axis=1)[:,np.newaxis]
    
    return (r0, r1)

def compute_dist_to_parametric_line(r0, r1, q):    
    # q is n_pts by n_dim
    sigma = (np.dot(r1,q.T)-np.sum(r0*r1,axis=1)[:,np.newaxis])/np.sum(r1**2,axis=1)[:,np.newaxis]        
    d = np.sqrt(np.sum(((r0[:,np.newaxis,:]+np.einsum("np,nd->npd",sigma,r1))-q[np.newaxis,...])**2, axis=-1))
    return d, sigma

#from itertools import combinations, chain
#from scipy.special import comb
#def comb_index(n, k):
#    count = comb(n, k, exact=True)
#    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
#                        int, count=count*k)
#    return index.reshape(-1, k)

def cartesian_product(arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)



import GaN_type_peak_assignments



plt.close('all')

#fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07094-v03.epos'
#fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07148-v01.epos'
fn = r'C:\Users\capli\Downloads\R58_07445-v02.epos'

epos = apt_fileio.read_epos_numpy(fn)

# Split out data into single and doubles
sing_idxs = np.where(epos['ipp'] == 1)[0]
epos_s = epos[sing_idxs]

tmp_idxs = np.where(epos['ipp'] == 2)[0]
doub_idxs = np.ravel(np.column_stack([tmp_idxs+i for i in range(2)]))

epos_d = epos[doub_idxs]

# voltage and bowl correct ToF data.  
p_volt = np.array([1, 0])
p_volt = np.array([])
p_bowl = np.array([ 0.89964083, -0.43114144, -0.27484715, -0.25883824])

# only use singles for V+B
# subsample down to 1 million ions for speedier computation
vb_idxs = np.random.choice(epos_s.size, int(np.min([epos_s.size, 1000*1000])), replace=False)
vb_epos = epos_s[vb_idxs]

tof_sing_corr, p_volt, p_bowl = do_voltage_and_bowl(vb_epos,p_volt,p_bowl)        

tof_vcorr_fac = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos['tof']),epos['v_dc'])
tof_bcorr_fac = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos['tof']),epos['x_det'],epos['y_det'])

# find the voltage and bowl coefficients for the doubles data
tof_vcorr_fac_d = tof_vcorr_fac[doub_idxs]
tof_bcorr_fac_d = tof_bcorr_fac[doub_idxs]

# Find transform to m/z space
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(epos_s['m2q'],tof_sing_corr)
epos['m2q'] = m2q_calib.mod_physics_m2q_calibration(p_m2q,mod_full_vb_correction(epos,p_volt,p_bowl))

plotting_stuff.plot_histo(epos['m2q'],fig_idx=1,  user_xlim=[0, 250],user_bin_width=0.03)

import GaN_fun

pk_data = GaN_type_peak_assignments.GaN_with_H()
bg_rois=[[0.4,0.9]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.05, 
        bg_rois=bg_rois)


cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=epos, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=3)

ppd.pretty_print_compositions(compositions,pk_data)

def m2q_to_tof(m2q, p_m2q):
    return np.sqrt(1e4*m2q/p_m2q[0])+p_m2q[1]

pk_m2qs = pk_params[is_peak.ravel()]['x0_mean_shift']
pk_times = m2q_to_tof(pk_m2qs, p_m2q)





# Make a correlation histogram of doubles
epos_d_vb = epos_d.copy()
epos_d_vb['tof'] = mod_full_vb_correction(epos_d,p_volt,p_bowl)

edges, ch = corrhist(epos_d_vb,roi = [0, 1000], delta=1)
centers = (edges[1:]+edges[:-1])/2.0

fig2 = plt.figure(num=3)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges, scale='log')
ax2.axis('equal')
ax2.axis('square')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')




# Remove peaks from doubles
idxs_to_remove = np.array([],dtype='int64')

win_wid = 5
for pk_idx in range(pk_times.size):
    idxs_to_remove = np.append(idxs_to_remove, np.where((epos_d_vb['tof']>= (pk_times[pk_idx]-win_wid/2)) & (epos_d_vb['tof']<= (pk_times[pk_idx]+win_wid/2)))[0])
    

idxs_to_remove = np.unique((idxs_to_remove/2).astype('int64')*2)
idxs_to_remove = np.append(idxs_to_remove,idxs_to_remove+1)
idxs_to_remove.sort()

epos_d_pruned = np.delete(epos_d, idxs_to_remove)
epos_d_vb_pruned = np.delete(epos_d_vb, idxs_to_remove)


# Make a correlation histogram of doubles without crosspeaks
edges, ch = corrhist(epos_d_vb_pruned,roi = [0, 5000], delta=1)
centers = (edges[1:]+edges[:-1])/2.0

fig2 = plt.figure(num=3)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges, scale='log')
ax2.axis('equal')
ax2.axis('square')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')





plotting_stuff.plot_histo(epos_d['m2q'],112233,user_bin_width=.11,user_xlim=[0,80])    
plotting_stuff.plot_histo(epos_d_pruned['m2q'],112233,clearFigure=False,user_bin_width=.1,user_xlim=[0,80])    





pk_coords = cartesian_product([pk_times for i in range(2)])



#tofin = epos_d['tof']

THRESH = 2


tofin = epos_d_pruned['tof'].copy()
vcorr_fac_in = np.delete(tof_vcorr_fac_d, idxs_to_remove)
bcorr_fac_in = np.delete(tof_bcorr_fac_d, idxs_to_remove)

tofin = np.reshape(tofin,(int(tofin.size/2),2))
vcorr_fac_in = np.reshape(vcorr_fac_in,(int(vcorr_fac_in.size/2),2))
bcorr_fac_in = np.reshape(bcorr_fac_in,(int(bcorr_fac_in.size/2),2))



N_SIGMA_PTS = 1024
DEL_SIGMA_PTS = 1600/N_SIGMA_PTS

# create data delta sigma histogram
sigmas = np.arange(N_SIGMA_PTS)*DEL_SIGMA_PTS
res_dat, sigmas, deltas = create_sigma_delta_histogram(tofin.flatten(), 
                                                       vcorr_fac_in.flatten(),
                                                       bcorr_fac_in.flatten(),
                                                       sigmas=sigmas,
                                                       delta_range=[0,700],
                                                       delta_step=0.125)

fig = plt.figure(num=111)
plt.clf()
ax = fig.gca()

ax.imshow(np.log10(1+res_dat.T[:,:]), aspect='auto',
          extent=_extents(sigmas) + _extents(deltas),
          origin='lower', cmap=cc.cm.CET_L8,
          interpolation='nearest')

ax.set_xlabel('$\Sigma_c$ (ns)')
ax.set_ylabel('$\Delta_c$ (ns)')
ax.set_title('histogram (counts)')

sum_pts = np.abs(pk_coords[:,1]+pk_coords[:,0])
del_pts = np.abs(pk_coords[:,1]-pk_coords[:,0])

ax.scatter(sum_pts,del_pts, color='w', alpha=0.5, c=None, facecolors='none', lw=2)










from scipy.ndimage.filters import uniform_filter1d

def smoothy(x,N=8):
    return uniform_filter1d(x, size=N)


fig = plt.figure(num=3333)
plt.clf()
ax = fig.gca()
ax.plot(smoothy(deltas),1e3*smoothy(res_dat[820,:].flatten()))
ax.plot(smoothy(deltas),1e2*smoothy(res_dat[410,:].flatten()))
ax.plot(smoothy(deltas),1e1*smoothy(res_dat[205,:].flatten()))
ax.plot(smoothy(deltas),1e0*smoothy(res_dat[102,:].flatten()))
ax.set_xlim(0,700)


#
#
#
#multi_idx = np.where(epos_d_pruned['ipp']==2)[0]
#tofin = np.zeros((np.sum(epos_d_pruned['ipp'][multi_idx]-1), 2))
#vcorr_fac_in = np.zeros((np.sum(epos_d_pruned['ipp'][multi_idx]-1), 2))
#bcorr_fac_in = np.zeros((np.sum(epos_d_pruned['ipp'][multi_idx]-1), 2))
#
#ct = 0
#for idx in multi_idx:
#    n = epos_d_pruned['ipp'][idx]    
#    tofin[ct:ct+n-1,0] = epos_d_pruned['tof'][idx:idx+n-1]
#    tofin[ct:ct+n-1,1] = epos_d_pruned['tof'][idx+1:idx+n]
#    vcorr_fac_in[ct:ct+n-1,0] = tof_vcorr_fac_d[idx:idx+n-1]
#    vcorr_fac_in[ct:ct+n-1,1] = tof_vcorr_fac_d[idx+1:idx+n]
#    bcorr_fac_in[ct:ct+n-1,0] = tof_bcorr_fac_d[idx:idx+n-1]
#    bcorr_fac_in[ct:ct+n-1,1] = tof_bcorr_fac_d[idx+1:idx+n]
#    ct += n-1
#       






       
r0,r1 = calc_parametric_line(tofin, vcorr_fac_in, bcorr_fac_in, n=2)
diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)

min_idxs = np.argmin(diff_mat,axis=1)
min_d = np.min(diff_mat,axis=1)
closest_sums = sigmas[range(sigmas.shape[0]), min_idxs]
closest_sums[min_d>THRESH] = 957

corr_dat = r0+r1*closest_sums[:,np.newaxis]

plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=True, scale_factor=1e0, user_color='C2')
#
#plotting_stuff.plot_histo(corr_dat[min_d<THRESH,1]-corr_dat[min_d<THRESH,0],fig_idx=332211,user_xlim=[-0,5000], user_bin_width=0.5, clearFigure=True)
#plotting_stuff.plot_histo(corr_dat[min_d>=THRESH,1]-corr_dat[min_d>=THRESH,0],fig_idx=332211,user_xlim=[-0,5000], user_bin_width=0.5, clearFigure=False)


closest_sums[:] = 957
corr_dat = r0+r1*closest_sums[:,np.newaxis]
plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=False, scale_factor=1e0, user_color='C1')






dd = np.random.exponential(size=tofin.size//2, scale=300.0)
ss = 10000.0*np.random.rand(tofin.size//2)
ttt1 = (ss-dd)/2
ttt2 = (ss+dd)/2
tofin[:,0] = ttt1
tofin[:,1] = ttt2
    

  
r0,r1 = calc_parametric_line(tofin, vcorr_fac_in, bcorr_fac_in, n=-1)
diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)

min_idxs = np.argmin(diff_mat,axis=1)
min_d = np.min(diff_mat,axis=1)
closest_sums = sigmas[range(sigmas.shape[0]), min_idxs]
closest_sums[min_d>THRESH] = 957


corr_dat = r0+r1*closest_sums[:,np.newaxis]

tmp = epos_d_pruned['tof']*vcorr_fac_in.flatten()*bcorr_fac_in.flatten()

plotting_stuff.plot_histo(tmp[1::2]-tmp[::2],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=False, scale_factor=1e0, user_color='C0')








plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],
                          fig_idx=332211,
                          user_xlim=[-0,1000],
                          user_bin_width=0.5,
                          clearFigure=False,
                          scale_factor=0.1)







closest_sums[:] = 957
corr_dat = r0+r1*closest_sums[:,np.newaxis]
plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],
                          fig_idx=332211,
                          user_xlim=[-0,1000],
                          user_bin_width=0.5,
                          clearFigure=False,
                          scale_factor=0.1)


for line in plt.gca().lines:
    line.set_linewidth(2.)


plt.grid(False)

plt.gca().set_xlim(0,700)
plt.gca().set_ylim(1,1e4)















# Fit out the 2D histogram
pk_params_2d = np.zeros((pk_times.size,pk_times.size), dtype={'names':('t1','t2','sigma','amp','cts'),
                          'formats':('f8','f8','f8','f8','i8')})












import sys
sys.exit(0)













