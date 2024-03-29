
# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
parent_path = '..\\'
if parent_path not in sys.path:
    sys.path.append(parent_path)    

# custom imports
from nistapttools import apt_fileio
from nistapttools import m2q_calib
from nistapttools import plotting_stuff
from nistapttools import initElements_P3
from nistapttools import histogram_functions 

import nistapttools.peak_param_determination as ppd

from nistapttools.histogram_functions import bin_dat
from nistapttools import voltage_and_bowl
from nistapttools.voltage_and_bowl import do_voltage_and_bowl
from nistapttools.voltage_and_bowl import mod_full_vb_correction


import colorcet as cc

#import sys
#sys.exit(0)

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
            
def calc_mult_mean(epos):
    mult_idxs = np.where(epos['ipp']>1)[0]    
    means = np.zeros_like(epos['tof'])
    for idx in mult_idxs:
        num_ions = epos['ipp'][idx]
        means[idx:idx+num_ions] = np.mean(epos['tof'][idx:idx+num_ions])
    return means
       
def calc_t0(tof,tof_vcorr_fac,tof_bcorr_fac,sigma):

#    A = tof_vcorr_fac[0::2]
#    
#    B_alpha = tof_bcorr_fac[0::2]
#    B_beta = tof_bcorr_fac[1::2]
#    
#    t_alpha = tof[0::2]
#    t_beta = tof[1::2]
#    
#    t0 = (B_alpha*t_alpha+B_beta*t_beta)/(B_alpha+B_beta) -\
#            sigma/(A*(B_alpha+B_beta))
#    
#    t0 = np.repeat(t0,2)
   
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

#
#def recusively_split_multihit(remaining_times, volt_coeff, bowl_coeff, pk_times, time_groupings=None):
#    if time_groupings is None:
#        time_groupings = [];
#        
#    n = remaining_times.size
#    
#
#    
#    if n==1:
#        time_groupings.append(remaining_times)
#    
#    if n<2:
#        return time_groupings
#    
#    THRESH = 2
#    MAX_K = 2 # pk_coords will overflow for k too big!
#    for k in range(min(n,MAX_K),1,-1):
#        pk_coords = cartesian_product([pk_times for i in range(k)])
#        possible_idxs = comb_index(n, k)
#        r0, r1 = calc_parametric_line(remaining_times[possible_idxs],
#                                      volt_coeff[possible_idxs],
#                                      bowl_coeff[possible_idxs],
#                                      n=-1)
#        diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)
#        min_idx = np.unravel_index(np.argmin(diff_mat, axis=None), diff_mat.shape)
#        min_d = diff_mat[min_idx]
#        
#        if min_d<THRESH:
#            grouped_idxs = possible_idxs[min_idx[0]]            
#            ungrouped_idxs = np.setdiff1d(np.arange(n), grouped_idxs)
#           
#            time_groupings.append(remaining_times[grouped_idxs])
#                      
#            return recusively_split_multihit(remaining_times[ungrouped_idxs],
#                                      volt_coeff[ungrouped_idxs],
#                                      bowl_coeff[ungrouped_idxs],
#                                      pk_times,
#                                      time_groupings)
#        else:
#            if k==2:
#                for i in range(n):
#                    time_groupings.append(np.array(remaining_times[i]))
#                return time_groupings
#
#    return
#
#multi_idxs = np.where(epos['ipp']>1)[0]
#
#import time
#start = time.time()
#ct = 0
#for ev_idx in multi_idxs:
#    ct += 1
#    n = epos['ipp'][ev_idx]
#    idxs = slice(ev_idx,ev_idx+n)
#    q = recusively_split_multihit(epos['tof'][idxs], tof_vcorr_fac_all[idxs], tof_bcorr_fac_all[idxs], pk_times)
#    print(str(q)+'\n')
#    if ct>100:
#        break
#end = time.time()
#print(multi_idxs.size*(end - start)/ct)
#
#
#
#
#dts = []
#for ev_idx in multi_idxs:
#    idxs = slice(ev_idx,ev_idx+n)
#    dts.append(np.diff(epos['tof'][idxs]))
#
#dts = np.concatenate(dts)
#
#
#plt.figure(321)
#plt.hist(dts,bins=np.arange(0,1000,0.25))










import GaN_type_peak_assignments



plt.close('all')


fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07094-v03.epos'
epos = apt_fileio.read_epos_numpy(fn)

# Split out data into single and doubles
sing_idxs = np.where(epos['ipp'] == 1)[0]
epos_s = epos[sing_idxs]

tmp_idxs = np.where(epos['ipp'] == 2)[0]
doub_idxs = np.ravel(np.column_stack([tmp_idxs+i for i in range(2)]))

multi_idxs = np.where(epos['ipp'] != 1)[0]

epos_d = epos[doub_idxs]

# voltage and bowl correct ToF data.  
p_volt = np.array([])
p_bowl = np.array([])

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

tof_vcorr_fac_m = tof_vcorr_fac[multi_idxs]
tof_bcorr_fac_m = tof_bcorr_fac[multi_idxs]



# Find transform to m/z space
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(epos_s['m2q'],tof_sing_corr)
epos['m2q'] = m2q_calib.mod_physics_m2q_calibration(p_m2q,mod_full_vb_correction(epos,p_volt,p_bowl))

plotting_stuff.plot_histo(epos['m2q'],fig_idx=1)

import GaN_fun

pk_data = GaN_type_peak_assignments.GaN_with_H()
bg_rois=[[0.4,0.9]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.1, 
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



# Fit out the 2D histogram
pk_params_2d = np.zeros((pk_times.size,pk_times.size), dtype={'names':('t1','t2','sigma','amp','cts'),
                          'formats':('f8','f8','f8','f8','i8')})




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

for pk_idx1 in range(pk_times.size):
    for pk_idx2 in range(pk_times.size):
        # Get ROI of data
        t1g = pk_times[pk_idx1]
        t2g = pk_times[pk_idx2]
        
        pk_params_2d['t1'][pk_idx1,pk_idx2] = t1g
        pk_params_2d['t2'][pk_idx1,pk_idx2] = t2g
                
        win_wid = 3
        
        roi_1 = t1g+[-win_wid/2, win_wid/2]
        roi_2 = t2g+[-win_wid/2, win_wid/2]
        
        roi_idxs1 = np.where((centers>roi_1[0]) & (centers<=roi_1[1]))[0]
        roi_idxs2 = np.where((centers>roi_2[0]) & (centers<=roi_2[1]))[0]
        
        tmp = ch[roi_idxs1[0]:roi_idxs1[-1],roi_idxs2[0]:roi_idxs2[-1]]
        
        pk_params_2d['cts'][pk_idx1,pk_idx2] = tmp.sum()
        
#        if(tmp.sum()>1000):
#            plt.matshow(tmp, fignum=332211, cmap=cc.cm.CET_L8)
#            plt.pause(0.1)


upper_tri_idxs = np.triu_indices_from(pk_params_2d,k=1)
pk_params_2d = pk_params_2d[upper_tri_idxs]

is_nonzero_idxs = np.where(pk_params_2d['cts']>16)[0]

pk_sigma = pk_params_2d['t2'][is_nonzero_idxs]+pk_params_2d['t1'][is_nonzero_idxs]
pk_delta = pk_params_2d['t2'][is_nonzero_idxs]-pk_params_2d['t1'][is_nonzero_idxs]
pk_amp = pk_params_2d['cts'][is_nonzero_idxs]


pk_coords = cartesian_product([pk_times for i in range(2)])



#tofin = epos_d['tof']

THRESH = 2

multi_idx = np.where(epos['ipp']==2)[0]
tofin = np.zeros((np.sum(epos['ipp'][multi_idx]-1), 2))
vcorr_fac_in = np.zeros((np.sum(epos['ipp'][multi_idx]-1), 2))
bcorr_fac_in = np.zeros((np.sum(epos['ipp'][multi_idx]-1), 2))

ct = 0
for idx in multi_idx:
    n = epos['ipp'][idx]    
    tofin[ct:ct+n-1,0] = epos['tof'][idx:idx+n-1]
    tofin[ct:ct+n-1,1] = epos['tof'][idx+1:idx+n]
    vcorr_fac_in[ct:ct+n-1,0] = tof_vcorr_fac[idx:idx+n-1]
    vcorr_fac_in[ct:ct+n-1,1] = tof_vcorr_fac[idx+1:idx+n]
    bcorr_fac_in[ct:ct+n-1,0] = tof_bcorr_fac[idx:idx+n-1]
    bcorr_fac_in[ct:ct+n-1,1] = tof_bcorr_fac[idx+1:idx+n]
    ct += n-1
       
r0,r1 = calc_parametric_line(tofin, vcorr_fac_in, bcorr_fac_in, n=-1)
diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)

min_idxs = np.argmin(diff_mat,axis=1)
min_d = np.min(diff_mat,axis=1)
closest_sums = sigmas[range(sigmas.shape[0]), min_idxs]
closest_sums[min_d>THRESH] = 957

corr_dat = r0+r1*closest_sums[:,np.newaxis]

plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=True)
#
#plotting_stuff.plot_histo(corr_dat[min_d<THRESH,1]-corr_dat[min_d<THRESH,0],fig_idx=332211,user_xlim=[-0,5000], user_bin_width=0.5, clearFigure=True)
#plotting_stuff.plot_histo(corr_dat[min_d>=THRESH,1]-corr_dat[min_d>=THRESH,0],fig_idx=332211,user_xlim=[-0,5000], user_bin_width=0.5, clearFigure=False)


closest_sums[:] = 957
corr_dat = r0+r1*closest_sums[:,np.newaxis]
plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=False)






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

tmp = epos_d['tof']*tof_bcorr_fac_d*tof_vcorr_fac_d

plotting_stuff.plot_histo(tmp[1::2]-tmp[::2],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=False)








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









import sys
sys.exit(0)

plt.figure(23231)
plt.clf()


plt.scatter(pk_coords[:,0], pk_coords[:,1],s = None, color='r')
ri = r0+r1*0
rf = r0+r1*1400

for i in np.arange(2048):
    plt.plot([ri[i,0], rf[i,0]], [ri[i,1], rf[i,1]],'b', alpha=0.1)
    
#    plt.plot(corr_dat[i,0],corr_dat[i,1],'g.')
#    plt.plot(pk_coords[min_idxs[i],0],pk_coords[min_idxs[i],1],'g.')

    plt.plot([corr_dat[i,0],pk_coords[min_idxs[i],0]],[corr_dat[i,1],pk_coords[min_idxs[i],1]],'g', alpha=0.1)
#    plt.pause(1)

plt.gcf().gca().set_aspect('equal', adjustable='box')

plotting_stuff.plot_histo(corr_dat[:,1]-corr_dat[:,0],fig_idx=33221122,user_xlim=[-0,1000], user_bin_width=0.5, clearFigure=False)

plt.figure()
plt.plot(corr_dat[:,0],corr_dat[:,1],',')
plt.scatter(pk_coords[:,0], pk_coords[:,1],s = None, color='r')


tof_vcorr_fac3 = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos_t['tof']),epos_t['v_dc'])
tof_bcorr_fac3 = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos_t['tof']),epos_t['x_det'],epos_t['y_det'])

aa,bb,cc = np.meshgrid(pk_times,pk_times,pk_times)

pk_coords = np.vstack((aa.ravel(),bb.ravel(),cc.ravel())).T

#pk_coords = np.vstack((pk_params_2d['t1'][is_nonzero_idxs],pk_params_2d['t2'][is_nonzero_idxs])).T

r0,r1 = calc_parametric_line(epos_t['tof'], tof_vcorr_fac3, tof_bcorr_fac3, n=3)
diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)


min_idxs = np.argmin(diff_mat,axis=1)
closest_sums = sigmas[range(sigmas.shape[0]), min_idxs]


corr_dat = r0+r1*closest_sums[:,np.newaxis]


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plt.gcf().gca().set_aspect('equal', adjustable='box')

ax.plot(corr_dat[:,1]-corr_dat[:,0],corr_dat[:,2]-corr_dat[:,1],corr_dat[:,2]-corr_dat[:,0],'o', ms=1, alpha = 0.2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

xxx = corr_dat[:,1]-corr_dat[:,0]
yyy = corr_dat[:,2]-corr_dat[:,1]
zzz = corr_dat[:,2]-corr_dat[:,0]

plotting_stuff.plot_histo(np.concatenate((xxx,yyy,zzz)), fig_idx=-1, user_xlim=[0,1000], user_bin_width=0.25)


plt.figure()
plt.plot(xxx,yyy,',')




plotting_stuff.plot_histo(corr_dat[:],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=.5, clearFigure=False)


plotting_stuff.plot_histo(tof_sing_corr,fig_idx=332211,user_xlim=[-0,1000], user_bin_width=.5, clearFigure=False)





tof_vcorr_fac_all = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos['tof']),epos['v_dc'])
tof_bcorr_fac_all = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos['tof']),epos['x_det'],epos['y_det'])
plotting_stuff.plot_histo(epos['tof']*tof_vcorr_fac_all*tof_bcorr_fac_all,fig_idx=332211,user_xlim=[-0,1000], user_bin_width=.5, clearFigure=False)


tof_vcorr_fac4 = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos_4['tof']),epos_4['v_dc'])
tof_bcorr_fac4 = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos_4['tof']),epos_4['x_det'],epos_4['y_det'])

aa,bb,cc,dd = np.meshgrid(pk_times,pk_times,pk_times,pk_times)

pk_coords = np.vstack((aa.ravel(),bb.ravel(),cc.ravel(),dd.ravel())).T

#pk_coords = np.vstack((pk_params_2d['t1'][is_nonzero_idxs],pk_params_2d['t2'][is_nonzero_idxs])).T

r0,r1 = calc_parametric_line(epos_4['tof'], tof_vcorr_fac4, tof_bcorr_fac4, n=4)
diff_mat, sigmas = compute_dist_to_parametric_line(r0, r1, pk_coords)


min_idxs = np.argmin(diff_mat,axis=1)
closest_sums = sigmas[range(sigmas.shape[0]), min_idxs]


corr_dat = r0+r1*closest_sums[:,np.newaxis]

plotting_stuff.plot_histo(corr_dat[:],fig_idx=332211,user_xlim=[-0,1000], user_bin_width=.5, clearFigure=False)






slope, intercept = calc_slope_and_intercept(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac)

diff_mat = compute_dist_to_line(slope[:,None], intercept[:,None], pk_sigma, pk_delta)

sig = 0.25*6
g = lambda x : np.exp(-x**2/(2*sig**2))

probs = pk_amp*g(diff_mat)
probs = probs/np.max(probs)
max_idxs = np.argmax(probs,axis=1)
largest_prob = np.max(probs,axis=1)


closest_sums = pk_sigma[range(pk_sigma.shape[0]), max_idxs]

closest_sums[largest_prob<1e-16] = np.sum(pk_amp*pk_sigma)/np.sum(pk_amp)

#
#
#exs = np.arange(1,128)
#
#cc = [closest_sums[largest_prob<10.0**(-1*ex)].size for ex in exs]



#closest_sums[min_dist>4] = np.random.uniform(0,2000,min_dist.shape)[min_dist>4]
#closest_sums[0,min_dist>4] = np.full(min_dist.shape,550.0)[min_dist>4]

extrapolated_diffs = slope*closest_sums+intercept

#extrapolated_diffs = slope*(np.sum(pk_amp*pk_sigma)/np.sum(pk_amp))+intercept
#plotting_stuff.plot_histo(extrapolated_diffs,fig_idx=9952341, user_xlim=[0,8000],user_bin_width=0.25)
#
#min_idxs = np.argmin(diff_mat,axis=1)
#closest_sums = pk_sigma[min_idxs]
#
#extrapolated_diffs = slope*closest_sums+intercept


#
#vbd = epos_d_vb['tof'][1::2]-epos_d_vb['tof'][0::2]
#vbs = epos_d_vb['tof'][1::2]+epos_d_vb['tof'][0::2]
#
#q_idxs = np.where(vbs<1400)[0]
#
#plotting_stuff.plot_histo(vbd[q_idxs ],fig_idx=9952341, user_xlim=[0,800],user_bin_width=0.25,clearFigure=False)
#
#
#plotting_stuff.plot_histo(extrapolated_diffs[q_idxs ],fig_idx=9952341, user_xlim=[0,800],user_bin_width=0.25,clearFigure=False)
#



#        
## find the voltage and bowl coefficients for the doubles data
#tof_vcorr_fac = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos_d['tof']),epos_d['v_dc'])
#tof_bcorr_fac = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos_d['tof']),epos_d['x_det'],epos_d['y_det'])
#
#
#
#slope, intercept = calc_slope_and_intercept(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac)
#
#intercept = np.random.random(intercept.shape)*1000
#
#diff_mat = compute_dist_to_line(slope[:,None], intercept[:,None], pk_sigma, pk_delta)
#
#sig = 0.25
#g = lambda x : np.exp(-x**2/(2*sig**2))
#
#probs = pk_amp*g(diff_mat)
#probs = probs/np.max(probs)
#max_idxs = np.argmax(probs,axis=1)
#largest_prob = np.max(probs,axis=1)
#
#
min_idxs = np.argmin((diff_mat**2)/1,axis=1)
#
#closest_sums = pk_sigma[max_idxs]
closest_sums = pk_sigma[min_idxs]

#closest_sums[largest_prob<1e-16] = np.sum(pk_amp*pk_sigma)/np.sum(pk_amp)

#
#
#exs = np.arange(1,128)
#
#cc = [closest_sums[largest_prob<10.0**(-1*ex)].size for ex in exs]



#closest_sums[min_dist>4] = np.random.uniform(0,2000,min_dist.shape)[min_dist>4]
#closest_sums[0,min_dist>4] = np.full(min_dist.shape,550.0)[min_dist>4]

#extrapolated_diffs = slope*closest_sums+intercept

#extrapolated_diffs = slope*(np.sum(pk_amp*pk_sigma)/np.sum(pk_amp))+intercept
#plotting_stuff.plot_histo(extrapolated_diffs,fig_idx=9952341, user_xlim=[0,8000],user_bin_width=0.25)
#
#min_idxs = np.argmin(diff_mat,axis=1)
#closest_sums = pk_sigma[min_idxs]
#
#extrapolated_diffs = slope*closest_sums+intercept





plotting_stuff.plot_histo(extrapolated_diffs,fig_idx=9952341, user_xlim=[0,800],user_bin_width=.25,clearFigure=False)



tt = epos_d['tof']*tof_bcorr_fac*tof_vcorr_fac
tt1 = tt[::2] 
tt2 = tt[1::2]
plotting_stuff.plot_histo(tt2-tt1,fig_idx=9952341, user_xlim=[0,800],user_bin_width=.25,clearFigure=False)















t1 = (closest_sums[largest_prob>1e-16]-extrapolated_diffs[largest_prob>1e-16])/2
t2 = (closest_sums[largest_prob>1e-16]+extrapolated_diffs[largest_prob>1e-16])/2

ts = interleave(t1,t2)

plotting_stuff.plot_histo(ts,fig_idx=332211, user_xlim=[0,1000],user_bin_width=.25, clearFigure=False)
#plotting_stuff.plot_histo(epos_d['tof'][sel_idxs]*tof_vcorr_fac[sel_idxs]*tof_bcorr_fac[sel_idxs],fig_idx=11152341, user_xlim=[0,1000],user_bin_width=1, clearFigure=False)
plotting_stuff.plot_histo(tof_sing_corr,fig_idx=11152341, user_xlim=[0,1000],user_bin_width=1, clearFigure=False)






fig = plt.figure(num=321311)
fig.clf()
ax = fig.gca()

ax.scatter(closest_sums,extrapolated_diffs,marker=',', s= 1)











# for the moment we are working with doubles only for simplicity
idxs = np.where(epos['ipp'] == 2)[0]
idxs = sorted(np.concatenate((idxs,idxs+1)))
epos_d = epos[idxs]



m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_singles_corr)



# Verify the alignment worked
m2q_corr = 1e-4*p_m2q[0]*np.square(epos['tof']\
                *voltage_and_bowl.mod_full_voltage_correction(p_volt,
                                                             np.ones_like(epos['tof']),
                                                             epos['v_dc'])\
              *voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,
                                                              np.ones_like(epos['tof']),
                                                              epos['x_det'],
                                                              epos['y_det'])
              -p_m2q[1])

plotting_stuff.plot_histo(ref_epos['m2q'], 33211233, clearFigure=True)
plotting_stuff.plot_histo(m2q_corr, 33211233, clearFigure=False)


def m2q_to_tof(m2q, p_m2q):
    return np.sqrt(1e4*m2q/p_m2q[0])+p_m2q[1]



pk_data = GaN_type_peak_assignments.GaN_with_H()
bg_rois=[[0.4,0.9]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.1, 
        bg_rois=bg_rois)


cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=epos, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=3)

ppd.pretty_print_compositions(compositions,pk_data)



pk_times = m2q_to_tof(pk_params[is_peak.ravel()]['x0_mean_shift'], p_m2q)


# find the voltage and bowl coefficients for the doubles data
tof_vcorr_fac = voltage_and_bowl.mod_full_voltage_correction(p_volt,np.ones_like(epos_d['tof']),epos_d['v_dc'])
tof_bcorr_fac = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,np.ones_like(epos_d['tof']),epos_d['x_det'],epos_d['y_det'])


N_SIGMA_PTS = 100
DEL_SIGMA_PTS = 2000/N_SIGMA_PTS

# create data delta sigma histogram
sigmas = np.arange(N_SIGMA_PTS)*DEL_SIGMA_PTS
res_dat, sigmas, deltas = create_sigma_delta_histogram(epos_d['tof'], 
                                                       tof_vcorr_fac, 
                                                       tof_bcorr_fac,
                                                       sigmas=sigmas,
                                                       delta_range=[0,700],
                                                       delta_step=0.25)

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

FoM = np.sum(res_dat**2, axis=1)
max_idx = np.argmax(FoM)

# create psf sigma delta histogram
psf_sigmas = np.arange(2*N_SIGMA_PTS)*DEL_SIGMA_PTS

sigma0 = np.mean(psf_sigmas)
delta0 = np.mean(deltas)

t1 = 0.5*(sigma0-delta0+np.random.normal(loc=0, scale=0.50, size=epos_d.size//2))/(tof_vcorr_fac[0::2]*tof_bcorr_fac[0::2])
t2 = 0.5*(sigma0+delta0+np.random.normal(loc=0, scale=0.50, size=epos_d.size//2))/(tof_vcorr_fac[0::2]*tof_bcorr_fac[1::2])
psf_tof = interleave(t1,t2)



psf_dat, psf_sigmas, psf_deltas = create_sigma_delta_histogram(psf_tof, 
                                                       tof_vcorr_fac, 
                                                       tof_bcorr_fac,
                                                       sigmas=psf_sigmas,
                                                       delta_range=[-350,700+350],
                                                       delta_step=0.25)

fig = plt.figure(222)
plt.clf()
ax = fig.gca()
ax.imshow(np.log10(1+10*psf_dat.T[:,:]), aspect='auto',
          extent=_extents(psf_sigmas) + _extents(psf_deltas),
          origin='lower', cmap=cc.cm.CET_L8,
          interpolation='nearest')

ax.set_xlabel('$\Sigma_c$ (ns)')
ax.set_ylabel('$\Delta_c$ (ns)')
ax.set_title('histogram (counts)')


# Do RL deconvolution on data using acerage PSF
image = res_dat
image = image/np.max(image)

psf = psf_dat
psf = psf/np.sum(psf)

#im_deconv = RL_deconv(image, psf, iterations=2**8)
im_deconv = restoration.richardson_lucy(image,psf,iterations=2**7,clip=False)

fig = plt.figure(num=333)
plt.clf()
ax = fig.gca()

ax.imshow(np.log10(1+10*im_deconv.T[:,:]), aspect='auto',
          extent=_extents(sigmas) + _extents(deltas),
          origin='lower', cmap=cc.cm.CET_L8,
          interpolation='nearest')

ax.set_xlabel('$\Sigma_c$ (ns)')
ax.set_ylabel('$\Delta_c$ (ns)')
ax.set_title('histogram (counts)')

#
#import photutils
#
##pks = photutils.detection.find_peaks(res_dat/np.max(res_dat), 
#pks = photutils.detection.find_peaks(im_deconv/np.max(im_deconv), 
#                                        threshold=2e-4, 
#                                       box_size=(20,20), 
#                                       footprint=None, 
#                                       mask=None, 
#                                       border_width=2, 
#                                       npeaks=np.inf, 
#                                       centroid_func=None, 
#                                       subpixel=False, 
#                                       error=None, 
#                                       wcs=None)


import skimage.feature
pks = skimage.feature.peak_local_max(im_deconv/np.max(im_deconv),
#pks = skimage.feature.peak_local_max(res_dat/np.max(res_dat),
                               threshold_abs=50e-4, 
#                               threshold_rel=None, 
                               exclude_border=False, 
                               indices=True, 
                               num_peaks=np.inf, 
                               footprint=np.full((20,20), True), 
                               labels=None, 
                               num_peaks_per_label=np.inf)                

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

#ax.scatter(sigmas[pks['y_peak']], deltas[pks['x_peak']] )
#ax.scatter(sigmas[pks[:,0]], deltas[pks[:,1]] )
thresh = 0
ax.scatter(pk_params_2d[pk_params_2d['cts']>thresh]['t1']+pk_params_2d[pk_params_2d['cts']>thresh]['t2'],
           pk_params_2d[pk_params_2d['cts']>thresh]['t2']-pk_params_2d[pk_params_2d['cts']>thresh]['t1'],
           s = pk_params_2d[pk_params_2d['cts']>thresh]['cts']+5,
           color=[1,1,1,0.5])







slope, intercept = calc_slope_and_intercept(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac)

def refine_peak(x_guess, y_guess, slope, intercept):
    
    xc_new, yc_new = x_guess, y_guess
    
    for i in range(99):
#        print((xc_new,yc_new))
        xc_old, yc_old = xc_new, yc_new
        weight = gauss(slope*xc_old+intercept, yc_old, sig)
        xc_new,yc_new = find_lines_center(slope, intercept, weight)
        
        step_size = np.sqrt((xc_new-xc_old)**2+(yc_new-yc_old)**2)
#        print(step_size)
        if (step_size <= 0.01) and i>3:
            print('convergence criteria met in '+str(i)+' steps')
            print((xc_new,yc_new))
            break
    return (xc_new,yc_new)    
    
def gauss(x,x0,sig):
    return np.exp(-(x-x0)**2/(2*sig**2))

sig = 0.5/2.355

pk_sigma_deconv = sigmas[pks[:,0]]
pk_delta_deconv = deltas[pks[:,1]]

pk_sigma_refine = np.zeros_like(pk_sigma_deconv)
pk_delta_refine = np.zeros_like(pk_delta_deconv)

for i in range(pk_delta_deconv.size):
    pk_sigma_refine[i], pk_delta_refine[i] =  refine_peak(pk_sigma_deconv[i], pk_delta_deconv[i], slope, intercept)

ax.scatter(pk_sigma_refine, pk_delta_refine )





diff_mat = compute_dist_to_line(slope[:,None], intercept[:,None], pk_sigma_refine, pk_delta_refine)

min_idxs = np.argmin(diff_mat,axis=1)

min_dist = np.min(diff_mat,axis=1)


closest_sums = pk_sigma_refine[min_idxs]
#closest_sums[min_dist>4] = np.random.uniform(0,2000,min_dist.shape)[min_dist>4]
#closest_sums[0,min_dist>4] = np.full(min_dist.shape,550.0)[min_dist>4]


extrapolated_diffs = slope*closest_sums+intercept

plotting_stuff.plot_histo(extrapolated_diffs,fig_idx=9952341, user_xlim=[0,5000],user_bin_width=0.5)





t1 = (closest_sums-extrapolated_diffs)/2
t2 = (closest_sums+extrapolated_diffs)/2

ts = interleave(t1,t2)

plotting_stuff.plot_histo(ts,fig_idx=11152341, user_xlim=[0,1000],user_bin_width=1)
#plotting_stuff.plot_histo(epos_d['tof'][sel_idxs]*tof_vcorr_fac[sel_idxs]*tof_bcorr_fac[sel_idxs],fig_idx=11152341, user_xlim=[0,1000],user_bin_width=1, clearFigure=False)
plotting_stuff.plot_histo(epos_d['tof'][other_idxs]*tof_vcorr_fac[other_idxs]*tof_bcorr_fac[other_idxs],fig_idx=11152341, user_xlim=[0,1000],user_bin_width=1, clearFigure=False)





fig = plt.figure(777)
plt.clf()
ax = fig.gca()
ax.plot(0.5*(pk_sigma_refine-pk_delta_refine),0.5*(pk_sigma_refine+pk_delta_refine),'o')
ax.plot(0.5*(pk_sigma_refine+pk_delta_refine),0.5*(pk_sigma_refine-pk_delta_refine),'o')
ax.plot(0.5*(pk_sigma_refine-pk_delta_refine),0.5*(pk_sigma_refine-pk_delta_refine))
#ax.plot(deltas_pts,sigmas_pts,'o')        
ax.set_aspect('equal', 'box')

ax.set_xlabel('$t1$')
ax.set_ylabel('$t2$')
ax.set_title('Interpolating ...')







def recip_decay( x, x0, a, gam, off):
    return a/np.sqrt((x-x0)**2/gam**2+1)+off
    
def res_recip_decay(params, xData, yData):
    return recip_decay(xData,*params[0:4])-yData

from scipy.optimize import leastsq

new_pks = pks.copy()
idx = -1
for d_idx,s_idx in zip(pks[:,1],pks[:,0]):
    xdat = sigmas
    win = 2
    ydat = np.mean(res_dat[:,d_idx-win:d_idx+win+1],axis=1)
    
    idx +=1
    fig = plt.figure(3213211)
    plt.clf()
    ax = fig.gca()
    ax.plot(xdat,ydat)
    ax.scatter(xdat[s_idx],ydat[s_idx])
    
    p_0 = [ xdat[s_idx], ydat[s_idx]-np.min(ydat), 150, np.min(ydat)]
    
    lb = np.max([xdat[0], xdat[s_idx]-250])
    ub = np.min([xdat[s_idx]+250, xdat[-1]])
    
    idxs = np.where((xdat>=lb) & (xdat<=ub))
    
    
    p_opt, ier = leastsq(res_recip_decay, p_0, args=(xdat[idxs], ydat[idxs]))
#    plt.plot(xdat,recip_decay(sigmas,*p_0))
    plt.plot(xdat[idxs],recip_decay(xdat[idxs],*p_opt))

    

#    new_pks[idx,0] = np.argmax(recip_decay(xdat,*p_opt))
    new_pks[idx,0] = p_opt[0]

    ax.scatter(p_opt[0],ydat[np.argmax(recip_decay(xdat,*p_opt))])

    plt.pause(3)


fig = plt.figure(num=111)
ax = fig.gca()
ax.scatter(new_pks[:,0], deltas[new_pks[:,1]] )



deltas_pts = deltas[new_pks[:,1]]
sigmas_pts = new_pks[:,0]

##
#deltas_pts = deltas[pks[:,1]]
#sigmas_pts = sigmas[pks[:,0]]

sort_idxs = np.argsort(deltas_pts )


deltas_pts = deltas_pts[sort_idxs]
sigmas_pts = sigmas_pts[sort_idxs]


from scipy.interpolate import interp1d 

g = interp1d(deltas_pts,sigmas_pts,fill_value="extrapolate",kind='nearest')


tof_corr_no_mod = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof'])

t0 = calc_t0(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac, sigmas[max_idx])
tof_corr_single = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof']-t0)+t0

sigmas_hat = g(np.abs(tof_corr_single[1::2]-tof_corr_single[0::2]))
t0 = calc_t0(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac, sigmas_hat)
tof_corr = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof']-t0)


epos_vb_no_mod = epos_d.copy()
epos_vb_no_mod['tof'] = tof_corr_no_mod.copy()

epos_vb_mod_single = epos_d.copy()
epos_vb_mod_single['tof'] = tof_corr_single.copy()

epos_vb_mod = epos_d.copy()
epos_vb_mod['tof'] = tof_corr.copy()


#fig = plt.figure(555)
#plt.clf()
#ax = fig.gca()
#x = epos_vb_mod_single['tof'][1::2]-epos_vb_mod_single['tof'][0::2]
#x = np.repeat(x,2)
#
#ax.plot(x,tof_bcorr_fac,'.')
#
#
#
#ax.plot()








idxs = np.where(epos_vb_mod['ipp'] == 2)[0]
mean_t = 0.5*(epos_d['tof'][idxs]+epos_d['tof'][idxs+1])
idxs = idxs[mean_t>000]
    
dts_mod = np.abs(epos_vb_mod['tof'][idxs]-epos_vb_mod['tof'][idxs+1])
dts_mod_single = np.abs(epos_vb_mod_single['tof'][idxs]-epos_vb_mod_single['tof'][idxs+1])
dts_no_mod = np.abs(epos_vb_no_mod['tof'][idxs]-epos_vb_no_mod['tof'][idxs+1])


lims = [0,1000]
bin_width = 0.5
fig = plt.figure(333)
plt.clf()
ax = fig.gca()

N, E = np.histogram(dts_no_mod, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Normal V+B", lw=2)

N, E = np.histogram(dts_mod_single, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Single $\Sigma$ V+B", lw=2)

N, E = np.histogram(dts_mod, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Piecewise $\Sigma$ V+B", lw=2)
ax.legend()


ax.set_yscale('log')
ax.set_xlabel('$\Delta_c$')
ax.set_ylabel('counts')

ax.set_xlim(0, 700)
ax.set_ylim(1, 1e4)


q = np.linspace(0,1000,2**11)

fig = plt.figure(444)
plt.clf()
ax = fig.gca()
ax.plot(q,g(q))
ax.plot(deltas_pts,sigmas_pts,'o')        
ax.set_xlabel('$\Delta_c$')
ax.set_ylabel('$\Sigma_c$')
ax.set_title('Interpolating $\Sigma_c$ from $\Delta_c$')


fig = plt.figure(777)
plt.clf()
ax = fig.gca()
ax.plot(0.5*(sigmas_pts-deltas_pts),0.5*(sigmas_pts+deltas_pts),'o')
ax.plot(0.5*(sigmas_pts+deltas_pts),0.5*(sigmas_pts-deltas_pts),'o')
ax.plot(0.5*(sigmas_pts-deltas_pts),0.5*(sigmas_pts-deltas_pts))
#ax.plot(deltas_pts,sigmas_pts,'o')        
ax.set_aspect('equal', 'box')

ax.set_xlabel('$t1$')
ax.set_ylabel('$t2$')
ax.set_title('Interpolating ...')



sigs_mod = g(dts_mod)

t1 = (sigs_mod-dts_mod)/2
t2 = (sigs_mod+dts_mod)/2

ts = interleave(t1,t2)

plotting_stuff.plot_histo(ts,fig_idx=52341, user_xlim=[0,1000],user_bin_width=0.5)
plotting_stuff.plot_histo(epos_d['tof']*tof_vcorr_fac*tof_bcorr_fac,fig_idx=52341, user_xlim=[0,1000],user_bin_width=0.5, clearFigure=False)





epos_vb_full_mod = epos_d.copy()
epos_vb_full_mod['tof'] = ts.copy()

edges, ch = corrhist(epos_vb_full_mod)
fig2 = plt.figure(num=2)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges)
ax2.axis('equal')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')


epos_vb_no_mod = epos_d.copy()
epos_vb_no_mod['tof'] = epos_d['tof']*tof_bcorr_fac*tof_vcorr_fac

edges, ch = corrhist(epos_vb_no_mod,roi = [0, 5000], delta=2)
fig2 = plt.figure(num=3)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges)
ax2.axis('equal')
ax2.axis('square')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')

#ax2.set_xlim(300,600)
#ax2.set_ylim(300,600)
#ax2.set_xlim(380,520)
#ax2.set_ylim(380,520)


a,b = np.meshgrid(pk_times,pk_times)

ass,bss = np.meshgrid(pk_params['amp'][is_peak.ravel()]*pk_params['std_fit'][is_peak.ravel()],pk_params['amp'][is_peak.ravel()]*pk_params['std_fit'][is_peak.ravel()])

w = .05*((ass*bss).ravel())

idxs = np.where(w>np.median(w))

ax2.scatter(a.ravel()[idxs],b.ravel()[idxs], s = w[idxs], alpha=0.4)

diffs = pk_times-pk_times[:,None]
diffs = np.abs(np.tril(diffs,0))
#diffs = diffs[diffs!=0]

sums = pk_times+pk_times[:,None]
sums = np.abs(np.tril(sums,0))
#sums = sums[sums!=0]


idxs = np.where((w>np.median(w)) & (sums.ravel()>1) & (diffs.ravel()>1))
idxs = np.where((sums.ravel()>1))





ax = plt.figure(111).gca()

ax.scatter(sums.ravel()[idxs],diffs.ravel()[idxs], s = 100*w[idxs], alpha=0.5)


#ax.scatter(sums, diffs )



tmp_idxs = np.where((epos_d['tof'][0::2]>1000) & (epos_d['tof'][1::2]>1000))[0]
sel_idxs = interleave(2*tmp_idxs,2*tmp_idxs+1)


tmp_idxs = np.where((epos_d['tof'][0::2]<1000) | (epos_d['tof'][1::2]<1000))[0]
other_idxs = interleave(2*tmp_idxs,2*tmp_idxs+1)

#sel_idxs = np.where(mean_t>2000)

slope, intercept = calc_slope_and_intercept(epos_d['tof'][sel_idxs], tof_vcorr_fac[sel_idxs], tof_bcorr_fac[sel_idxs])

plt.figure(332211), plt.clf(), plt.hist(slope*725+intercept,bins=4*1024,range=[0,1000])


idxs = np.where((w>np.median(w)) & (sums.ravel()>10) & (diffs.ravel()>0.1))

ref_sums = sums.ravel()[idxs][None,:]
ref_diffs = diffs.ravel()[idxs][None,:]




diff_mat = compute_dist_to_line(slope[:,None], intercept[:,None], ref_sums, ref_diffs)

min_idxs = np.argmin(diff_mat,axis=1)

min_dist = np.min(diff_mat,axis=1)


closest_sums = ref_sums[:,min_idxs]
closest_sums[0,min_dist>4] = np.random.uniform(0,1000,min_dist.shape)[min_dist>4]
#closest_sums[0,min_dist>4] = np.full(min_dist.shape,550.0)[min_dist>4]


extrapolated_diffs = slope*closest_sums+intercept

plotting_stuff.plot_histo(extrapolated_diffs,fig_idx=9952341, user_xlim=[0,1000],user_bin_width=0.1)







t1 = (closest_sums-extrapolated_diffs)/2
t2 = (closest_sums+extrapolated_diffs)/2

ts = interleave(t1,t2)

plotting_stuff.plot_histo(ts,fig_idx=11152341, user_xlim=[0,1000],user_bin_width=0.2)
plotting_stuff.plot_histo(epos_d['tof'][sel_idxs]*tof_vcorr_fac[sel_idxs]*tof_bcorr_fac[sel_idxs],fig_idx=11152341, user_xlim=[0,1000],user_bin_width=0.2, clearFigure=False)
plotting_stuff.plot_histo(epos_d['tof'][other_idxs]*tof_vcorr_fac[other_idxs]*tof_bcorr_fac[other_idxs],fig_idx=11152341, user_xlim=[0,1000],user_bin_width=0.2, clearFigure=False)






epos_total_fudge = epos_d.copy()
epos_total_fudge = epos_total_fudge[sel_idxs]

epos_total_fudge['tof'] = ts.copy()

edges, ch = corrhist(epos_total_fudge)
fig2 = plt.figure(num=22)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges)
ax2.axis('equal')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')







m2q_corr = 1e-4*p_m2q[0]*np.square(epos_total_fudge['tof'] - p_m2q[1])

epos_total_fudge['m2q'] = m2q_corr

cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=epos_total_fudge, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=3)

ppd.pretty_print_compositions(compositions,pk_data)

plotting_stuff.plot_histo(epos_total_fudge['m2q'],2245411)
plotting_stuff.plot_histo(epos_vb_mod_single['m2q'],2245411, clearFigure=False)




import sys 














import sys 
sys.exit()
    
#FoM = np.sum(res_dat**2, axis=1)
#max_idx = np.argmax(FoM)
##    fig = plt.figure(111)
##    plt.clf()
##    ax = fig.gca()
#ax2.plot(sigmas, FoM, '-')
#ax2.plot(sigmas[max_idx], FoM[max_idx], 'o')
#ax2.text(1.05*sigmas[max_idx], FoM[max_idx], '$\Sigma_{max}$ ='+str(sigmas[max_idx]))
#ax2.set_xlabel('$\Sigma_c$ (ns)')
#ax2.set_ylabel('Peakyness FoM')


sigma_max = sigmas[max_ind[0]]
delta_max = deltas[max_ind[1]]

t0 = calc_t0(epos_d_trimmed['tof'],tof_vcorr_fac_trimmed,tof_bcorr_fac_trimmed,sigma_max)
tof_corr = tof_vcorr_fac_trimmed*tof_bcorr_fac_trimmed*(epos_d_trimmed['tof']-t0)
dts = np.abs(tof_corr[:-1:2]-tof_corr[1::2])

#    fig = plt.figure(333)
#    plt.clf()
#    ax = fig.gca()
N, edges = np.histogram(dts,bins=np.linspace(0,700,1401))

ax3.plot(0.5*(edges[1:]+edges[:-1]),N)
ax3.set_yscale('log')
ax3.scatter(deltas[max_ind[1]],N[max_ind[1]])


    
    
    






#
#FoM = np.sum(res_dat**2, axis=1)
#max_idx = np.argmax(FoM)
#
#fig = plt.figure(111)
#plt.clf()
#ax = fig.gca()
#ax.plot(sigmas, FoM, '-')
#ax.plot(sigmas[max_idx], FoM[max_idx], 'o')
#ax.text(1.05*sigmas[max_idx], FoM[max_idx], '$\Sigma_{max}$ ='+str(sigmas[max_idx]))
#ax.set_xlabel('$\Sigma_c$ (ns)')
#ax.set_ylabel('Peakyness FoM')















from  scipy.interpolate import interp2d
f = interp2d(sigmas, centers, res_dat.T, fill_value=0, kind='cubic')

def find_loc_max(p0,f):
    
#    opts = {'maxiter' : None,    # default value.
#        'disp' : True,    # non-default value.
#        'gtol' : 1e-5,    # default value.
#        
#        }  # default value.
#    opts = {'return_all': True,
#            'disp' : True,
#            'eps' : np.array([50.0,1.0]),
##            'gtol' : 1e-3,
#}    # non-default value
    
    f_neg = lambda p : -1*f(p[0],p[1])
    from scipy import optimize
    opts = {
            'disp' : True,
            'direc' : ([50,0],[0,1])  
            }
    res = optimize.minimize(f_neg, p0, method='Powell', options=opts)
    return res
    
    


start_shift = sigmas[max_idx]

pts = []

for dt in centers:
    res = find_loc_max(np.array([start_shift,dt]), f)
    res.x
    
    if res.fun<-100: 
#        ax.scatter(res.x[0],res.x[1],10,'w',alpha=0.5)
        pts.append(res.x)
#        ax.arrow(start_shift,dt,res.x[0]-start_shift,res.x[1]-dt,head_width=5, head_length=5,color='w',alpha=0.25)
    
    
    
    
pts_arr = np.zeros((len(pts),2))
for idx in np.arange(len(pts)):
    pts_arr[idx,:] = pts[idx]

import sklearn.cluster as cluster

scale_fac = 10
pts_arr[:,0] /= scale_fac 
labels = cluster.MeanShift(bandwidth=2, cluster_all=False).fit_predict(pts)
pts_arr[:,0] *= scale_fac 

cm = cc.cm.glasbey    

from collections import Counter, defaultdict


control_pts = []
tot_ct = 0
num_in_clust = Counter(labels)
for idx in np.arange(len(pts)):
    if num_in_clust[labels[idx]]>2:
        tot_ct +=1
#        ax.scatter(pts_arr[idx,0],pts_arr[idx,1],20,color=cm(labels[idx]), alpha=0.25)
        control_pts.append(pts_arr[idx,:])



def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


fused_pts = fuse(control_pts, 1)

fused_pts_arr = np.zeros((len(fused_pts),2))
for idx in np.arange(len(fused_pts)):
    fused_pts_arr[idx,:] = fused_pts[idx][:]
    ax.scatter(fused_pts[idx][0],fused_pts[idx][1],100,color=cm(idx), alpha=1)
        
fused_pts_arr

from scipy.interpolate import interp1d 

g = interp1d(fused_pts_arr[:,1],fused_pts_arr[:,0],fill_value="extrapolate",kind='nearest')



tof_corr_no_mod = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof'])

t0 = calc_t0(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac, sigmas[max_idx])
tof_corr_single = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof']-t0)+t0

sigmas_hat = g(np.abs(tof_corr_single[1::2]-tof_corr_single[0::2]))
t0 = calc_t0(epos_d['tof'], tof_vcorr_fac, tof_bcorr_fac, sigmas_hat)
tof_corr = tof_vcorr_fac*tof_bcorr_fac*(epos_d['tof']-t0)


epos_vb_no_mod = epos_d.copy()
epos_vb_no_mod['tof'] = tof_corr_no_mod.copy()

epos_vb_mod_single = epos_d.copy()
epos_vb_mod_single['tof'] = tof_corr_single.copy()

epos_vb_mod = epos_d.copy()
epos_vb_mod['tof'] = tof_corr.copy()


#fig = plt.figure(555)
#plt.clf()
#ax = fig.gca()
#x = epos_vb_mod_single['tof'][1::2]-epos_vb_mod_single['tof'][0::2]
#x = np.repeat(x,2)
#
#ax.plot(x,tof_bcorr_fac,'.')
#
#
#
#ax.plot()








idxs = np.where(epos_vb_mod['ipp'] == 2)[0]
mean_t = 0.5*(epos_d['tof'][idxs]+epos_d['tof'][idxs+1])
idxs = idxs[mean_t>000]
    
dts_mod = np.abs(epos_vb_mod['tof'][idxs]-epos_vb_mod['tof'][idxs+1])
dts_mod_single = np.abs(epos_vb_mod_single['tof'][idxs]-epos_vb_mod_single['tof'][idxs+1])
dts_no_mod = np.abs(epos_vb_no_mod['tof'][idxs]-epos_vb_no_mod['tof'][idxs+1])


lims = [0,1000]
bin_width = 0.5
fig = plt.figure(333)
plt.clf()
ax = fig.gca()

N, E = np.histogram(dts_no_mod, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Normal V+B", lw=2)

N, E = np.histogram(dts_mod_single, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Single $\Sigma$ V+B", lw=2)

N, E = np.histogram(dts_mod, bins=2001, range=[0,1000])
plt.plot((E[:-1]+E[1:])/2, N, label="Piecewise $\Sigma$ V+B", lw=2)
ax.legend()


ax.set_yscale('log')
ax.set_xlabel('$\Delta_c$')
ax.set_ylabel('counts')

ax.set_xlim(0, 700)
ax.set_ylim(1, 1e4)


q = np.linspace(0,1000,2**11)

fig = plt.figure(444)
plt.clf()
ax = fig.gca()
ax.plot(q,g(q))
ax.plot(fused_pts_arr[:,1],fused_pts_arr[:,0],'o')        
ax.set_xlabel('$\Delta_c$')
ax.set_ylabel('$\Sigma_c$')
ax.set_title('Interpolating $\Sigma_c$ from $\Delta_c$')



#    

import sys 
sys.exit()














def interleave(a,b):
    return np.ravel(np.column_stack((a,b)))

sigma0 = 1000
delta0 = 350

t1 = 0.5*(sigma0-delta0)/(tof_vcorr_fac[0::2]*tof_bcorr_fac[0::2])+np.random.normal(loc=0, scale=0.50, size=epos_d.size//2)
t2 = 0.5*(sigma0+delta0)/(tof_vcorr_fac[0::2]*tof_bcorr_fac[1::2])+np.random.normal(loc=0, scale=0.50, size=epos_d.size//2)

tof = interleave(t1,t2)

mean_t = (t1+t2)/2


sigmas = np.linspace(-1000,3000,2**8)
#sigmas = np.linspace(000,2000,2**7)

idxs = np.arange(int(epos_d.size/2))[mean_t>000]

for sigma_idx in np.arange(sigmas.size):
    
    t0 = calc_t0(tof, tof_vcorr_fac, tof_bcorr_fac, sigmas[sigma_idx])
    tof_corr = tof_vcorr_fac*tof_bcorr_fac*(tof-t0)

    dts_mod = np.abs(tof_corr[2*idxs]-tof_corr[2*idxs+1])

    centers,bin_dat = plotting_stuff.bin_dat(dts_mod,
                                             bin_width=0.5,
                                             user_roi=[0,700],
                                             isBinAligned=True)
    
    if sigma_idx==0:
        psf_dat = np.zeros((sigmas.size,bin_dat.size))
                
    psf_dat[sigma_idx,:] = bin_dat
    
    print("Loop index "+str(sigma_idx+1)+" of "+str(sigmas.size))

FoM = np.sum(psf_dat**2, axis=1)
max_idx = np.argmax(FoM)



fig = plt.figure(222)
plt.clf()
ax = fig.gca()
ax.imshow(np.log10(1+psf_dat.T[:,:]), aspect='auto',
          extent=_extents(sigmas) + _extents(centers),
          origin='lower', cmap=cc.cm.CET_L8,
          interpolation='nearest')

ax.set_xlabel('$\Sigma_c$ (ns)')
ax.set_ylabel('$\Delta_c$ (ns)')
ax.set_title('histogram (counts)')













sigma0 = 1000
delta0 = 350

sigmas = np.linspace(0,2000,2**12)



plot_it(res_dat,346)



fig = plt.figure(1111)
plt.clf()
ax = fig.gca()

max_idxs = np.argmax(res_dat, axis=0)
maxes = res_dat[max_idxs,np.arange(res_dat.shape[1])]
ax.plot(maxes/np.max(maxes))

max_idxs = np.argmax(im_deconv, axis=0)
maxes = im_deconv[max_idxs,np.arange(res_dat.shape[1])]
ax.plot(15*maxes/np.max(maxes))

ax.set_yscale('log')

for i in range(res_dat.shape[1]):
    fig = plt.figure(1337)
    plt.clf()
    ax = fig.gca()
    ax.plot(res_dat[:,i].flatten())
    plt.pause(1)
    
    
    
    
prof = np.mean(im_deconv, axis=0)
plt.figure()
plt.plot(centers,prof/np.max(prof))
    


plt.figure()

prof = np.max(im_deconv, axis=0)
plt.plot(centers,prof/np.max(prof))

prof = np.max(image, axis=0)
plt.plot(centers,prof/np.max(prof))




    
plt.ylim(1e-4,1)


plt.figure()

max_idxs = np.argmax(image, axis=0)
plot_it(im_deconv,666)
plot_it(image,667)

sz = np.array([image[a,b] for a,b in zip(max_idxs, np.arange(image.shape[1]))])
plt.gcf().gca().scatter(max_idxs, np.arange(image.shape[1]), 200*sz**2, color='g')
#plt.gcf().gca().plot(max_idxs, np.arange(image.shape[1]), color='g')









fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.linspace(0, 2000, 2**7)
Y = np.linspace(0.5,700,1400)
X, Y = np.meshgrid(X,Y, indexing='ij')

# Plot the surface.
surf = ax.plot_surface(X,Y, im_deconv, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)




fig = plt.figure(3232)
plt.clf()
ax = fig.gca()

q = np.arange(psf.shape[1])
q = q-np.mean(q)
for i in np.arange(-128,128,16):
    plt.plot(q/(np.abs(i)+0),(np.abs(i)+0)*psf[128+i,:])
    
    ax.set_xlim(-5,5)
    
    plt.pause(1)
    
plt.figure(41244)
plt.clf()


ydat = np.max(psf[:,699:701],axis=1)
ydat /= np.max(ydat)
xdat = 1.0*np.arange(ydat.size)
xdat -= np.mean(xdat)

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)
def res_lorentzian(params, xData, yData):
    return lorentzian(xData,*params[0:4])-yData

def recip_decay( x, x0, a, gam ):
    return a/np.sqrt((x-x0)**2/gam**2+1)
    
def res_recip_decay(params, xData, yData):
    return recip_decay(xData,*params[0:4])-yData

from scipy.optimize import leastsq

startValues = [ 0, 1, 1]
#popt, ier = leastsq( res_lorentzian, startValues, args=( xdat, ydat ) )
popt, ier = leastsq( res_recip_decay, startValues, args=( xdat, ydat ) )

plt.plot(xdat, ydat)
plt.plot(xdat,recip_decay(xdat,*popt))



def testA(x):
    x = x + 1
    return x

def testB(x):
    y = x + 1
    return y

def testC(x):
    x += 1
    return x


xq = np.zeros(3)
testA(xq)
print(xq)
testB(xq)
print(xq)
testC(xq)
print(xq)




def test(x):
    x += 1
    return x

xq = np.zeros(3)
print(xq)
test(xq)
print(xq)







ttt = epos_d['tof']*tof_bcorr_fac*tof_vcorr_fac

regs = [500,1000,2500,4000]

t1 = ttt[0::2]
t2 = ttt[1::2]

s = t1+t2

fig = plt.figure(num=334455)
plt.clf()
ax = fig.gca()

for i,reg in enumerate(regs):
    
    
    idxs = np.where(np.abs(s-2*reg)<400)[0]
    
    print(idxs.size)
    
    
    N,e = np.histogram(t2[idxs]-t1[idxs],bins=np.linspace(0,650,650+1))
    
    ax.plot(0.5*(e[:-1]+e[1:]), (N+1)*100**i)
    
    ax.set_yscale('log')
    
    ax.set_xlim(0, 750)
    ax.set_ylim(1, 1e8)
    
    
    
    
    
    plotting_stuff.plot_histo(t2[idxs]-t1[idxs],
                              fig_idx=223311,
                              clearFigure=(i==0),
                              user_xlim=[0,700],
                              user_bin_width=1)

    
    
idxs = np.where(np.abs(s-2*3000)<4000)[0]

print(idxs.size)


plotting_stuff.plot_histo(t2[idxs]-t1[idxs],
                          fig_idx=223311,
                          clearFigure=False,
                          user_xlim=[0,700],
                          user_bin_width=1)




    
idxs = np.where(np.abs(s-2*500)<100)[0]

print(idxs.size)


plotting_stuff.plot_histo(t2[idxs]-t1[idxs],
                          fig_idx=223311,
                          clearFigure=False,
                          user_xlim=[0,700],
                          user_bin_width=1)














