
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
              interpolation='bicubic')


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

def get_tof_to_m2q_coeffs(m2qs, tofs):
    rat = np.sqrt(m2qs[0]/m2qs[1])
        
    p1 = (tofs[1]*rat-tofs[0])/(rat-1)
    p0 = m2qs[0]/np.square(tofs[0]-p1)
    
    return np.array([p0*1e4, p1])
    




plt.close('all')


ref_fn = r'.\R44_03711-v01_ref.epos'

#fn = r'C:\Users\bwc\Documents\NetBeansProjects\cal with 2kV-9kV_\recons\recon-v01\default\cal with 2kV-9kV_R44_03727-v01.epos'
fn = r'C:\Users\bwc\Documents\NetBeansProjects\R44_03705\recons\recon-v01\default\R44_03705-v01.epos'
#fn = r'C:\Users\bwc\Documents\NetBeansProjects\R44_03707\recons\recon-v01\default\R44_03707-v01.epos'
#fn = r'C:\Users\bwc\Documents\NetBeansProjects\R44_03711\recons\recon-v01\default\R44_03711-v01.epos'

fn = r'C:\Users\bwc\Documents\NetBeansProjects\R20_28285\recons\recon-v01\default\R20_28285-v01.epos'



ref_epos = apt_fileio.read_epos_numpy(ref_fn)

epos = apt_fileio.read_epos_numpy(fn)

# voltage and bowl correct ToF data.  
p_volt = np.array([])
p_bowl = np.array([ 0.89964083, -0.43114144, -0.27484715, -0.25883824])


_, p_volt, p_bowl = do_voltage_and_bowl(epos[epos['ipp']==1],p_volt,p_bowl)   

#tofs = mod_full_vb_correction(epos,p_volt,p_bowl)
#plotting_stuff.plot_histo(tofs, fig_idx=1,user_label='',clearFigure=True,scale_factor=1/epos.size, user_xlim=[0, 1000], user_bin_width=0.25)
#p0_m2q = get_tof_to_m2q_coeffs(m2qs=[1, 70.92], tofs=[65.7, 626.86])
#m2q_corr = m2q_calib.mod_physics_m2q_calibration(p0_m2q, tofs)

# Find transform to m/z space
m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],mod_full_vb_correction(epos,p_volt,p_bowl))


sc = 1 

plotting_stuff.plot_histo(epos['m2q'],fig_idx=1,user_label='(ivas)',clearFigure=True,scale_factor=1/epos.size, user_xlim=[0, 150], user_bin_width=0.03)
plotting_stuff.plot_histo(m2q_corr/sc,fig_idx=1,user_label='(python)',clearFigure=False,scale_factor=1/epos.size*1e3, user_xlim=[0, 150], user_bin_width=0.03)


plotting_stuff.plot_bowl_slices(epos['m2q'], epos, 3,user_ylim=[0, 110])
plotting_stuff.plot_bowl_slices(m2q_corr, epos, 3,user_ylim=[0, 110])

#
#epos_vb = epos.copy()
#epos_vb['tof'] = np.sqrt(1e4*epos['m2q'])

#
#epos_vb = epos_vb[idxs2]
#
#edges, ch = corrhist(epos_vb,roi = [00, 600], delta=.1)
#centers = (edges[1:]+edges[:-1])/2.0
#
#fig2 = plt.figure(num=3)
#fig2.clf()
#ax2 = fig2.gca()   
#plot_2d_histo(ax2, ch, edges, edges, scale='log')
#ax2.axis('equal')
#ax2.axis('square')
#ax2.set_xlabel('ns')
#ax2.set_ylabel('ns')
#ax2.set_title('Si')






#
#idxs1 = np.where(epos['ipp']==1)[0]
#
#idxs2 = np.where(epos['ipp']==2)[0]
#idxs2 = np.sort(np.r_[idxs2,idxs2+1])
#
#idxs3 = np.where(epos['ipp']==3)[0]
#idxs3 = np.sort(np.r_[idxs3,idxs3+1,idxs3+2])
#
#plotting_stuff.plot_histo(np.sqrt(epos['m2q'][idxs1]*1e4), 111, clearFigure=True, user_xlim=[0, 1000], user_bin_width=0.1)
#plotting_stuff.plot_histo(np.sqrt(epos['m2q'][idxs2]*1e4), 111, clearFigure=False, user_xlim=[0, 1000], user_bin_width=0.1)
#plotting_stuff.plot_histo(np.sqrt(epos['m2q'][idxs3]*1e4), 111, clearFigure=False, user_xlim=[0, 1000], user_bin_width=0.1)
#





import sys
sys.exit(0)








# Make a correlation histogram of doubles
epos_vb_R44 = epos_R44.copy()
epos_vb_R44['tof'] = np.sqrt(1e4*epos_R44['m2q'])

edges, ch = corrhist(epos_vb_R44,roi = [0, 1000], delta=1)
centers = (edges[1:]+edges[:-1])/2.0

fig2 = plt.figure(num=3)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges, scale='log')
ax2.axis('equal')
ax2.axis('square')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')
ax2.set_title('R44')

epos_vb_R20 = epos_R20.copy()
epos_vb_R20['tof'] = np.sqrt(1e4*epos_R20['m2q'])

edges, ch = corrhist(epos_vb_R20,roi = [0, 1000], delta=1)
centers = (edges[1:]+edges[:-1])/2.0

fig2 = plt.figure(num=4)
fig2.clf()
ax2 = fig2.gca()   
plot_2d_histo(ax2, ch, edges, edges, scale='log')
ax2.axis('equal')
ax2.axis('square')
ax2.set_xlabel('ns')
ax2.set_ylabel('ns')
ax2.set_title('R20')


plotting_stuff.plot_histo(epos_vb_R44['tof'],fig_idx=5,user_label='R44',scale_factor=1/epos_R44.size,user_xlim=[0, 1000],user_bin_width=.1)
plotting_stuff.plot_histo(epos_vb_R20['tof'],fig_idx=5,user_label='R20',clearFigure=False,scale_factor=1/epos_R20.size,user_xlim=[0, 1000],user_bin_width=0.1)


plotting_stuff.plot_histo(epos_vb_R44['tof'][epos_vb_R44['ipp']==1],fig_idx=7,user_label='R44 s',scale_factor=1/epos_R44.size,user_xlim=[0, 1000],user_bin_width=.1)
plotting_stuff.plot_histo(epos_vb_R20['tof'][epos_vb_R20['ipp']==1],fig_idx=7,user_label='R20 s',clearFigure=False,scale_factor=1/epos_R20.size,user_xlim=[0, 1000],user_bin_width=0.1)

plotting_stuff.plot_histo(epos_vb_R44['tof'][epos_vb_R44['ipp']!=1],fig_idx=9,user_label='R44 m',scale_factor=1/epos_R44.size,user_xlim=[0, 1000],user_bin_width=.1)
plotting_stuff.plot_histo(epos_vb_R20['tof'][epos_vb_R20['ipp']!=1],fig_idx=9,user_label='R20 m',clearFigure=False,scale_factor=1/epos_R20.size,user_xlim=[0, 1000],user_bin_width=0.1)


plotting_stuff.plot_histo(epos_vb_R20['tof'][epos_vb_R20['ipp']==1],fig_idx=101,user_label='R20 s',clearFigure=True,scale_factor=1/epos_R20.size,user_xlim=[0, 1000],user_bin_width=0.1)
plotting_stuff.plot_histo(epos_vb_R20['tof'][epos_vb_R20['ipp']!=1],fig_idx=101,user_label='R20 m',clearFigure=False,scale_factor=1/epos_R20.size,user_xlim=[0, 1000],user_bin_width=0.1)

plotting_stuff.plot_histo(epos_vb_R44['tof'][epos_vb_R44['ipp']==1],fig_idx=103,user_label='R44 s',clearFigure=True,scale_factor=1/epos_R44.size,user_xlim=[0, 1000],user_bin_width=.1)
plotting_stuff.plot_histo(epos_vb_R44['tof'][epos_vb_R44['ipp']!=1],fig_idx=103,user_label='R44 m',clearFigure=False,scale_factor=1/epos_R44.size,user_xlim=[0, 1000],user_bin_width=.1)




lhs_roi = [489.3, 491]
rhs_roi = [491, 492.7]


#lhs_roi = [345.7, 346.5]
#rhs_roi = [346.5, 348]

lhs_idxs = np.where((epos_vb_R44['tof']>lhs_roi[0]) & (epos_vb_R44['tof']<lhs_roi[1]))[0]
rhs_idxs = np.where((epos_vb_R44['tof']>rhs_roi[0]) & (epos_vb_R44['tof']<rhs_roi[1]))[0]


fig = plt.figure(num=400)
fig.clf()
ax = fig.gca()   

ax.scatter(epos_vb_R44['x_det'][lhs_idxs],epos_vb_R44['y_det'][lhs_idxs],alpha=0.1,s=3)
ax.axis('equal')
ax.axis('square')
ax.set_xlabel('x')
ax.set_ylabel('y')


fig = plt.figure(num=401)
fig.clf()
ax = fig.gca()   

ax.scatter(epos_vb_R44['x_det'][rhs_idxs],epos_vb_R44['y_det'][rhs_idxs],alpha=0.1,s=3)
ax.axis('equal')
ax.axis('square')
ax.set_xlabel('x')
ax.set_ylabel('y')



plotting_stuff.plot_bowl_slices(
    epos_vb_R44['tof'],
    epos_vb_R44,
    1000,
    clearFigure=True,
    user_ylim=[0, 1000],
)


















lhs_roi = [489., 490.2]
rhs_roi = [490.2, 491]



lhs_idxs = np.where((epos_vb_R20['tof']>lhs_roi[0]) & (epos_vb_R20['tof']<lhs_roi[1]))[0]
rhs_idxs = np.where((epos_vb_R20['tof']>rhs_roi[0]) & (epos_vb_R20['tof']<rhs_roi[1]))[0]


fig = plt.figure(num=500)
fig.clf()
ax = fig.gca()   

ax.scatter(epos_vb_R20['x_det'][lhs_idxs],epos_vb_R20['y_det'][lhs_idxs],alpha=0.1,s=3)
ax.axis('equal')
ax.axis('square')
ax.set_xlabel('x')
ax.set_ylabel('y')


fig = plt.figure(num=501)
fig.clf()
ax = fig.gca()   

ax.scatter(epos_vb_R20['x_det'][rhs_idxs],epos_vb_R20['y_det'][rhs_idxs],alpha=0.1,s=3)
ax.axis('equal')
ax.axis('square')
ax.set_xlabel('x')
ax.set_ylabel('y')



plotting_stuff.plot_bowl_slices(
    epos_vb_R20['tof'],
    epos_vb_R20,
    1000,
    clearFigure=True,
    user_ylim=[0, 1000],
)



import sys
sys.exit(0)













