# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:58:24 2020

@author: capli
"""

import colorcet as cc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(xs,ys,x_roi=None,y_roi=None,num_x=128,num_y=128):
    N,x_edges,y_edges = np.histogram2d(xs,ys,bins=[num_x,num_y],range=[x_roi,y_roi],density=False)
    return (N, x_edges, y_edges)


# standard imports 



def create_det_hit_plots_SI(epos, pk_data, pk_params, fig_idx=200):
        
    def create_axes(ax, sel_idxs,title):

        NNN = 64
        N,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs],
                                             x_roi=[-35,35],y_roi=[-35,35],
                                             num_x=NNN, num_y=NNN)
    
        ax.imshow(np.transpose(N), aspect='auto', 
                   extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
                   interpolation='nearest')
        ax.set_aspect('equal', 'box')
        
        ax.set(xlabel='$x_{detector}$ (mm)')
        ax.set(ylabel='$y_{detector}$ (mm)')
        ax.set_title(title)
        
        return
    
    
    def create_axes_v2(ax, sel_idxs1,sel_idxs2,title):

        NNN = 64
        N1,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs1],epos['y_det'][sel_idxs1],
                                             x_roi=[-35,35],y_roi=[-35,35],
                                             num_x=NNN, num_y=NNN)
        N2,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs2],epos['y_det'][sel_idxs2],
                                             x_roi=[-35,35],y_roi=[-35,35],
                                             num_x=NNN, num_y=NNN)
        
        
    
        ax.imshow(np.transpose(np.log10(N2/N1)), aspect='auto', 
                   extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
                   interpolation='nearest')
        ax.set_aspect('equal', 'box')
        
        ax.set(xlabel='$x_{detector}$ (mm)')
        ax.set(ylabel='$y_{detector}$ (mm)')
        ax.set_title(title)
        
        return
    
    
       
    
    fig = plt.figure(num=fig_idx)
    fig.clear()    
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, num=fig_idx)
    axes = axes.flatten()
    
    ax_idx = 0
    
    # Get all events
    m2q_roi = [6, 120]
    sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
    create_axes(axes[ax_idx],sel_idxs,'ion histogram m/z $\in$'+m2q_roi.__str__())
    ax_idx += 1
    
    # CSR
    HW = 0.3        
    Ga1p = np.array([68.92, 70.92])
    Ga2p = Ga1p/2
        
    sel_idxs1 = np.where(((epos['m2q']>(Ga1p[0]-HW)) & (epos['m2q']<(Ga1p[0]+HW))) \
                         | ((epos['m2q']>(Ga1p[1]-HW)) & (epos['m2q']<(Ga1p[1]+HW))))
    sel_idxs2 = np.where(((epos['m2q']>(Ga2p[0]-HW)) & (epos['m2q']<(Ga2p[0]+HW))) \
                         | ((epos['m2q']>(Ga2p[1]-HW)) & (epos['m2q']<(Ga2p[1]+HW))))
    
    
    create_axes_v2(axes[ax_idx],sel_idxs1,sel_idxs2,'CSR')
    


    return None            


def create_det_hit_plots(epos, pk_data, pk_params, fig_idx=200):
        
    def create_axes(ax, sel_idxs,title):

    
        N,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs],
                                             x_roi=[-35,35],y_roi=[-35,35])
    
        ax.imshow(np.transpose(N), aspect='auto', 
                   extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
                   interpolation='nearest')
        ax.set_aspect('equal', 'box')
        
        ax.set(xlabel='det_x')
        ax.set(ylabel='det_y')
        ax.set_title(title)
        
        return
    
        
    keys = list(pk_data.dtype.fields.keys())
    keys.remove('m2q')
    
    
    fig = plt.figure(num=fig_idx)
    fig.clear()    
    fig, axes = plt.subplots(nrows=1, ncols=len(keys)+3, sharex=True, sharey=True, num=fig_idx)
    axes = axes.flatten()
    
    ax_idx = 0
    
    # Get all events
    m2q_roi = [6, 120]
    sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
    create_axes(axes[ax_idx],sel_idxs,'roi='+m2q_roi.__str__())
    ax_idx += 1
    
    # Get hydrogen events
    m2q_roi = [0.9, 1.2]
    sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
    create_axes(axes[ax_idx],sel_idxs,'roi='+m2q_roi.__str__())
    ax_idx += 1

    # Get bg events
    m2q_roi = [120, 200]
    sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
    create_axes(axes[ax_idx],sel_idxs,'roi='+m2q_roi.__str__())
    ax_idx += 1
    
    for k in keys:
        k_pks = np.where(pk_data[k]>0)[0]    
        sel_idxs = np.zeros(0,dtype='int64')
        for pk in k_pks:
            ev_idxs = np.where((epos['m2q']>pk_params['pre_rng'][pk]) & (epos['m2q']<pk_params['post_rng'][pk]))[0]
            sel_idxs = np.concatenate((sel_idxs,ev_idxs))
        
        create_axes(axes[ax_idx], sel_idxs, k)
        ax_idx +=1
        
    return None            


def create_csr_2d_plots(epos, pk_params, Ga1p_idxs, Ga2p_idxs, fig_idx=500):
    

    def get_events(pk_idxs):        
        is_part = (epos['m2q']<0)        
        for pk_idx in pk_idxs:
            is_part = is_part | ((epos['m2q'] >= pk_params['pre_rng'][pk_idx]) & (epos['m2q'] <= pk_params['post_rng'][pk_idx]))
                
        return epos[is_part]
        
    
    Ga1p_sub_epos = get_events(Ga1p_idxs)
    Ga2p_sub_epos = get_events(Ga2p_idxs)

    num_disc = 32

    N1,x_edges,y_edges = create_histogram(Ga1p_sub_epos['x_det'],Ga1p_sub_epos['y_det'],
                                         x_roi=[-35,35],y_roi=[-35,35],
                                         num_x=num_disc, num_y=num_disc)

    N2,x_edges,y_edges = create_histogram(Ga2p_sub_epos['x_det'],Ga2p_sub_epos['y_det'],
                                         x_roi=[-35,35],y_roi=[-35,35],
                                         num_x=num_disc, num_y=num_disc)

    from scipy.ndimage import gaussian_filter

    EPS = 1e-6
    CSR = (gaussian_filter(N2, sigma=1)/gaussian_filter(N1+EPS, sigma=1))
    
    bad_dat = ~((N2>0) & (N1>0))
    
    CSR[bad_dat] = CSR[~bad_dat].min()
    
    

    fig = plt.figure(num=fig_idx)
    fig.clear()  
    ax = fig.gca()
    ax.imshow(np.transpose(CSR), aspect='auto', 
                   extent=extents(x_edges) + extents(y_edges), origin='lower',
                   vmin=np.percentile(CSR.flatten(),1), vmax=np.percentile(CSR.flatten(),99),
                   cmap=cc.cm.CET_L8,
                   interpolation='nearest')
    ax.set_aspect('equal', 'box')
    
    ax.set(xlabel='det_x')
    ax.set(ylabel='det_y')
    ax.set_title('CSR')

    

    return fig         

def mean_shift(xs,ys):
    radius = 5
    
    x_curr = 0
    y_curr = 0
    
    N_LOOPS = 64
    
    xi = np.zeros(N_LOOPS)
    yi = np.zeros(N_LOOPS)
    
    for i in np.arange(N_LOOPS):
        x_prev = x_curr
        y_prev = y_curr
        
        idxs = np.where(((xs-x_curr)**2+(ys-y_curr)**2) <= radius**2)
#        print(idxs)
        
        x_q = np.mean(xs[idxs])
        y_q = np.mean(ys[idxs])
           
        dx = x_q-x_prev
        dy = y_q-y_prev
        
        x_curr = x_prev-dx
        y_curr = y_prev-dy
        
        if np.sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2) < (radius*1e-2):
#            print('iter  B #',i,'    ',x_curr,y_curr)
#            print(i)
            break
#        else:
#            print('iter NB #',i,'    ',x_curr,y_curr)
#        print('iter #',i,'    ',x_curr,y_curr)
        xi[i] = x_curr
        yi[i] = y_curr

    return (xi[:i], yi[:i])




def chop_data_rad_and_time(epos,c_pt,time_chunk_size=2**16,N_ann_chunks=3):
    
    es2cs = lambda es : (es[:-1]+es[1:])/2.0
    
    N_time_chunks = int(np.floor(epos.size/time_chunk_size))  
    time_chunk_size = epos.size//N_time_chunks
    time_chunk_edges = np.arange(N_time_chunks+1)*time_chunk_size
    time_chunk_centers = es2cs(time_chunk_edges)
    
    R_DET_MAX = 28
    R_C = np.sqrt(c_pt[0]**2+c_pt[1]**2)
    R_MAX = R_DET_MAX-R_C
    
    r_edges = np.sqrt(np.linspace(0, R_MAX**2, N_ann_chunks+1))
    r_centers = es2cs(r_edges)
    
    rows, cols = (N_time_chunks, N_ann_chunks) 
    idxs_list = [[0 for i in range(cols)] for j in range(rows)]

    r = np.sqrt(np.square(epos['x_det']-c_pt[0])+np.square(epos['y_det']-c_pt[1]))
    
    for t_idx in np.arange(N_time_chunks):
        for a_idx in np.arange(N_ann_chunks):
            idxs = np.where((r>r_edges[a_idx]) & (r<=r_edges[a_idx+1]))[0]
            
            idxs_list[t_idx][a_idx] = np.intersect1d(idxs,np.arange(time_chunk_edges[t_idx],time_chunk_edges[t_idx+1]))
            
    return (time_chunk_centers, r_centers, idxs_list)
    
    

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



# Fit plane to QW and rotate data to 'flatten' wrt z-axis
def qw_plane_fit(x,y,z,p_guess,fwhm=3):

    import scipy.linalg
    
    # three parameters alpha*x+beta*y+gamma
    p_old = -1
    p_new = p_guess
    mod_fun = lambda p: p[0]*x+p[1]*y+p[2]

    for i in np.arange(64):
        p_old = p_new
        resid = np.abs(z-mod_fun(p_new))
        is_near = np.where(resid<(fwhm/2.0))[0]
        
        xq, yq, zq = (x[is_near], y[is_near], z[is_near])
            
        A = np.c_[xq,yq,np.ones(xq.shape)]
        p_new, _, _, _ = scipy.linalg.lstsq(A, zq)
        print(p_new)
        if np.sum(np.square(p_new-p_old)) < 1e-12:
            print('break early. idx: ',i)
            break
    return p_new

def rotate_data_flat(p,x,y,z):
    # Takes the equation for a plane and rotates the point cloud in a manner
    # such that the plane (if rotated the same way, is in the x-y plane)
    
    N = -np.array([p[0],p[1],-1])
    N = N/np.sqrt(np.sum(N**2))    
    F = np.array([0,0,1])    
    ang_to_z_axis = -np.arccos(np.sum(F*N))
    
    
    theta_z = np.arctan2(p[0],p[1])
    c, s = (np.cos(theta_z), np.sin(theta_z))
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])   
    P = np.c_[x,y,z].T
    P = Rz@P
        
    c, s = (np.cos(ang_to_z_axis), np.sin(ang_to_z_axis))
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    P = R@P

    iRz = np.linalg.inv(Rz)
    
    P = iRz@P
    
    return (P[0,:], P[1,:], P[2,:])

    
    

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    
    
    

import plotting_stuff

def load_epos(run_number, epos_trim, fig_idx=-1):
    import GaN_data_paths
    import apt_fileio
    
    fn = GaN_data_paths.get_epos_path(run_number=run_number)
    fn = fn.rsplit(sep='.', maxsplit=1)[0]+'_vbm_corr.epos'
    epos = apt_fileio.read_epos_numpy(fn)

    # Plot m2q vs event index and show the current ROI selection
    roi_event_idxs = np.arange(epos_trim[0],epos.size-epos_trim[1])
    
    
    if fig_idx>0:
        ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,fig_idx=fig_idx)
        ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
        ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
        ax.set_title('roi selected to start analysis')
    
    
    sub_epos = epos[roi_event_idxs]
    
    print('ROI includes the following % of events: ',sub_epos.size/epos.size)
       
    return sub_epos
    
    
    
    
def fit_spectrum(epos, pk_data, peak_height_fraction, bg_rois):
    import initElements_P3
    import peak_param_determination as ppd

    # Define peaks to range
    ed = initElements_P3.initElements()
    
    # Define which peaks to use for CSR calcs
    Ga1p_m2qs = [ed['Ga'].isotopes[69][0], ed['Ga'].isotopes[71][0]]
    Ga2p_m2qs = [ed['Ga'].isotopes[69][0]/2, ed['Ga'].isotopes[71][0]/2]

    Ga1p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga1p_m2qs]
    Ga2p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga2p_m2qs]

    # Determine the global background    
    glob_bg_param = ppd.get_glob_bg(epos['m2q'],rois=bg_rois)

    # Range the peaks
    pk_params = ppd.get_peak_ranges(epos,
                                    pk_data['m2q'],
                                    peak_height_fraction=peak_height_fraction,
                                    glob_bg_param=0)    
    
    return (pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs)

    
def count_and_get_compositions(epos, pk_data, pk_params, glob_bg_param, bg_frac=1, noise_threshhold=2):
    import peak_param_determination as ppd

    
    # Count the peaks, local bg, and global bg
    cts = ppd.do_counting(epos,pk_params,glob_bg_param)
    cts['local_bg'] = cts['local_bg']*bg_frac
    cts['global_bg'] = cts['global_bg']*bg_frac

    # Test for peak S/N and throw out craptastic peaks
    B = np.max(np.c_[cts['local_bg'][:,None],cts['global_bg'][:,None]],1)[:,None]
    T = cts['total'][:,None]
    S = T-B
    std_S = np.sqrt(T+B)
    # Make up a threshold for peak detection... for the most part this won't matter
    # since weak peaks don't contribute to stoichiometry much... except for Mg!
    is_peak = S>(noise_threshhold*np.sqrt(2*B))
    for idx, ct in enumerate(cts):
        if not is_peak[idx]:
            for i in np.arange(len(ct)):
                ct[i] = 0
            
    # Calculate compositions
    compositions = ppd.do_composition(pk_data,cts)

    return (cts, compositions, is_peak)
    

def bin_and_smooth_spectrum(epos, user_roi, bin_wid_mDa=30, smooth_wid_mDa=-1):
    from histogram_functions import bin_dat
    import peak_param_determination as ppd

    xs, ys = bin_dat(epos['m2q'],user_roi=user_roi,isBinAligned=True,bin_width=bin_wid_mDa/1000.0)
    if smooth_wid_mDa>0:
        ys_sm = ppd.moving_average(ys,smooth_wid)*smooth_wid
    else:
        ys_sm = ys
    return (xs, ys_sm)

    
def plot_spectrum_gory_detail(epos, user_roi, pk_params, bg_rois, glob_bg_param, is_peak, fig_idx):
    from histogram_functions import bin_dat
    import peak_param_determination as ppd

    xs, ys = bin_dat(epos['m2q'],user_roi=user_roi,isBinAligned=True)
    #ys_sm = ppd.do_smooth_with_gaussian(ys,30)
    ys_sm = ppd.moving_average(ys,30)

    glob_bg = ppd.physics_bg(xs,glob_bg_param)    

    fig = plt.figure(num=fig_idx)
    fig.clear()
    ax = fig.gca()

    ax.plot(xs,ys_sm,label='hist')
    ax.plot(xs,glob_bg,label='global bg')

    ax.set(xlabel='m/z (Da)', ylabel='counts')
    ax.grid()
    fig.tight_layout()
    fig.canvas.manager.window.raise_()
    ax.set_yscale('log')    
    ax.legend()

    for idx,pk_param in enumerate(pk_params):
        
        if is_peak[idx]:
            ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
            ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
            ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
            ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
            
            ax.plot(np.array([pk_param['pre_bg_rng'],pk_param['post_bg_rng']]) ,np.ones(2)*pk_param['loc_bg'],'g--')
        else:
            ax.plot(np.array([1,1])*pk_param['x0_mean_shift'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
            
    for roi in bg_rois:
        xbox = np.array([roi[0],roi[0],roi[1],roi[1]])
        ybox = np.array([0.1,np.max(ys_sm)/10,np.max(ys_sm)/10,0.1])
        
        ax.fill(xbox,ybox, 'b', alpha=0.2)

            
    plt.pause(0.1)


    return None
    
    
    
    
    
    
    
    
    
    
    
    
    