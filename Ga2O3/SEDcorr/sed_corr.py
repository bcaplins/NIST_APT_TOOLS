# -*- coding: utf-8 -*-
"""
This module contains the functions necessary to correct atom probe datasets
for systematic energy deficits.  Details on the algorithm are given in a
manuscript titled:
    ``An Algorithm for Correcting Systematic Energy Deficits in the Atom Probe
    Mass Spectra of Insulating Samples''

Created on Mon Nov 25 14:02:12 2019

@author: bwc

>  NIST Public License - 2019

>  This software was developed by employees of the National Institute of
>  Standards and Technology (NIST), an agency of the Federal Government
>  and is being made available as a public service. Pursuant to title 17
>  United States Code Section 105, works of NIST employees are not subject
>  to copyright protection in the United States.  This software may be
>  subject to foreign copyright.  Permission in the United States and in
>  foreign countries, to the extent that NIST may hold copyright, to use,
>  copy, modify, create derivative works, and distribute this software and
>  its documentation without fee is hereby granted on a non-exclusive basis,
>  provided that this notice and disclaimer of warranty appears in all copies.

>  THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND,
>  EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
>  TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
>  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
>  AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION
>  WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE
>  ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING,
>  BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES,
>  ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE,
>  WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER
>  OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND
>  WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF,
>  OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

"""


import numpy as np
import histo_funcs as hf
from scipy.interpolate import interp1d


def get_shifts(ref, N, max_shift=150, global_shift_fun=None):
    """
    This computes the optimal shift between a 1d reference array and the
    columns of a 2d data array using an fft based cross correlation.  The
    optimal shift is assumed to be where the cross correlation is maximal.
    Because no padding or normalization is applied, this works best for
    relatively small shifts.  This is explicitly enforced by the ``max_shift``
    parameter.

    Parameters
    ----------
    ref : real numeric 1d array (shape is nx1)
        The reference array.  Should be in a nx1 column array.
    N : real numeric 2d array (shape is nxm)
        The data array.  Each column of this array will be cross-correlated to
        the reference array using an fft.
    max_shift : int (default is 150)
        The maximum number of bins that are searched for the cross-correlation.
        The cross correlation is set to zero for lags less than -max_shift and
        greater than +max_shift.
    global_shift_fun : function handle (default is None)
        This function applies an additional subtractive shift to the returned
        shift array.  I generally set this to numpy.mean or numpy.median, but
        more complex shift functions could be used (including lambdas).

    Returns
    -------
    shifts : numeric 1d array
        The shift (in units of bin index) that was required to align each
        column of data to the reference spectrum.

    """
    # Note: Use real valued fft to improve speed/memory

    # FFT the ref and data arrays
    rfft_ref = np.fft.rfft(ref, axis=1)
    rfft_N = np.fft.rfft(N, axis=1)

    # Compute the cross-correlation and take the inverse fft
    xc = np.fft.irfft(rfft_N*np.conj(rfft_ref), axis=1)

    # Set the cross correlation to zero for lags greater than max_shift
    xc[:, max_shift:xc.shape[1]-max_shift] = 0

    # Find the lags corresponding to the maximum of the cross correlation and
    # then shift them to correspond to the appropriate origin.
    max_idxs = np.argmax(xc, axis=1)
    max_idxs[max_idxs > xc.shape[1]//2] = \
        max_idxs[max_idxs > xc.shape[1]//2] - xc.shape[1]

    # Apply a global_shift_fun shift if specified
    if global_shift_fun is not None:
        shifts = max_idxs - global_shift_fun(max_idxs)
    else:
        shifts = max_idxs

    return shifts


def get_all_scale_coeffs(event_dat,
                         max_scale=1.1,
                         roi=None,
                         cts_per_chunk=2**10,
                         delta_logdat=5e-4):
    """
    This attempts to best align event data.  The basic assumption is that if
    the ``event_dat`` is binned into a 2d history (histogram) then there are
    clearly defined features (i.e. peaks) that will shift around in a
    systematic manner.  When this data is projected onto a single dimension
    then any features (peaks) will be broader than they really should be.  This
    algorithm discretizes the event_dat into `chunks' and attempts to align
    each chunk to a reference dataset using a scalar multiplicative
    coefficient.  The reference dataset is the middle 50% of the ``event_dat``.
    The alignment is performed using a logarithm based cross correlation
    approach.  Two iterations of this algorithm are performed before the result
    is returned.  In all tests performed thus far a single iteration was
    sufficient however we used two iterations in an abundance of caution.

    Parameters
    ----------
    event_dat : real float 1d array
        Event data.  Typically the data is either the mass-to-charge or
        time-of-flight of each event.  The ordering of the data is assumed to
        be chronological (i.e. the order in which they were detected).
    max_scale : real float (default is 1.1)
        The maximum possible scale factor allowed (relative to the reference
        data)
    roi : real numeric list or array (default is [0.5,200])
        The domain that the data should be evaluated over.
        Specified as [min, max] values.
    cts_per_chunk : int (default is 1024)
        The number of events to be collected into a single `chunk' of data.
    delta_logdat : real float (default is 5e-4)
        The discretization of the log(data) over the roi specified.  Smaller
        deltas are more time/memory intensive.  For deltas much less than one,
        this effectively gives a discretization/resolution of the
        multiplicative factor of 1+delta_logdat.  For the atom probe data I
        have worked with, the noise on the shift is on the order of 1e-3 and
        so setting the delta to be smaller than this, ensures that the
        discretization error is not a significant problem.

    Returns
    -------
    eventwise_scales : real float array
        An array that contains the computed scale factor for each event that
        best aligns the data.  To correct the data, just divide the event_dat
        array by the eventwise_scales array.

    """
    if roi is None:
        roi = [0.5, 200]
    log_roi = np.log(roi)

    # Take the log of data
    logdat = np.log(event_dat)

    # Create the histogram.  Compute centers and delta y
    N, seq_edges, logdat_edges = \
        hf.create_histogram(logdat,
                            roi=log_roi,
                            cts_per_chunk=cts_per_chunk,
                            delta_dat=delta_logdat)
    seq_centers, logdat_centers = hf.edges_to_centers(seq_edges, logdat_edges)
#    print('specified delta_logdat = '+str(delta_logdat))
    delta_logdat = logdat_edges[1]-logdat_edges[0]
#    print('actual delta_logdat = '+str(delta_logdat))

    # Initialize the total eventwise log(dat) shift
    eventwise_logdat_shifts = np.zeros(event_dat.size)

    # Do one iteration with the center 50% of the data as a reference
    # Note: Make it is 2d (even though it is just a single column array)
    ref = np.mean(N[N.shape[0]//4:3*N.shape[0]//4, :], axis=0)[None, :]

    # Get the maximum possible shift in bins.
    max_pixel_shift = int(np.ceil(np.log(max_scale)/delta_logdat))

    # Determine the chunkwise shifts
    chunkwise_shifts0 = delta_logdat*get_shifts(ref,
                                                N,
                                                max_shift=max_pixel_shift,
                                                global_shift_fun=np.mean)

    # Interpolate (linear) from chunkwise to eventwise shifts
    f = interp1d(seq_centers, chunkwise_shifts0, fill_value='extrapolate')

    # Accumulate the shift for the first iteration.
    eventwise_logdat_shifts += f(np.arange(event_dat.size))

    # Correct the log(data)
    logdat_corr = logdat - eventwise_logdat_shifts

    # Recompute the histogram with newly corrected log(data)
    N, seq_edges, logdat_edges = \
        hf.create_histogram(logdat_corr,
                            roi=log_roi,
                            cts_per_chunk=cts_per_chunk,
                            delta_dat=delta_logdat)
    seq_centers, logdat_centers = hf.edges_to_centers(seq_edges, logdat_edges)
    delta_logdat = logdat_edges[1]-logdat_edges[0]

    # Use the center 50% of the data as a reference
    # Note: Make it is 2d (even though it is just a single column array)
    ref = np.mean(N[N.shape[0]//4:3*N.shape[0]//4, :], axis=0)[None, :]

    # Get the maximum possible shift in bins.
    max_pixel_shift = int(np.ceil(np.log(max_scale)/delta_logdat))

    # Determine the chunkwise shifts
    chunkwise_shifts1 = delta_logdat*get_shifts(ref,
                                                N,
                                                max_shift=max_pixel_shift,
                                                global_shift_fun=np.mean)

    # Interpolate to get eventwise shifts
    f = interp1d(seq_centers, chunkwise_shifts1, fill_value='extrapolate')

    # Accumulate the shift for the second iteration.
    eventwise_logdat_shifts += f(np.arange(event_dat.size))

    # Compute total eventwise shifts for output
    eventwise_scales = np.exp(eventwise_logdat_shifts)

#    # Uncomment this to see the relative importance of the two iterations
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(np.exp(chunkwise_shifts0), label='iter 0')
#    plt.plot(np.exp(chunkwise_shifts1), label='iter 1')
#    plt.legend()

    return eventwise_scales
