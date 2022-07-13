# -*- coding: utf-8 -*-
"""
This module contains functions for creating 2d histgrams from atom probe
event-based datasets.  The resulting histgrams can be used for plotting or
calculation purposes.  Note that the histogram data structures may become
memory intensive.

Created on Wed Aug 28 13:41:03 2019

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


def create_histogram(dat, roi, delta_dat, cts_per_chunk):
    """
    Create a 2d histogram from event data.  Note that the ``cts_per_chunk`` and
    ``delta_dat`` parameters are suggestions, not exact numbers.  Most notably,
    the number of bins is rounded to the nearest power of 2 for fft speedup,
    and so the real ``delta_dat`` may be somewhat different than the requested
    one. At present, this function creates the full 2d histogram and may be
    quite memory intensive.  For very large datasets it may be best/necessary
    to process the data in batches.

    Parameters
    ----------
    dat : real float array
        The event data.
    roi : real float list/array
        Domain the data should be binned over.
    delta_dat : real float
        Discretization for the data bins.
    cts_per_chunk : int
        Number of events per chunk.

    Returns
    -------
    N : real 2d array
        Number of events for each chunk/bin.
    seq_edges : real numeric 1d array
        Edges for the chunk axis.
    dat_edges : real numeric 1d array
        Edges for the data axis.
    """

    # Compute the number of bins. Round to nearest power of 2 for fft speed.
#    num_bins = int(np.abs(np.diff(roi))/delta_dat)
#    num_bins = int(np.ceil(np.abs(np.diff(roi))/delta_dat/2)*2)
    num_bins = int(2**np.round(np.log2(np.abs(np.diff(roi))/delta_dat)))
    num_chunks = int(dat.size/cts_per_chunk)

    # If the number of bins is greater than the maximum 32 bit integer value
    # then numpy may get pretty grumpy.  Note that the histogram array may
    # occupy on the order of 8 GB of memory before this assertion fails.
    assert (num_bins*num_chunks) < (2**31-1), \
        "Numpy histogram2d won't like this..."

    # Print these numbers since it can be useful to see
    print('[num_chunks, num_bins] = [%d, %d]' % (num_chunks, num_bins))

    # Leverage histogram2d's evenly spaced bin speedup by letting histogram2d
    # create the edges.
    N, seq_edges, dat_edges = np.histogram2d(np.arange(dat.size),
                                             dat,
                                             bins=[num_chunks, num_bins],
                                             range=[[1, dat.size], list(roi)],
                                             density=False)

    # Print these numbers since it can be useful to see
    print('[requested delta, actual delta] = [%.2e, %.2e]'
          % (delta_dat, dat_edges[1]-dat_edges[0]))

    return N, seq_edges, dat_edges


def edges_to_centers(*edges):
    """
    Convert bin edges to bin centers

    Parameters
    ----------
    *edges : bin edges

    Returns
    -------
    centers : list of bin centers

    """
    centers = []
    for es in edges:
        centers.append((es[0:-1]+es[1:])/2)
    return centers
