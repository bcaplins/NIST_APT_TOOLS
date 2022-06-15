# -*- coding: utf-8 -*-
"""
This module contains functions to read and write atom probe tomography data
files saved in the .pos and .epos formats.  The datafiles are read into
structured arrays.  The .pos and .epos file types are commonly created using
the IVAS software package.  Information on the .pos and .epos file formats
available in Appendix A of 'Atom Probe tomography: A Users Guide'.

Created on Mon Nov 25 13:31:37 2019

@author: bwc

Numpy versions of functions by https://github.com/MHC03
https://github.com/MHC03/apt-tools/blob/master/apt_importers.py

Original functions by https://github.com/oscarbranson
https://github.com/oscarbranson/apt-tools/blob/master/apt_importers.py

The `read' functions are basically unchanged from the version created by MHC03.
The write functions were written by bwc.

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


def read_pos_numpy(file_path):
    """
    Loads an APT .pos file as structured array

    Parameters
    ----------
    file_path : string
        A path to the .pos file

    Returns
    -------
    pos_arr : structured array
        Columns:
            x: Reconstructed x position
            y: Reconstructed y position
            z: Reconstructed z position
            m2q: mass/charge ratio of ion

    See Appendix A of 'Atom Probe tomography: A Users Guide' for more
    information on the .epos format.
    """

    if not file_path.lower().endswith('.pos'):
        raise ValueError("File Path does not end with .pos.")
    with open(file_path, 'rb') as f:
        # >f4 is big endian 4 byte float
        pos_d_type = np.dtype({'names': ['x', 'y', 'z', 'm2q'],
                               'formats': ['>f4', '>f4', '>f4', '>f4']})
        # returns a numpy array, where you can, for example, access all 'x' by
        # pos_arr['x'] and so on
        pos_arr = np.fromfile(f, pos_d_type, -1)
    return pos_arr


def read_epos_numpy(file_path):
    """
    Loads an APT .epos file as structured array

    Parameters
    ----------
    file_path : string
        A path to the .epos file

    Returns
    -------
    pos_arr : structured array
        Columns:
            x: Reconstructed x position
            y: Reconstructed y position
            z: Reconstructed z position
            m2q: Mass/charge ratio of ion
            tof: Ion Time Of Flight
            v_dc: Potential
            v_pulse: Size of voltage pulse (voltage pulsing mode only)
            x_det: Detector x position
            y_det: Detector y position
            pslep: Pulses since last event pulse (i.e. ionisation rate)
            ipp: Ions per pulse (multihits)

    When more than one ion is recorded for a given pulse, only the
    first event will have an entry in the "Pulses since last evenT
    pulse" column. Each subsequent event for that pulse will have
    an entry of zero because no additional pulser firings occurred
    before that event was recorded. Likewise, the "Ions Per Pulse"
    column will contain the total number of recorded ion events for
    a given pulse. This is normally one, but for a sequence of records
    a pulse with multiply recorded ions, the first ion record will
    have the total number of ions measured in that pulse, while the
    remaining records for that pulse will have 0 for the Ions Per
    Pulse value.

    See Appendix A of 'Atom Probe tomography: A Users Guide' for more
    information on the .epos format.
    """

    if not file_path.lower().endswith('.epos'):
        raise ValueError("File Path does not end with .epos.")
    with open(file_path, 'rb') as f:
        # >f4 is big endian 4 byte float
        epos_d_type = np.dtype({'names': ['x', 'y', 'z',
                                          'm2q', 'tof',
                                          'v_dc', 'v_pulse',
                                          'x_det', 'y_det',
                                          'pslep', 'ipp'],
                                'formats': ['>f4', '>f4', '>f4',
                                            '>f4', '>f4',
                                            '>f4', '>f4',
                                            '>f4', '>f4',
                                            '>i4', '>i4']})

        # returns a numpy array, where you can, for example, access all 'x' by
        # pos_arr['x'] and so on
        pos_arr = np.fromfile(f, epos_d_type, -1)
    return pos_arr


def write_epos_numpy(epos, fn):
    """
    Write the data to a .epos file (binary file type).  See the
    ``read_epos_numpy'' function for more information on the .epos file format.

    Parameters
    ----------
    epos : Numpy array
        The data to write
    fn : str
        The filename to which to write

    Returns
    -------
    equal_bytes, fn : bool, str
        Tuple containing whether the number of bytes written was equal to the
        number of bytes in the epos file, and the filename written to
    """

    if not fn.lower().endswith('.epos'):
        fn += '.epos'
    with open(fn, 'wb') as f:
        bytes_written = f.write(epos.tobytes())
        equal_bytes = (bytes_written == epos.nbytes)
    return equal_bytes, fn


def write_pos_numpy(pos, fn):
    """
    Write the data to a .pos file (binary file type).  See the
    ``read_pos_numpy'' function for more information on the .pos file format.

    Parameters
    ----------
    pos : Numpy array
        The data to write
    fn : str
        The filename to which to write

    Returns
    -------
    equal_bytes, fn : bool, str
        Tuple containing whether the number of bytes written was equal to the
        number of bytes in the epos file, and the filename written to
    """

    if not fn.lower().endswith('.pos'):
        fn += '.pos'
    with open(fn, 'wb') as f:
        bytes_written = f.write(pos.tobytes())
        equal_bytes = (bytes_written == pos.nbytes)
    return equal_bytes, fn
