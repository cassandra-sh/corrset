# -*- coding: utf-8 -*-
"""
fits2hdf.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

Program Description:
    Turns a fits file into an HDF file. 
    
Inputs for this correlation astronomy code are all in HDF objects, so a user
might want to use this to convert their files over. 

Currently, loads the entire fits file into memory to do this, which is very
memory inefficient, but unfortunately astropy doesn't effectively chunk tables,
so until I implement pyfits this will have to do. 
"""

import os
import numpy    as np
import pandas   as pd
import warnings
from astropy.io import fits

def fits_to_hdf(path, new_path, overwrite=False):
    """
    Take a fits file and use astropy to retrieve the data. Save the data in
    an hdf5 table file.
    
    performance: low (astropy.io.fits memory limited)
    
    @params
        path
            Path to get the fits file from
        new_path
            Path to save the hdf5 table to
        overwrite=False
            If false, raise an error if something is already at new_path
    """
    if overwrite == False and os.path.isfile(new_path):
        raise IOError("Overwrite = False but there is a file at " + new_path)
    
    f = fits.open(path)
    dat = f[1].data
    names = f[1].columns.names
    types = [f[1].columns.dtype[i] for i in range(len(names))]
    
    shapes = [len(np.shape(types[i])) for i in range(len(names))]
    not_list_columns = np.where(np.less(shapes, 1))[0]
    
    if len(not_list_columns) == len(names):
        frame = pd.DataFrame(data = dat)
        frame.to_hdf(new_path, key="primary", format="table")
    else:
        dropped_names = [names[i] for i in np.where(np.greater(shapes,0))[0]]
        warnings.warn(path + " has lists stored in columns " +
                      str(dropped_names) + ", which will be dropped.")
        dct = {}
        for i in not_list_columns:
            dct[names[i]] = dat[names[i]]
        frame = pd.DataFrame.from_dict(dct)
        frame.to_hdf(new_path, key="primary",format="table")