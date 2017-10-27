#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:14:35 2017
builder.py
-Constructs the catalogs from scratch, step by step

Using hdf files primarily

@author: csh4
"""

import os
import healpy as hp
import numpy as np
import pymangle
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import gc
from scipy import spatial
import time
import sys
import matplotlib.pyplot as plt
import io
import warnings
import re
import subprocess
from matplotlib.colors import LogNorm
import psutil
from ast import literal_eval
import pandas as pd


"""
Catalog/file object nomenclature
name_x_fits - fits file type
p_name  - path to file
name_u  - unprocessed at all (typically a fits)
name_v  - venice mask applied (fits in and out)
name_vw - venice and wise mask applied
name_p  - partially processed (cross match started but not finished)
name_f  - fully processed/finished catalog
name_q  - quality cut catalog (whatever is decided on)
"""

"""Input parameters"""
hsc_csize = 1500000
rand_csize = 3000000

#Some extra functionality for predicting how many rows there are
n_hsc_rows = 38130959
n_rand_rows = 52094203
get_row_nums = False

"""Catalogs info"""
#note: this version assumes the random is already assembled
p_rand_u_fits = "/scr/depot0/csh4/cats/unprocessed/rand_u.fits"
p_rand_v_fits = "/scr/depot0/csh4/cats/partial/rand_v.fits"
p_rand_v =      "/scr/depot0/csh4/cats/partial/rand_v.hdf5"
p_rand_vw =     "/scr/depot0/csh4/cats/partial/rand_vw.hdf5"
p_rand_f =      "/scr/depot0/csh4/cats/processed/rand_f.hdf5"

p_agn_u_fits =  "/scr/depot0/csh4/cats/unprocessed/agn_u.fits"
p_agn_u =       "/scr/depot0/csh4/cats/unprocessed/agn_u.hdf5"
p_agn_p =       "/scr/depot0/csh4/cats/partial/agn_p.hdf5"     #1/2 cross matches done (hsc)
p_agn_f =       "/scr/depot0/csh4/cats/processed/agn_f.hdf5"   #2/2 cross matches done (sdss)

p_hsc_u_fits =  "/scr/depot0/csh4/cats/unprocessed/hsc_u.fits"
p_hsc_v_fits =  "/scr/depot0/csh4/cats/partial/hsc_v.fits"
p_hsc_v =  "/scr/depot0/csh4/cats/partial/hsc_v.hdf5"
p_hsc_vw =      "/scr/depot0/csh4/cats/partial/hsc_vw.hdf5"
p_hsc_f =       "/scr/depot0/csh4/cats/processed/hsc_f.hdf5"  #specz cross match done

p_specz_fits = "/scr/depot0/csh4/cats/reference/DR1_specz_catalog.fits"
p_specz = "/scr/depot0/csh4/cats/reference/DR1_specz_catalog.hdf5"
specz_raname   = 'ra2000'
specz_decname  = 'decl2000'
specz_specname = 'redshift'

p_sdss_fits = "/scr/depot0/csh4/cats/reference/sdss_quasar_cat.fits"
p_sdss = "/scr/depot0/csh4/cats/reference/sdss_quasar_cat.hdf5"
mgII_name = 'FWHM_MGII'
cIV_name  = 'FWHM_CIV'

"""Masks Info"""
wise_mangles = ["/u/csh4/Dropbox/research/agn/polygons/wise_mask_allwise_stars.ply",
                "/u/csh4/Dropbox/research/agn/polygons/wise_mask_allsky_pix.ply"]

"""Venice info"""
venice = "/scr/depot0/csh4/tools/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice"
venice_masks = " -m /scr/depot0/csh4/tools/HSC-SSP_brightStarMask_Arcturus/reg/masks_all.reg"
file_in = "/scr/depot0/csh4/cats/unprocessed/rand_d.fits"
file_out = "/scr/depot0/csh4/cats/unprocessed/rand_f.fits"

sample_in = (venice + venice_masks + " -f all -cat " + file_in + " -xcol RA -ycol DEC -o " + file_out)

"""Stilts info"""
stilts = "/scr/depot0/csh4/tools/topcat/stilts"

"""Flag info"""
flag_names = ["iflags_pixel_saturated_any",
              "iflags_pixel_edge",
              "iflags_pixel_interpolated_any",
              "iflags_pixel_interpolated_center",
              "iflags_pixel_saturated_any",
              "iflags_pixel_saturated_center",
              "iflags_pixel_cr_any",
              "iflags_pixel_cr_center",
              "iflags_pixel_bad",
              "iflags_pixel_suspect_any",
              "iflags_pixel_suspect_center",
              "iflags_pixel_offimage",
              "iflags_pixel_bright_object_center",
              "iflags_pixel_bright_object_any",
              "icmodel_flags_badcentroid"]


def main():
    """
    Build everything from scratch. Comment out the finished parts.
    """
    start_time = int(time.clock())
    def current_time():
        return (int(time.clock()) - start_time)
    def report(report_string):
        sys.stdout.flush()
        time = current_time()
        print(report_string)
        print("Time is " + str(float(time/60.)) + " minutes from start. ", end="")
        print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
        sys.stdout.flush()
        gc.collect()
        print("")
    report("Beginning builder.main()")
    
    if get_row_nums:
        pass
    
#    """
#    Step 1: Venice flags for
#                -rand_u_fits -> rand_v_fits
#                -hsc_u_fits  -> hsc_v_fits
#    Already works/done
#    """
#    report("Flagging each relevant catalog with Venice")
#    venice_mask(p_rand_u_fits, p_rand_v_fits, overwrite=True)
#    venice_mask(p_hsc_u_fits, p_hsc_v_fits, overwrite=True)
    
#    """
#    Step 2: FITS to hdf for
#                -sdss_fits   -> sdss
#                -specz_fits  -> specz
#                -hsc_v_fits  -> hsc_v
#                -rand_v_fits -> rand_v
#                -agn_u_fits  -> agn_u
#       
#    """
#    report("File 1")
#    fits_to_hdf(p_sdss_fits, p_sdss, overwrite=True)
#    report("File 2")
#    fits_to_hdf(p_specz_fits, p_specz, overwrite=True)
#    report("File 3")
#    fits_to_hdf(p_hsc_v_fits, p_hsc_v, overwrite=True)
#    report("File 4")
#    fits_to_hdf(p_rand_v_fits, p_rand_v, overwrite=True)
#    report("File 5")
#    fits_to_hdf(p_agn_u_fits, p_agn_u, overwrite=True)
    """
    Step 3: WISE flags for
                -rand_v -> rand_vw
                -hsc_v  -> hsc_vw
    """
    report("Flagging hsc catalog with wise mask")
    add_wise_mask_column(p_hsc_v, p_hsc_vw, chunksize=hsc_csize, verbose=True,
                         n_rows=n_hsc_rows)
    report("Flagging random catalog with wise mask")
    add_wise_mask_column(p_rand_v, p_rand_vw, chunksize=rand_csize, verbose=True,
                         n_rows=n_rand_rows)

    """
    Step 4: Cross matches
                -specz,  hsc_vw -> hsc_f (right join)
                -agn_u,  hsc_f  -> agn_p (left  join)
                -agn_p,  sdss   -> agn_f (left  join)
    """
    report("Cross matching HSC to Spec-Z")
    file_cross_match(p_specz, p_hsc_vw, p_hsc_f,
                     method='closest', radius=2, needle_ra_name = specz_raname,
                     needle_dec_name = specz_decname, sfx='_sz', append="right",
                     chunksize=hsc_csize, verbose=True, n_rows=n_hsc_rows)
    
    report("Cross matching WISE to HSC")
    file_cross_match(p_agn_u, p_hsc_f, p_agn_p, chunksize=hsc_csize,
                     n_rows=n_hsc_rows,  method='brightest', radius=2, 
                     sfx='_hsc1', verbose=True, needle_ra_name = "RA",
                     needle_dec_name = "DEC")
    
    report("Cross matching WISE to SDSS")
    file_cross_match(p_agn_p, p_sdss, p_agn_f, chunksize=None, verbose=True,
                     method='closest', radius=2, sfx='_sdss_t1')
    
    report("Done")


def venice_mask(unprocessed, masked, ra_name='ra', dec_name='dec', overwrite=False):
    """
    Venice masks the fits at the given path. 
    
    Adds a column called 'flag' which is 1 if the object is not masked and 
    0 if it is masked.
    
    dependencies: subprocess, Venice
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new file will be saved
        ra_name, dec_name - the name of the columns
    """
    o = ""
    if overwrite == True:
        o = "!"
    command = (venice + venice_masks + " -f all -cat " + unprocessed +
                     " -xcol " + ra_name + " -ycol " + dec_name + " -o " + o + masked)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()


def file_len(path):
    #Figure out the number of lines in a file
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

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
            
#    Possible fitsio implementation:
#
#    #Make sure to populate names with the non-list column names first
#
#    #Get the fits file with fitsio
#    fits=fitsio.FITS('data.fits')
#        
#    #Figure out a smart way to take into account the end of the file
#    range1 = np.arange(0, filelimit, chunksize)
#    range2 = np.arange(chunksize, filelimit, chunksize)
#    range2[-1] = filelimit
#    for i in range(len(range1)):
#       dat = fits[1][range[1][i]:range2[i]]
#       dct_to_add = {}
#        for j in range(len(names)):
#            dct_to_add[names[j]] = dat[names[j]]
#        frame = pd.DataFrame.from_dict(dct_to_add)
#        if i == 0:
#            frame.to_hdf(new_path, mode='w', format="table", key="primary")
#        else:
#            frame.to_hdf(new_path, mode='a', append=True, format="table",
#                         key="primary")
            
    
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
        pd.DataFrame.from_dict(dct).to_hdf(new_path, key="primary",
                              format="table")
    
def add_wise_mask_column(unprocessed, masked, ra_name='ra', dec_name='dec',
                         chunksize=10000, verbose=False, n_rows=None):
    """
    Apply the WISE mangle mask to a file. 
    
    Adds a column called "wise_flag" which is 1 if the object is not masked
    and 0 if it is masked.
    
    For this one, higher chunksize is much better and will improve performance
    if memory is available.
    
    performance: low
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new masked file will be saved
        ra_name, dec_name - the name of the columns
        chunksize         - size of chunks to read and write the file in
    """
    #0. Prepare some things
    if verbose:
        if n_rows == None:
            n_rows = (int(file_len(unprocessed)/chunksize)+1)
        else:
            n_rows = int(n_rows/chunksize)+1
        print("╠", end="")
        for i in range(n_rows): 
            print("═", end="")
        print("╣")
        print("╠", end="")
    
    #1. Get the unprocessed csv in chunks
    frame = pd.read_hdf(unprocessed, chunksize=chunksize)
    if chunksize == None: 
        frame = [frame]
    
    n = 0    
    #2. Iterate through the chunks
    for chunk in frame:
        if verbose: print("█", end="")
        #i.   Get ra and dec
        ra = chunk[ra_name]
        dec = chunk[dec_name]
        #ii.  Feed the RA and DEC into the wise mask function
        unmasked_indices = wise_mask(ra, dec)
        #iii. Use the array returned to construct an array to add to the DataFrame
        array_to_add = np.zeros(len(ra), dtype=int)
        for i in unmasked_indices:
            array_to_add[i] = 1
        #iv.  Add it to the DataFrame
        chunk['wise_flag']  = array_to_add
        #v.   Append to the csv
        if n == 0:
            chunk.to_hdf(masked, mode='w', format="table", key="primary")
            n = 1
        else:
            chunk.to_hdf(masked, mode='a', append=True, format="table",
                         key="primary")
        #vi.  Clean up
        chunk = None
        gc.collect()
        
    if verbose: print("╣")

def wise_mask(ra, dec, mangle_files=wise_mangles):
    """
    Takes a list of RA and DEC and returns the indices which are permissible
    
    Uses equatorial coordinates, converts them to galactic, does the thing, and
    returns the indices.
    
    Dependencies: numpy, astropy.coordinates.SkyCoord, mangle
    
    performance: medium (probably max)
    
    @params
        ra_list, dec_list
            The coordinate pairs to be compared to the footprint
        mangle_files = wise_mangles
            The iterable of full filepaths to the .ply files to be used for
            masking
        galactic = True
        
    @returns indices
            The indices of ra and dec which are not masked 
    """
    coordinates = SkyCoord(ra=ra*u.degree,
                           dec=dec*u.degree,
                           frame='icrs')
    coordinates = coordinates.transform_to("galactic")
    ga_ra_list = coordinates.l.deg
    ga_dec_list = coordinates.b.deg
    m = []
    for path in mangle_files:
        m.append(pymangle.Mangle(path))
    ins = []
    for j in range(0, len(m)):
        ins.append(m[j].contains(ga_ra_list, ga_dec_list))
    ins_true = ins[0]
    for j in range(1, len(m)):
        ins_true = np.logical_or(ins_true, ins[j])
    return np.where(ins_true == False)[0]

def file_cross_match(needle_path, haystack_path, new_path,
                     method="closest", radius=2, prefix='', verbose=False,
                     needle_ra_name='ra', needle_dec_name='dec', n_rows=None,
                     haystack_ra_name='ra', haystack_dec_name='dec',
                     metric='imag_kron', chunksize=10000, sfx="_cmatch", 
                     append='left'):
    """
    Does a cross match with the given file paths, using cross_match

    performance: not sure yet, high priority
    
    @params
        needle_path, haystack_path
            paths to hdf5 files which are to be crossmatched
        new_path
            path to save the new hdf5 to
        method = "closest", "brightest", "circle", or "bayesian"
            Three methods for searching. Each do the following:
                "closest"   - returns the closest object within the limit
                "highest"   - return the object with the higest of a given value
                                which is haystack[1].data[highest_metric], 
                                imag_kron by default
                "bayesian"  - return an index prioritized by some kind of
                                function that cares about brightness and 
                                closeness]
                (circle not allowed because this is a file cross match, which 
                needs 1 result per entry)
        radius = 2
            Number of arc seconds to look within for the search. 
            Give limit <= 0 for no limit.
        prefix = ''
            name to put at beginning of column names to specify source
        needle_ra_name, needle_dec_name, haystack_ra_name, haystack_dec_name
            The names of the columns in the needle and haystack lists. 
            Case insensitive.
        chunksize
            Amount of rows to pull from haystack at any given time
            None for no chunking
        verbose
            If true, print some minimal diagnostics mid run to let the user know
            how much longer there is to go
        sfx = "_cmatch"
            String to append to each cross matched column
        n_rows = None
            Number of rows in haystack. Optional parameter, which will be used
            (and generated if not provided) if and only if verbose is true
        append = "left" or "right"
            "left"  - Add haystack columns to needle and save
            "right" - Add needle columns to haystack and save        
    """    
    #0.  Getting the DataFrames 
    needle = pd.read_hdf(needle_path)
    haystack = pd.read_hdf(haystack_path, chunksize=chunksize)
    
    #Wrap haystack in a list if not using Chunker
    if chunksize == None: haystack = [haystack] 
    if verbose:
        print("Running a cross match with " + needle_path + " and " + haystack_path)
        if n_rows == None:
            n_rows = int((file_len(haystack_path)-1)/chunksize)+1
        else:
            n_rows = int(n_rows/chunksize)+1

    #1. Get the indices of the matches
    haystack_indices = cross_match(needle, haystack, method=method,
                                   radius=radius,
                                   n_ra = needle_ra_name,
                                   n_dec = needle_dec_name,
                                   h_ra = haystack_ra_name,
                                   h_dec = haystack_dec_name,
                                   metric = metric,
                                   verbose = verbose, n_rows = n_rows)
    #(reload haystack after iterating through chunks)
    haystack = pd.read_hdf(haystack_path, chunksize=chunksize)
    if chunksize == None:
        haystack = [haystack]
        
    needle_indices = range(0, needle.shape[0])
    
    #2. Iterate through the haystack adding the relevant entries to needle
    frame_to_add = None
    if verbose:
        print("Using cross match result to add catalogs together")
        print("╠", end="")
        for i in range(n_rows): print("═", end="")
        print("╣")
        print("╠", end="")
    for chunk in haystack:
        if verbose: print("█", end="")
        
        #i.   Figure out which indices are relevant to this chunk
        chunk_min, chunk_max = min(chunk.index), max(chunk.index)
        relevant = np.logical_and(np.less(haystack_indices, chunk_max),
                                  np.greater(haystack_indices, chunk_min))
        relevant_indices = np.where(relevant)[0]
        relevant_needle_indices = [needle_indices[i] for i in relevant_indices]
        relevant_haystack_indices = [haystack_indices[i] for i in relevant_indices]
        
        #ii.  Append relevant haystack entries the frame_to_add object
        if append == "left":
            attach = haystack.iloc[relevant_haystack_indices]
            attach["index"] = relevant_needle_indices
        elif append == "right":
            attach = needle.iloc[relevant_needle_indices]
            attach["index"] = relevant_haystack_indices
        else: raise ValueError("append = " + str(append) +
                              " is not 'left' or 'right'")
        
        if frame_to_add == None: frame_to_add = attach
        else: frame_to_add = frame_to_add.append(attach, ignore_index=True)
        
        #iii.  Clean up
        chunk = None
        gc.collect()
    if verbose: print("╣")
    
    #3. Attach the frame to add to the desired frame and save
    frame_to_add.set_index('index')
    
    if append == "left":
        needle = needle.join(frame_to_add, rsuffix=sfx)
        needle.to_hdf(new_path, key='primary', format="table")
    elif append == "right":
        #(reload haystack one last time)
        haystack = pd.read_hdf(haystack_path, chunksize=chunksize)
        if chunksize == None:
            haystack = [haystack]
        
        #Iterate through haystack chunks, saving to the new path
        n = 0
        for fistful_of_hay in haystack:
            fistful_of_hay = fistful_of_hay.join(frame_to_add, rsuffix=sfx)
            if n == 0:
                fistful_of_hay.to_hdf(new_path, mode='w', format="table", key="primary")
                n = 1
            else:
                fistful_of_hay.to_hdf(new_path, mode='a', header=False,
                                      append=True, format="table",
                                      key="primary")
    else: raise ValueError("append = " + str(append) +
                           " is not 'left' or 'right'")
    

def cross_match(needle, haystack, method="closest", radius=2,
                metric="imag_kron", n_ra = 'ra', n_dec = 'dec',
                h_ra = 'ra', h_dec = 'dec', verbose=False, n_rows=None):
    """
    Given the needles and haystack, return a list of indices that lead
    from each needle object to a haystack object that is searched for using the
    given algorithm. 
    
    Implemented with pandas. if n = len(needle) and m = len(haystack) then
        O ~ O((m + n)(log(n) + 1))
    
    performance: medium, probably max
    
    @params
        needle, haystack
            Data frames. Haystack must be in chunks.
        method = "closest", "brightest", "circle", or "bayesian"
            Four methods for searching. Each do the following:
                "closest"   - returns the closest object within the limit
                "highest"   - return the object with the higest of a given value
                                which is haystack[1].data[highest_metric], 
                                imag_kron by default
                "circle"    - return a list of every index within the circle
                "bayesian"  - return an index prioritized by some kind of
                                function that cares about brightness and 
                                closeness
        radius = 2
            Number of arc seconds to look within for the search. 
            Give limit <= 0 for no limit.
        metric
            the key of the value to use for "brightest" or "bayesian" searches
            in the haystack csv file
        n_ra, n_dec, h_ra, h_dec = 'ra', 'dec', 'ra', 'dec'
            The names of the keyes for ra and dec in needle and haystack
        n_rows
            Optional parameter which contains the number of rows for a verbose 
            print
    
    @returns
        indices
            A list of ints with shape (len(needle)) OR a list of lists of int
            (but only if method = "circle").
            
            Entry will be None if no match was found (or an empty list for 
            "circle")
    """
    #0. Prep some things
    radius_in_cartesian = 2*np.sin((2*np.pi*radius)/(2*3600*360))
    if radius <= 0: #because we are projected on a unit sphere, 3 is more than enough.
        radius_in_cartesian = 3.0 
    metrics = []
    
    #1. Get the needle ra and dec, build the KDTree
    ra, dec = needle[n_ra], needle[n_dec]
    tree = spatial.KDTree(np.array(ra_dec_to_xyz(ra, dec)).T)
    
    #2. Use pandas chunker to iterate on the haystack
    indices = [list() for i in ra]
    lengths = [list() for i in ra]
    if method == "highest" or method == "bayesian": metrics = [list() for i in ra]
    n = 0
    if verbose:
        print("Doing a cross match")
        print("╠", end="")
        for i in range(n_rows): 
            print("═", end="")
        print("╣")
        print("╠", end="")
    for chunk in haystack:
        n = n + 1
        if verbose: print("█", end="")
        
        #i.   get ra, dec, and metric if applicable
        ra, dec = chunk[h_ra], chunk[h_dec]
        if method == "highest" or method == "bayesian":
            metrics = metrics + chunk[metric].values
        key = np.array(ra_dec_to_xyz(ra, dec)).T
        
        #ii.  search the tree
        chunk_distances, chunk_indices = tree.query(key, k=1, distance_upper_bound = radius_in_cartesian)
        
        #iii. record results back to the indices list
        for i in range(0, chunk.shape[0]):
            if chunk_distances[i] < 3:
                indices[chunk_indices[i]].append(chunk.index[i])
                lengths[chunk_indices[i]].append(chunk_distances[i])
                if method == "highest" or method == "bayesian":
                    metrics[chunk_indices[i]].append(chunk[metric][chunk.index[i]])
                    
        #iv.  Clean up
        chunk = None
        gc.collect()
        
    if verbose: print("╣")
        
    #3. Discern the correct values based on the given algorithm
    if method == "circle": #just return the circle
        return indices
    elif method == "closest":
        for i in range(0, len(indices)):
            indices[i] = max(range(len(lengths[i])), default=None,
                             key=lengths[i].__getitem)
        return indices
    else:
        if method == "brightest":
            for i in range(0, len(indices)):
                indices[i] = max(range(len(metric[i])), default=None,
                                 key=metric[i].__getitem)
            return indices
        elif method == "bayesian":
            raise NotImplementedError("method = bayesian is not implemented yet")
        else:
            raise ValueError("method = " + method +
                             "is not an allowed value for cross_match")
    
def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]


def xyz_to_ra_dec(x, y, z):
    """
    Convert back to RA and DEC
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    xyz = [x, y, z]
    xy = xyz[:,0]**2 + xyz[:,1]**2
    dec = np.arctan2(xyz[:,2], np.sqrt(xy))
    ra = np.arctan2(xyz[:,1], xyz[:,0])
    return ra, dec

    
if __name__ == "__main__":
    main()
