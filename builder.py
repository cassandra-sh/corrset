#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:14:35 2017
builder.py
-Constructs the catalogs from scratch, step by step
Steps, in order
1. add venice flags
2. turn fits into csvs
3. add wise flags
4. do the cross matches
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
import re
import subprocess
from matplotlib.colors import LogNorm
import psutil
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


"""Catalogs info"""
#note: this version assumes the random is already assembled
p_rand_u_fits = "/scr/depot0/csh4/cats/unprocessed/rand_u.fits"
p_rand_v_fits = "/scr/depot0/csh4/cats/partial/rand_v.fits"
p_rand_v =      "/scr/depot0/csh4/cats/partial/rand_v.csv"
p_rand_vw =     "/scr/depot0/csh4/cats/partial/rand_vw.csv"
p_rand_f =      "/scr/depot0/csh4/cats/processed/rand_f.csv"

p_agn_u_fits =  "/scr/depot0/csh4/cats/unprocessed/agn_u.fits"
p_agn_u =       "/scr/depot0/csh4/cats/unprocessed/agn_u.csv"
p_agn_p =       "/scr/depot0/csh4/cats/partial/agn_p.csv"     #1/2 cross matches done (hsc)
p_agn_f =       "/scr/depot0/csh4/cats/processed/agn_f.csv"   #2/2 cross matches done (sdss)

p_hsc_u_fits =  "/scr/depot0/csh4/cats/unprocessed/hsc_u.fits"
p_hsc_v_fits =  "/scr/depot0/csh4/cats/partial/hsc_v.fits"
p_hsc_v =       "/scr/depot0/csh4/cats/partial/hsc_v.csv"
p_hsc_vw =      "/scr/depot0/csh4/cats/partial/hsc_vw.csv"
p_hsc_f =       "/scr/depot0/csh4/cats/processed/hsc_f.fits"  #specz cross match done

p_specz_fits = "/scr/depot0/csh4/cats/reference/DR1_specz_catalog.fits"
p_specz = "/scr/depot0/csh4/cats/reference/DR1_specz_catalog.csv"
specz_raname   = 'ra2000  '
specz_decname  = 'decl2000'
specz_specname = 'redshift'

p_sdss_fits = "/scr/depot0/csh4/cats/reference/sdss_quasar_cat.fits"
p_sdss = "/scr/depot0/csh4/cats/reference/sdss_quasar_cat.csv"
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

#To run Venice from Python for file_in, file_out, use 
#os.popen(sample_in)

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
         print(report_string)
         print("Time is " + str(int(current_time()/60)) + " minutes from start. ", end="")
         print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
         sys.stdout.flush()
         gc.collect()
         print("")
    report("Beginning builder.main()")
    
#    """
#    Step 1: Venice flags for
#                -rand_u_fits -> rand_v_fits
#                -hsc_u_fits  -> hsc_v_fits
#    Already done
#    """
#    report("Flagging each relevant catalog with Venice")
#    venice_mask(p_rand_u_fits, p_rand_v_fits, overwrite=True)
#    venice_mask(p_hsc_u_fits, p_hsc_v_fits, overwrite=True)
#    
#    """
#    Step 2: FITS to csv for
#                -sdss_fits   -> sdss
#                -specz_fits  -> specz
#                -hsc_v_fits  -> hsc_v
#                -rand_v_fits -> rand_v
#                -agn_u_fits  -> agn_u
#    
#    Currently using topcat for this part    
#    """
#    report("File 1")
#    fits_to_csv(p_sdss_fits, p_sdss, overwrite=True)
#    report("File 2")
#    fits_to_csv(p_specz_fits, p_sdss, overwrite=True)
#    report("File 3")
#    fits_to_csv(p_hsc_v_fits, p_hsc_v, overwrite=True)
#    report("File 4")
#    fits_to_csv(p_rand_v_fits, p_rand_v, overwrite=True)
#    report("File 5")
#    fits_to_csv(p_agn_u_fits, p_agn_u, overwrite=True)
    
    """
    Step 3: WISE flags for
                -rand_v -> rand_vw
                -hsc_v  -> hsc_vw
    """
    report("Flagging each relevant catalog with WISE mask from DiPompeo et al. 2017")
    add_wise_mask_column(p_rand_v, p_rand_vw, chunksize=500000)
    add_wise_mask_column(p_hsc_v, p_hsc_vw, chunksize=500000)
    
    """
    Step 4: Cross matches
                -hsc_vw, specz -> hsc_f
                -agn_u,  hsc_f -> agn_p
                -agn_p,  sdss  -> agn_f
    """
    report("Cross matching HSC to Spec-Z")
    file_cross_match(p_hsc_vw, p_specz, p_hsc_f,
                     method='closest', radius=2, haystack_ra_name = specz_raname,
                     haystack_dec_name = specz_decname, sfx='_specz', chunksize=500000)
    report("Cross matching WISE to HSC")
    file_cross_match(p_agn_u, p_hsc_f, p_agn_p, chunksize=500000,
                     method='brightest', radius=2, sfx='hsc1')
    report("Cross matching WISE to SDSS")
    file_cross_match(p_agn_p, p_sdss, p_agn_f, chunksize=500000,
                     method='closest', radius=2, haystack_ra_name = specz_raname,
                     haystack_dec_name = specz_decname, sfx='_sdss_t1')
    
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
    
def add_wise_mask_column(unprocessed, masked, ra_name='ra', dec_name='dec',
                         chunksize=10000):
    """
    Apply the WISE mangle mask to a file. 
    
    Adds a column called "wise_flag" which is 1 if the object is not masked
    and 0 if it is masked
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new masked file will be saved
        ra_name, dec_name - the name of the columns
        chunksize         - size of chunks to read and write the file in
    """
    #1. Get the unprocessed csv in chunks
    frame = pd.read_csv(unprocessed, chunksize=chunksize)
    #2. Iterate through the chunks
    for chunk in frame:
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
        chunk.to_csv(masked, mode='a')

def wise_mask(ra, dec, mangle_files=wise_mangles):
    """
    Takes a list of RA and DEC and returns the indices which are permissible
    
    Uses equatorial coordinates, converts them to galactic, does the thing, and
    returns the indices.
    
    Dependencies: numpy, astropy.coordinates.SkyCoord, mangle
    
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

def file_cross_match(needle_path, haystack_path, new_needle_path,
                     method="closest", radius=2, prefix='',
                     needle_ra_name='ra', needle_dec_name='dec',
                     haystack_ra_name='ra', haystack_dec_name='dec',
                     metric='imag_kron', chunksize=10000, sfx="_cmatch"):
    """
    Does a cross match with the given file paths, using cross_match

    @params
        needle_path, haystack_path
            paths to fits files which are to be crossmatched
        new_needle_path
            path to save the new fits file to
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
    """
    #0. Prepare some things.
    needle = pd.read_csv(needle_path)
    haystack = pd.read_csv(haystack_path, chunksize=chunksize)
    
    #1. Get the indices of the matches
    haystack_indices = cross_match(needle, haystack, method=method,
                                   radius=radius,
                                   n_ra = needle_ra_name,
                                   n_dec = needle_dec_name,
                                   h_ra = haystack_ra_name,
                                   h_dec = haystack_dec_name,
                                   metric = metric)
    #reload haystack after iterating through chunks
    haystack = pd.read_csv(haystack_path, chunksize=chunksize) 
    needle_indices = range(0, needle.shape[0])
    #2. Iterate through the haystack adding the relevant entries to needle
    frame_to_add = None
    for chunk in haystack:
        #i.   Figure out which indices are relevant to this chunk
        chunk_min, chunk_max = min(chunk.index), max(chunk.index)
        relevant = np.logical_and(np.less(haystack_indices, chunk_max),
                                  np.greater(haystack_indices, chunk_min))
        relevant_indices = np.where(relevant)[0]
        relevant_needle_indices = [needle_indices[i] for i in relevant_indices]
        relevant_haystack_indices = [haystack_indices[i] for i in relevant_indices]
        #ii.  Append relevant haystack entries the frame_to_add object
        attach = haystack.iloc[relevant_haystack_indices]
        attach["index"] = relevant_needle_indices
        if frame_to_add == None: frame_to_add = attach
        else: frame_to_add = frame_to_add.append(attach, ignore_index=True)
    #3. Attach the frame to add to the needle frame
    frame_to_add.set_index('index')
    needle = needle.join(attach, rsuffix=sfx)
    #4. Save the new (needle) object
    needle.to_csv(new_needle_path)
    

def cross_match(needle, haystack, method="closest", radius=2,
                metric="imag_kron", n_ra = 'ra', n_dec = 'dec',
                h_ra = 'ra', h_dec = 'dec'):
    """
    Given the needles and haystack, return a list of indices that lead
    from each needle object to a haystack object that is searched for using the
    given algorithm. 
    
    Implemented with pandas. if n = len(needle) and m = len(haystack) then
        O ~ O((m + n)(log(n) + 1))
    
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
    indices = [list() for i in len(ra)]
    lengths = [list() for i in len(ra)]
    if method == "highest" or method == "bayesian": metrics = [list() for i in len(ra)]
    for chunk in haystack:
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

def fits_to_csv(path, new_path, overwrite=False):
    """
    Takes a fits file and turns it into an csv file using astropy.
    """
    raise NotImplementedError("""(Use topcat) - Overheard in code before crash:
      
                              Me:      'Astropy, convert this fits file into a
                                        csv file so I can use pandas...'
                              Astropy: 'I can't let you do that Cassandra.'
                              Astropy: 'I'm scared Cassandra...'""")
    
if __name__ == "__main__":
    main()
