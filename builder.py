#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:14:35 2017

builder.py
-Constructs the catalogs from scratch, step by step

Steps, in order
1. assemble random catalog
2. add all the flags
3. do the cross matches

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
from matplotlib.colors import LogNorm
import psutil

"""Catalogs info"""
raw_random_directory = "/scr/depot0/csh4/cats/unprocessed/randoms/"
random_unprocessed = "/scr/depot0/csh4/cats/unprocessed/rand_u.fits"
random_venice_masked = "/scr/depot0/csh4/cats/partial/rand_venice_masked.fits"
random_wise_venice_masked = "/scr/depot0/csh4/cats/partial/rand_wise_venice_masked.fits"
random_processed = "/scr/depot0/csh4/cats/processed/rand_p.fits"

agn_unprocessed = "/scr/depot0/csh4/cats/unprocessed/agn_u.fits"
agn_partial1_processed = "/scr/depot0/csh4/cats/partial/agn_p1.fits"
agn_partial2_processed = "/scr/depot0/csh4/cats/partial/agn_p2.fits"
agn_processed = "/scr/depot0/csh4/cats/processed/agn_p.fits"
agn_type1_processed = "/scr/depot0/csh4/cats/processed/agn1_p.fits"
agn_type2_processed = "/scr/depot0/csh4/cats/processed/agn2_p.fits"

hsc_unprocessed = "/scr/depot0/csh4/cats/unprocessed/hsc_u.fits"
hsc_venice_masked = "/scr/depot0/csh4/cats/partial/hsc_venice_masked.fits"
hsc_wise_venice_masked = "/scr/depot0/csh4/cats/partial/hsc_wise_venice_masked.fits"
hsc_processed = "/scr/depot0/csh4/cats/processed/hsc_p.fits"

specz_reference = "/scr/depot0/csh4/cats/reference/DR1_specz_catalog.fits"
specz_raname   = 'ra2000  '
specz_decname  = 'decl2000'
specz_specname = 'redshift'

sdss_reference = "/scr/depot0/csh4/cats/reference/sdss_quasar_cat.fits"
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
    Build everything from scratch, with options to skip each part
    """
    start_time = int(time.clock())
    def current_time():
        return (int(time.clock()) - start_time)
    def report(report_string):
         print(report_string)
         print("Time is " + str(current_time/60) + "  minutes from start. ", end="")
         print("Memory use is" + str(psutil.virtual_memory().percent) + "%")
         sys.stdout.flush()
         gc.collect()
         print("")
    report("Beginning builder.main()")
    """
    Step 1: Assemble random catalog from randoms directory
    """
    report("Assembling the random catalog from directory")
    combine(random_unprocessed)
    """
    Step 2: Flag each catalog with WISE flags, venice flags
    """
    report("Flagging each relevant catalog with Venice")
    venice_mask(random_unprocessed, random_venice_masked)
    venice_mask(hsc_unprocessed, hsc_venice_masked)
    report("Flagging each relevant catalog with WISE mask from DiPompeo et al. 2017")
    add_wise_mask_column(random_venice_masked, random_wise_venice_masked)
    add_wise_mask_column(hsc_venice_masked, hsc_wise_venice_masked)
    """
    Step 3: Do each cross match
    
    file_cross_match(needle_path, haystack_path, new_needle_path,
                     method="closest", radius=2,
                     needle_ra_name='ra', needle_dec_name='dec',
                     haystack_ra_name='ra', haystack_dec_name='dec',
                     highest_metric='imag_kron'):
    """
    report("Cross matching WISE to HSC")
    file_cross_match(agn_unprocessed, hsc_wise_venice_masked, agn_partial1_processed,
                     method='brightest', radius=2, prefix='hsc1')
    report("Cross matching WISE to Spec-Z")
    file_cross_match(agn_partial1_processed, specz_reference, agn_partial2_processed,
                     method='closest', radius=2, haystack_ra_name = specz_raname,
                     haystack_dec_name = specz_decname, prefix='specz')
    report("Cross matching WISE to SDSS")
    file_cross_match(agn_partial2_processed, sdss_reference, agn_processed,
                     method='closest', radius=2, haystack_ra_name = specz_raname,
                     haystack_dec_name = specz_decname, prefix='sdsst1')
    report("Cross matching HSC to Spec-Z")
    file_cross_match(hsc_wise_venice_masked, specz_reference, hsc_processed,
                     method='closest', radius=2, haystack_ra_name = specz_raname,
                     haystack_dec_name = specz_decname, prefix='specz')
    report("Done")

def combine(new_path, directory = raw_random_directory, overwrite=False):
    """
    Combines the fits files in a directory, assuming they all have the same
    columns and in the same order. 
    """
    objects = os.listdir(directory)
    
    good_objects = []
    for i in range(0, len(objects)):
        if objects[i][-4:] == "fits":
            good_objects.append(str(directory + objects[i]))
    data_arrays = []
    col_names = []
    col_list = []
    combined_data = []
    formats = []
    for i in range(0, len(good_objects)):
        f = fits.open(good_objects[i])
        col_names = f[1].columns.names
        col_vals = []
        for n in col_names:
            col_vals.append(f[1].data[n].tolist())
        data_arrays.append(col_vals)
    for i in range(len(data_arrays[0])):
        combined_data.append([])
        for j in range(len(data_arrays)):
            combined_data[i] = combined_data[i] + data_arrays[j][i]
    for i in range(len(combined_data)):
        t = type(combined_data[i][0])
        if t == int:
            formats.append("K")
        elif t == float:
            formats.append("D")
        elif t == bool:
            formats.append("L")
        else:
            formats.append("a")
    for i in range(len(combined_data)):
        col_list.append(fits.Column(name=col_names[i], format=formats[i],
                                              array=data_arrays[i]))
    hdu = fits.HDUList(hdus = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(col_list)])
    try:
        hdu.writeto(new_path)
    except OSError:
        if overwrite:
            os.remove(new_path)
            hdu.writeto(new_path)
        else:
            raise OSError("There is already a file at " + new_path)


def add_column(hdu, path, arrays, names, overwrite=False):
    """
    Takes the given HDU and makes a new one with new columns added on to it.
    Saves the HDU to the path, closes it, opens the new HDU from that path, and
    returns it.
    
    @params
        hdu
            The thing to add the columns to
        path
            The file path to save the new HDU to
        arrays
            The arrays to add. Must be an array of arrays. If adding only one, 
            you have to wrap it so arrays[0] = your data
        names
            The names associated with the arrays to add
        overwrite = True
            Will only overwrite a file at path if overwrite=True
    
    @returns
        The new HDU
    """
    formats = []
    for i in range(len(arrays)):
        t = type(arrays[i][0])
        if t == int:
            formats.append("K")
        elif t == float:
            formats.append("D")
        elif t == bool:
            formats.append("L")
        else:
            formats.append("a")
    column_list_to_add = []
    for i in range(len(arrays)):
        column_list_to_add.append(fits.Column(name=names[i], format=formats[i],
                                              array=arrays[i]))
    hdu[1] = fits.BinTableHDU.from_columns(hdu[1].columns + column_list_to_add)
    try:
        hdu.writeto(path)
    except OSError:
        if overwrite:
            os.remove(path)
            hdu.writeto(path)
        else:
            raise OSError("There is already a file at '" +
                          path + "' and overwrite = False")
    #cleaning up and returning
    column_list_to_add = None
    hdu = None
    gc.collect()
    new_hdu = fits.open(path)
    return new_hdu

def venice_mask(unprocessed, masked, ra_name='ra', dec_name='dec'):
    """
    Venice masks the fits at the given path. 
    
    Adds a column called 'flag' which is 1 if the object is not masked and 
    0 if it is masked.
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new file will be saved
        ra_name, dec_name - the name of the columns
    """
    os.popen(venice + venice_masks + " -f all -cat " + unprocessed +
             " -xcol " + ra_name + " -ycol " + dec_name + " -o " + masked)

def add_wise_mask_column(unprocessed, masked, ra_name='ra', dec_name='dec'):
    """
    Apply the WISE mangle mask to a file. 
    
    Adds a column called "wise_flag" which is 1 if the object is not masked
    and 0 if it is masked
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new masked file will be saved
        ra_name, dec_name - the name of the columns
    """
    file = fits.open(unprocessed)
    ra = file[1].data[ra_name]
    dec = file[1].data[dec_name]
    unmasked_indices = wise_mask(ra, dec)
    array_to_add = np.zeros(len(ra))
    for i in unmasked_indices:
        array_to_add[i] = 1
    add_column(file, masked, array_to_add, 'wise_flag', overwrite=True)

def wise_mask(ra, dec, mangle_files=wise_mangles):
    """
    Takes a list of RA and DEC and returns the indices which are permissible
    
    Uses equatorial coordinates, converts them to galactic, does the thing, and
    returns the indices.
    
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
                     highest_metric='imag_kron'):
    """
    Does a cross match with the given file paths, using cross_match
    
        
    @params
        needle_path, haystack_path
            paths to fits files which are to be crossmatched
        new_needle_path
            path to save the new fits file to
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
        prefix = ''
            name to put at beginning of column names to specify source
        needle_ra_name, needle_dec_name, haystack_ra_name, haystack_dec_name
            The names of the columns in the needle and haystack lists. 
            Case insensitive.

    """
    needle = fits.open(needle_path)
    haystack = fits.open(haystack_path)
    indices = cross_match(needle, haystack, method=method, radius=radius,
                          needle_ra_name = needle_ra_name, needle_dec_name = needle_dec_name,
                          haystack_ra_name = haystack_ra_name, haystack_dec_name = haystack_dec_name,
                          highest_metric = highest_metric)
    haystack_col_names = haystack[1].columns.names
    
    #Here we need to pull arrays from haystack one at a time, get the relevant values,
    #make a new column, add it to the needle hdu, save it, reload the needle HDU, and
    #get the next array and repeat, as to keep the load on the memory as low as possible
    
    for i in range(len(haystack_col_names)):
        name_to_use = str(prefix + haystack_col_names[i])
        haystack_array = haystack[1].data[haystack_col_names[i]]
        needlized_array = [haystack_array[i] for i in indices]
        needle = add_column(needle, new_needle_path, needlized_array,
                            name_to_use, overwrite=True)
        haystack_array = None
        needlized_array = None
        gc.collect()
       

def cross_match(needle, haystack, method="closest", radius=2,
                needle_ra_name='ra', needle_dec_name='dec',
                haystack_ra_name='ra', haystack_dec_name='dec',
                highest_metric='imag_kron'):
    """
    Given the paths to needle and haystack, return a list of indices that lead
    from each needle object to a haystack object that is searched for using the
    given algorithm. 
    
    @params
        needle, haystack
            HDUlist objects (i.e. from fits.open()) which are to be crossmatched
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
        needle_ra_name, needle_dec_name, haystack_ra_name, haystack_dec_name
            The names of the columns in the needle and haystack lists. 
            Case insensitive.
    
    @returns
        indices
            A list of int indices of length = len(needle[1].data)
            If "circle" is given, it will be a list of lists of closest indices.
            Entry will be blank if there is no valid index for a given needle
            object. 
    """
    
    needle_search_set =  ra_dec_to_xyz(needle[1].data[needle_ra_name],
                                       needle[1].data[needle_dec_name])
    needle_search_set = np.array([needle_search_set[0],
                                  needle_search_set[1],
                                  needle_search_set[2]]).T
    haystack_search_set = ra_dec_to_xyz(haystack[1].data[haystack_ra_name], 
                                        haystack[1].data[haystack_dec_name])
    haystack_search_set = np.array([haystack_search_set[0],
                                    haystack_search_set[1],
                                    haystack_search_set[2]]).T
    haystack_tree = spatial.KDTree(haystack_search_set)
    radius_in_cartesian = 2*np.sin((2*np.pi*radius)/(2*3600*360))
    indices = []
    if radius <= 0: #because we are projected on a unit sphere, 3 is more than enough.
        radius_in_cartesian = 3.0 
    if method == "closest": #Just get the closest one
        distances, indices = haystack_tree.query(needle_search_set, eps=0.0, k=1,
                                                 distance_upper_bound = radius_in_cartesian)
    else: #For the others we have to query a circle and check what we get
        indices = haystack_tree.query_ball_point(needle_search_set, radius_in_cartesian)
        if method == "circle": #just return the circle
            return indices
        else:
            metric = haystack[1].data[highest_metric]
            if method == "brightest": #return the index of the brightest, None if none
                for i in range(0, len(indices)):
                    metric_array = [metric[j] for j in indices[i]]
                    indices[i] = max(range(len(metric_array)),
                           key=metric_array.__getitem, default=None)
            elif method == "bayesian":
                raise NotImplementedError("method = bayesian is not implemented yet")
            else:
                raise ValueError("method = " + method +
                                 "is not an allowed value for cross_match")
    #clean up and return
    haystack_tree = None
    haystack_search_set = None
    needle_search_set = None
    gc.collect()
    return indices
    
def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere

    Parameters
    ----------
    ra, dec : ndarrays

    Returns
    x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return  cos_ra * sin_dec, sin_ra * sin_dec, cos_dec


def xyz_to_ra_dec(x, y, z):
    """
    Convert back to RA and DEC

    Parameters
    ----------
    x, y, z : ndarrays

    Returns
    ra, dec : ndarrays
    """
    xyz = [x, y, z]
    xy = xyz[:,0]**2 + xyz[:,1]**2
    dec = np.arctan2(xyz[:,2], np.sqrt(xy))
    ra = np.arctan2(xyz[:,1], xyz[:,0])
    return ra, dec

if __name__ == "__main__":
    main()
    pass