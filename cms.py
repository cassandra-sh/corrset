#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:56:54 2018

cms - cross match simple

@author: csh4
"""

from   scipy  import spatial
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import gc
import os
import time
import sys
from matplotlib.colors import LogNorm
import hsc_random_gen
import matplotlib.colors as colors




def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere.
    
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

    return  np.array([cos_ra * sin_dec, sin_ra * sin_dec, cos_dec], dtype=float)


def cm_df(left, right, target, radius=2.0, l_ran = 'ra', l_decn = 'dec',
                                           r_ran = 'ra', r_decn = 'dec',
                     sfx='_join', l_csize = 1000000, r_csize = 3000000):
    """
    Conduct a cross match between two catalogs, saving the result.
    
    Attaches the right matches to the left catalog, so adjust inputs accordingly.
    
    @params
        left             - source for the left hdf
        right            - source for the right hdf
        target           - where to save the new HDF
        radius           - radius to search within for the cross match, in 
                           arcsec
        l_ran, l_decn    - ra and dec column names for the left catalog
        r_ran, r_decn    - ra and dec column names for the right catalog
        sfx              - string to append to attached column names
        l_csize, r_csize - chunksize to chunk left and right catalogs
    """
    
    #
    # Convert the search radius in arc seconds to cartesian distance on a unit sphere
    #
    rad_cart = 2*np.sin((np.pi/180.0)*(radius/3600.0)*(0.5))
    
    #
    # Prepare storage for the matches
    #
    l_all = []
    r_all = []
    d_all = []
    
    #
    # Go through each chunk on the right
    #
    right_chunks = pd.read_hdf(right, key='primary', chunksize=r_csize)
    for right_chunk in right_chunks:
        
        #
        # Get the right chunk data out, convert to cartesian coordinates on a unit sphere, and build a tree
        #
        right_chunk_xyz = ra_dec_to_xyz(right_chunk[r_ran], right_chunk[r_decn]).T
        right_chunk_tree = spatial.KDTree(right_chunk_xyz)
        
        # 
        # Then with each chunk on the left     
        #
        left_chunks  = pd.read_hdf(left,  key='primary', chunksize=l_csize)
        for left_chunk in left_chunks:
            
            #
            # Get the left chunk data out, convert to cartesian coordinates, and search the tree
            #
            left_chunk_xyz = ra_dec_to_xyz(left_chunk[l_ran], left_chunk[l_decn]).T
            distances, indices = right_chunk_tree.query(left_chunk_xyz, k=1, distance_upper_bound=rad_cart)
        
            #
            # With the search results, get all valid matches
            #
            l, r, d = [], [], []
            for i in range(len(indices)):
                if indices[i] != right_chunk_tree.n:
                    l.append(i)
                    r.append(indices[i])
                    d.append(distances[i])
            
            #
            # Update the indexing to reflect the current position in the chunks
            #
            l = (np.array(l, dtype=int) + min( left_chunk.index)).tolist()
            r = (np.array(r, dtype=int) + min(right_chunk.index)).tolist()
    
            #
            # And add to the current list of matches
            #
            l_all = l_all + l
            r_all = r_all + r
            d_all = d_all + d
            gc.collect()
            

    #
    # Get arrays to store the indices and distances in
    #
    store = pd.HDFStore(left)
    n_left  = store.get_storer('primary').nrows
    store.close()

    left_2_right_indices   = -1.0 * np.ones(n_left, dtype=int)
    left_2_right_dists     =        np.ones(n_left, dtype=float)
    
    #
    # Select the best matches indexed by the left frame (going by min distance)
    #
    for i in range(len(l_all)):
        if left_2_right_dists[l_all[i]] > d_all[i]:
            left_2_right_dists[l_all[i]] = d_all[i]
            left_2_right_indices[l_all[i]] = r_all[i]
    
    plt.figure()
    plt.hist(left_2_right_dists, bins=np.linspace(0.0, 0.0002424/2.0, 500))
    plt.title("cross match radius")
    plt.show()
    
    
    #
    # Build the data frame that contains the cross matches from the right, 
    # which will be added to the left
    #
    right_chunk_dtypes = []
    addition = pd.DataFrame()
    right_chunks = pd.read_hdf(right, key="primary", chunksize=r_csize)
    for right_chunk in right_chunks:
        
        #
        # Store some info about the chunk data types to use later
        #
        right_chunk_dtypes = right_chunk.dtypes
        
        #
        # Figure out the max and min indices for the right that are in this chunk
        #
        chunk_min = min(right_chunk.index)
        chunk_max = max(right_chunk.index)
        
        #
        # Figure out the indices to the left catalog which have matches in this chunk
        #
        left_in_chunk  = np.where(np.logical_and(np.greater_equal(left_2_right_indices, chunk_min),
                                                          np.less(left_2_right_indices, chunk_max)))[0]
        
        #
        # Get only the indices to those matches within this chunk
        #
        right_in_chunk = [int(left_2_right_indices[i]) for i in left_in_chunk]
        
        #
        # And get the slice of the chunk corresponding to those matches
        #
        right_chunk_frame = right_chunk.loc[right_in_chunk]
    
        #
        # Update that slice's indexing to correspond to the list of matches
        # a.k.a. the left catalog to append this to
        #
        right_chunk_frame['tempindexcol'] = left_in_chunk
        right_chunk_frame.set_index('tempindexcol', drop=True, inplace=True)
        
        #
        # And use it to add to/update the hdf to append
        #
        addition = right_chunk_frame.combine_first(addition)
    
    
    #
    # Get out the left dataframe
    #
    left_df = pd.read_hdf(left, key="primary") 
    
    #
    # Add rows for unmatched sources and fill those rows with datatype appropriate values
    #
    addition = addition.reindex(np.arange(0, max(left_df.index)+1))
    for i in range(len(addition.columns)):
        if 'bool' in str(right_chunk_dtypes[i]):
            addition[addition.columns[i]].fillna(False, inplace = True)
    addition = addition.infer_objects()
    
    #
    # Rename the columns, adding the suffix
    #
    new_colnames = {}
    for col in addition.columns:
        new_colnames.update({col:(col+sfx)})
    addition.rename(index=int, columns = new_colnames, inplace=True)
    
    #
    # And join the two DataFrames
    #
    left_df = left_df.join(addition, rsuffix="_copy")
    
    #
    # And save the data frame
    #
    left_df.to_hdf(target, key='primary', format='table')    
    
    
    
    
    


def inbin(x, lo, hi, addition):
    good = np.logical_and(np.greater_equal(x, lo),
                          np.less_equal(x, hi))
    good = np.logical_and(good, addition)
    return np.where(good)[0]

def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-((x-mu)**2))/(2*(sigma**2)))

def gauss_sum(x, mu, sigma):
    out = np.zeros(len(x))
    for i in range(len(mu)):
        out = out + gauss(x, mu[i], sigma[i])
    return out


def main():
    """
    PREPARE ANDY'S PHOTO-Z CATALOG FOR USE
    """
    start_time = int(time.time())
    
    def current_time():
        return (int(time.time()) - start_time)
    
    def report(s):
        print("Report: " + str(s))
        print("Current time is " + str(current_time()))
        print("")
        sys.stdout.flush()
        

    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    
    agn_source_raw     = parent + "/cats/CASSANDRA/HSC_WISE_QSOPHOTOZ_v2_1.fits"
    agn_source_veniced = parent + "/cats/raw/agn_apz_veniced.fits"
    agn_source         = parent + "/cats/raw/agnpz.hdf5"
    agn_source_unwise  = parent + "/cats/raw/agn_apz_unwise.hdf"
    hsc_source         = "/scratch/csh4/cats/hsc_25_zdist_f.hdf"
    agn_target         = parent + "/cats/raw/agn_err.hdf"


    PREP = True
    if PREP:
        #
        # VENICE mask the AGN catalog
        #
        report("Venice masking")
        hsc_random_gen.venice_mask(agn_source_raw, agn_source_veniced, overwrite=True,
                                   ra_name='RA_WISE', dec_name='DEC_WISE')
        
        #
        # Turn into HDF
        #
        report("Turning to HDF")
        hsc_random_gen.fits_to_hdf(agn_source_veniced, agn_source, overwrite=True)
        
        #
        # Cross match with HSC 
        #
        report("Cross matching into HSC")
        cm_df(agn_source, hsc_source, agn_source_unwise, l_ran='RA_HSC', l_decn='DEC_HSC', sfx = '_hsc2')
        
        #
        # Add a WISE mask
        #
        report("Adding WISE mask")
        hsc_random_gen.add_wise_mask_column(agn_source_unwise, agn_target,
                                            overwrite=True, ra_name='RA_WISE', 
                                            dec_name='DEC_WISE', use_mp=False)
        
        report("DONE PREPPING")
    
    #
    # Get out results
    #
    #report("Reading out results")
    agn = pd.read_hdf(agn_target, key="primary")
    plt.figure()
    plt.hist2d(agn['G_PSFFLUX_MAG']-agn['W3'], agn['W2']-agn['W3'], bins=[500,500])
    plt.show()
    print(agn.columns)
#
#    type_1 = np.where(np.less_equal(agn['G_KRONFLUX_MAG'] - agn['W2'], 6.7))[0]
#    type_2 = np.where(np.greater(   agn['G_KRONFLUX_MAG'] - agn['W2'], 6.7))[0]
#    
#    t1z = [agn['ZBEST_PEAK'][i] for i in type_1]
#    t2z = [agn['ZBEST_PEAK'][i] for i in type_2]
#    
#    #
#    # Redshift distribution for each type
#    #
#    plt.figure()
#    plt.hist(t1z, bins=np.linspace(0, 5.0, 30), label='Type 1', edgecolor='blue',
#             linewidth=4, histtype='step')
#    plt.hist(t2z, bins=np.linspace(0, 5.0, 30), label='Type 2', edgecolor='red',
#             linewidth=4, histtype='step')
#    plt.xlabel("Redshift")
#    plt.ylabel("Number")
#    plt.legend()
#    plt.show()
#    
#    
#    #
#    # Color color histogram plot
#    #
#    plt.figure()
#    plt.hist2d((agn['G_KRONFLUX_MAG']-agn['W2']), (agn['W1']-agn['W2']),
#               bins=[50,50], cmap='pink_r', norm=colors.PowerNorm(gamma=0.33))
#    plt.colorbar()
#    plt.xlabel("g - W2")
#    plt.ylabel('W1 - W2')
#    plt.axvline(6.7, color='black', linewidth=2)
#    plt.show()
#    
#    #
#    # Cross match on sky plot
#    #
##    good = np.where(np.logical_not(np.isnan(agn['ra_hsc2'])))[0]
##    bad = np.where(np.isnan(agn['ra_hsc2']))[0]
##    
##    good_ra = [[agn['RA_HSC'][i],  agn['ra_hsc2'][i]]  for i in good]
##    good_de = [[agn['DEC_HSC'][i], agn['dec_hsc2'][i]] for i in good]
##    
##    bad_ra = [agn['RA_HSC'][i]  for i in bad]
##    bad_de = [agn['DEC_HSC'][i] for i in bad]
##    
##    plt.figure()
##    for i in range(len(good_ra)):
##        plt.plot(good_ra[i], good_de[i], color='blue')
##    plt.scatter(bad_ra, bad_de, color='red', s=1)
##    plt.show()
#    
#    
#    
#    frankenz_mean  = agn['frankenz_best_hsc2'].tolist()
#    frankenz_stdev = agn['frankenz_std_hsc2'].tolist()
#    andyz_mean  = agn['ZBEST_PEAK'].tolist()
#    andyz_uperr = agn['ZBEST_UPERR'].tolist()
#    andyz_loerr = agn['ZBEST_LOERR'].tolist()
#    andyz_stdev = (np.array(andyz_uperr, dtype=float) + np.array(andyz_loerr, dtype=float))/2.0
#    #specz = chunk['SPECZ'].tolist()
#    #arcturus_ok = np.greater(agn['flag'].tolist(), 0)
#    
#    has_both = np.logical_and(np.greater(andyz_stdev, 0), np.greater(frankenz_stdev, 0))
#    
#    
#    gmag = np.array(agn['G_KRONFLUX_MAG'].tolist(), dtype=float)
#    w2mag = np.array(agn['W2'].tolist(), dtype=float)
#    gmw2 = gmag - w2mag
#    type_1 = np.greater(gmw2, 6.7)
#    
#    zbins = [0.3,  0.5, 0.7]
#    zbin_indices_andyz    = [inbin(andyz_mean,    zbins[i],
#                                   zbins[i+1], has_both) for i in range(len(zbins)-1)]
#    zbin_indices_frankenz = [inbin(frankenz_mean, zbins[i],
#                                   zbins[i+1], has_both) for i in range(len(zbins)-1)]
#    
#    xax = np.linspace(0.0, 3.0, 1000)
#    
#    andyzi_andyz_means       = [[andyz_mean[i] for i in zbin_indices_andyz[j]] for j in range(len(zbins)-1)]
#    andyzi_andyz_stdev       = [[andyz_stdev[i] for i in zbin_indices_andyz[j]] for j in range(len(zbins)-1)]
#    andyzi_frankenz_means    = [[frankenz_mean[i] for i in zbin_indices_andyz[j]] for j in range(len(zbins)-1)]
#    andyzi_frankenz_stdev    = [[frankenz_stdev[i] for i in zbin_indices_andyz[j]] for j in range(len(zbins)-1)]
#    frankenzi_andyz_means    = [[andyz_mean[i] for i in zbin_indices_frankenz[j]] for j in range(len(zbins)-1)]
#    frankenzi_andyz_stdev    = [[andyz_stdev[i] for i in zbin_indices_frankenz[j]] for j in range(len(zbins)-1)]
#    frankenzi_frankenz_means = [[frankenz_mean[i] for i in zbin_indices_frankenz[j]] for j in range(len(zbins)-1)]
#    frankenzi_frankenz_stdev = [[frankenz_stdev[i] for i in zbin_indices_frankenz[j]] for j in range(len(zbins)-1)]
#    
#    andyzi_andyz    = [gauss_sum(xax, andyzi_andyz_means[i], andyzi_andyz_stdev[i]) for i in range(len(zbins)-1)]
#    andyzi_frankenz = [gauss_sum(xax, andyzi_frankenz_means[i], andyzi_frankenz_stdev[i]) for i in range(len(zbins)-1)]
#    frankenzi_andyz    = [gauss_sum(xax, frankenzi_andyz_means[i], frankenzi_andyz_stdev[i]) for i in range(len(zbins)-1)]
#    frankenzi_frankenz = [gauss_sum(xax, frankenzi_frankenz_means[i], frankenzi_frankenz_stdev[i]) for i in range(len(zbins)-1)]
#    
#    #
#    # Simple PDZ plot
#    #
#    f = plt.figure()
#    plt.suptitle("Photo-z PDZ Comparison")
#    f.add_subplot('121')
#    plt.title("Andyz selection")
#    for zbound in zbins:
#        plt.axvline(zbound, color='black', linewidth=1)
#    plt.axhline(0.0, color='black', linewidth=1)
#    for i in range(len(zbins)-1):
#        if i == 0:
#            plt.plot(xax, andyzi_andyz[i], label='andyz', color='blue')
#            plt.plot(xax, andyzi_frankenz[i], label='frankenz', color='red')
#        else:
#            plt.plot(xax, andyzi_andyz[i], color='blue')
#            plt.plot(xax, andyzi_frankenz[i], color='red')
#    plt.legend()
#    f.add_subplot('122')
#    plt.title("Frankenz selection")
#    for zbound in zbins:
#        plt.axvline(zbound, color='black', linewidth=1)
#    plt.axhline(0.0, color='black', linewidth=1)
#    for i in range(len(zbins)-1):
#        plt.plot(xax, frankenzi_andyz[i], color='blue')
#        plt.plot(xax, frankenzi_frankenz[i], color='red')
#    #
#    # redshift comparison result plot
#    #
#    plt.figure()
#    plt.title("Photo-z Mean Comparison, color=g(kron)-W2")
#    
#    both_i = np.where(has_both)[0]
#    has_sz = np.where(np.logical_and(has_both, np.logical_not(np.isnan(agn['SPECZ']))))[0]
#    
#    andyz_means_all = [andyz_mean[i] for i in both_i]
#    frankenz_means_all = [frankenz_mean[i] for i in both_i]
#    gmw2_all = np.array([gmw2[i] for i in both_i], dtype=float)
#    
#    andyz_means_sz = [andyz_mean[i] for i in has_sz]
#    frankenz_means_sz = [frankenz_mean[i] for i in has_sz]
#    
#    
#    plt.scatter(andyz_means_all, frankenz_means_all, cmap='magma', c=gmw2_all, s=0.75, vmin=4, vmax=10)
#    
#    
#    plt.xlabel("Andy's photo-z")
#    plt.show()
#    
#    plt.ylabel("Franken-z")
#    plt.colorbar()
#    
#    plt.scatter(andyz_means_sz, frankenz_means_sz, color='blue', s=1)
#    
#    for zbound in zbins:
#        plt.axvline(zbound, color='black', linewidth=1)
#        plt.axhline(zbound, color='black', linewidth=1)
#    plt.show()
    
if __name__ == "__main__":
    main()
