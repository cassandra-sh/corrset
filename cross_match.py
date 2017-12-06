#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:49:55 2017

Special program for doing cross matches quickly and efficiently

@author: csh4
"""

import numpy as np
import gc
from scipy import spatial
import time
import sys
import psutil
import pandas as pd
import multiprocessing as mp


temp_hdf = "/scr/depot0/csh4/cats/temp"

def file_cross_match(left, right, new, suffix, chunksize, radius=2, verbose=True, 
                     algorithm='closest', right_metric = None, 
                     append = "left", left_ran = 'ra', left_decn = 'dec',
                     right_ran = 'ra', right_decn = 'dec'):
    """
    file_cross_match
    
    Takes two paths to HDF5 files and returns a crossmatch of the two.
    Appends right to left (append='left') or left to right(append='right').
    Chunksize effects right, so always put the larger file as right and
    the smaller as left.
    """
    start_time = int(time.time())
    #Defining a couple in-method helper functions
    def current_time():
        return (int(time.time()) - start_time)
    def report(report_string):
        if verbose:
            sys.stdout.flush()
            time = current_time()
            print("")
            print("--- cross_match.file_cross_match() reporting ---")
            print(report_string)
            print("Time is " + str(time) + " seconds from start. ", end="")
            print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
            print("")
            sys.stdout.flush()
        gc.collect()
    
    #Getting radius and number of chunks
    radius_in_cartesian = 2*np.sin((2*np.pi*radius)/(2*3600*360))
    store = pd.HDFStore(right)
    n_rows = store.get_storer('primary').nrows
    store.close()
    n_chunks = int(n_rows / chunksize) + 1
    
    
    #Send message noting beginning the algorithm.
    report("About to start cross match algorithm")
    
    if append == "left":
        #1. Get the left and
        report("Getting search data (left)")
        left_ra, left_dec = get_ra_dec_file(left, left_ran, left_decn)
        
        #right sides
        report("Getting search data (right)")
        right_ra, right_dec, right_metric_values = [], [], None
        if right_metric != None and algorithm != "closest":
            right_ra, right_dec, right_metric_values = get_ra_dec_file(right,
                                                                       right_ran,
                                                                       right_decn,
                                                                       chunksize=chunksize,
                                                                       met=right_metric)
        else:
            right_ra, right_dec, = get_ra_dec_file(right,right_ran,right_decn,
                                                   chunksize=chunksize)
        a, b, c = ra_dec_to_xyz(right_ra, right_dec)
        right_search_set = np.array([a, b, c]).T
        
        #2. Grow the tree
        report("Growing the tree")
        tree = spatial.KDTree(right_search_set)
        
        #3. Feed to the multiprocessing tree search function
        report("Searching tree")
        indices = tree_query_multi(tree, left_ra, left_dec,
                                   radius_in_cartesian, algorithm,
                                   metric=right_metric_values)
        
        #4. With the indices of the cross match, iterate through the right side
        #   and construct the DataFrame to be attached to the left side.
        report("Going through chunks and building DataFrame to append")
        chunks = pd.read_hdf(right, key="primary", chunksize=chunksize, mode="r+")
        n = 0
        frame_to_add = 0
        left_indices = range(0, len(indices))
        for chunk in chunks:
            #Try to print 10-ish times, rather than n_chunks times
            if n_chunks > 10:
                if n % int(n_chunks/10) == 0: 
                    report("Adding for chunk " + str(n+1) + " of " + str(n_chunks) + ".")
            else:
                report("Adding for chunk " + str(n+1) + " of " + str(n_chunks) + ".")
            chunk_indices = chunk.index
            chunk_max = max(chunk_indices)
            chunk_min = min(chunk_indices)
            
            #Figure out the indices from the cross match which correspond to 
            #this chunk. 
            relevant = np.logical_and(np.less(indices, chunk_max),
                                  np.greater_equal(indices, chunk_min))
            relevant_indices = np.where(relevant)[0]
            relevant_left_indices = [left_indices[i] for i in relevant_indices]
            
            #While chunk.index returns the indexing in the entire HDF file,
            #chunk.iloc[] requires indexing per chunk. Thus, we need to subtract
            #the minimum of the chunk before calling chunk.iloc[]
            relevant_right_indices = [int(indices[i]-chunk_min) for i in relevant_indices]
            
            #Get the rows of those indices from this chunk and keep track of  
            #their cross matched indices.
            attach = chunk.iloc[relevant_right_indices]
            attach["index"] = relevant_left_indices
            #Add to the DataFrame we're going to attach to the left later
            if n == 0:
                frame_to_add = attach
            else:
                frame_to_add = frame_to_add.append(attach, ignore_index=True)
            
            n = n + 1
        
        chunks.close()
        
        #6. Get the left frame to append to and merge them. 
        report("Joining the cross matched catalog")
        left_df = pd.read_hdf(left, key="primary", mode="r+")
        
        #Correct the indexing to match the left side
        frame_to_add.set_index('index', inplace=True)
        
        #Add the suffix to the columns to be added
        names = frame_to_add.columns
        namechange = {}
        for n in names:
            namechange[n] = str(n+suffix)
        frame_to_add.rename(columns=namechange, inplace=True)
        
        #Preserve the datatypes befor merging
        dct = left_df.dtypes.to_dict()
        dct.update(frame_to_add.dtypes.to_dict())
        
        #And merge them. 
        left_df = left_df.join(frame_to_add, rsuffix="___")
        
        #7. This joining fills non matched values with "nan". We need to replace
        #   these values with appropriate values for their data types. 
        new_col_names = left_df.columns
        new_datatypes = left_df.dtypes
        #For each column, will with the appropriate values
        for i in range(0, len(left_df.columns)):
            new_dtype = new_datatypes[i]
            old_dtype = dct[new_col_names[i]]            
            if new_dtype == old_dtype:
                pass
            else:
                fillval = "nan"
                if "int" in str(old_dtype):
                    fillval = -1
                elif "bool" in str(old_dtype):
                    fillval = False
                left_df[new_col_names[i]].fillna(fillval, inplace=True)
        
        #Cast everything as the original datatype
        left_df = left_df.apply(lambda x: x.astype(dct[x.name]))
        
        #8. And save, finally. 
        report("Saving the new hdf")
        left_df.to_hdf(new, key='primary', format="table")
        
    elif append == "right":    
        #Writing a right sided append with a metric other than closeness needs a 
        #different approach, so for now I've left that functionality out.
        #After all, I don't need it, for now.
        if algorithm == "bayesian" or algorithm == "lowest" or algorithm == "circle":
            raise NotImplemented("bayesian, lowest and circle algorithms are",
                                 "not implemented for append=right")
        
        #1. Get the right side ra and dec to build the tree.
        report("Getting tree data and growing tree (right)")
        right_ra, right_dec = get_ra_dec_file(right, right_ran, right_decn, 
                                              chunksize=chunksize)
        a, b, c = ra_dec_to_xyz(right_ra, right_dec)
        right_search_set = np.array([a, b, c]).T
        tree = spatial.KDTree(right_search_set)
        
        #2. Get the left side ra and dec values
        report("Getting tree search data (left)")
        left_ra, left_dec = get_ra_dec_file(left, left_ran, left_decn)
        
        #3. Send to the multiprocessing cross match search method
        report("Searching tree")
        indices = tree_query_multi(tree, left_ra, left_dec,
                                   radius_in_cartesian, algorithm)
        
        #4. Get out the whole left DataFrame for appending to the right
        #   and make sure to rename the DataFrame's columns with the suffix added
        report("Preparing to save results")
        left_df = pd.read_hdf(left, key="primary", mode="r+")        
        names = left_df.columns
        namechange = {}
        for n in names:
            namechange[n] = str(n+suffix)
        left_df.rename(columns=namechange, inplace=True)        
        dct = left_df.dtypes.to_dict()
             
        
        #5. With the indices of the cross match, iterate through the right side
        #   and append the cross matched left rows to the right, then save.
        report("Going through chunks and saving results")
        chunks = pd.read_hdf(right, key="primary", chunksize=chunksize, mode="r+")
        n = 0
        frame_to_add = 0
        left_indices = range(0, len(indices))
        for chunk in chunks:            
            #Try to print 10-ish times, rather than n_chunks times
            if n_chunks > 10:
                if n % int(n_chunks/10) == 0: 
                    report("Adding for chunk " + str(n+1) + " of " + str(n_chunks) + ".")
            else:
                report("Adding for chunk " + str(n+1) + " of " + str(n_chunks) + ".")
                
            chunk_indices = chunk.index
            chunk_max = max(chunk_indices)
            chunk_min = min(chunk_indices)
            
            
            #Figure out the indices from the cross match which correspond to 
            #this chunk. 
            relevant = np.logical_and(np.less(indices, chunk_max),
                                  np.greater_equal(indices, chunk_min))
            relevant_indices = np.where(relevant)[0]
            relevant_left_indices = [left_indices[i] for i in relevant_indices]
            
            #While chunk.index returns the indexing in the entire HDF file,
            #chunk.iloc[] requires indexing per chunk. Thus, we need to subtract
            #the minimum of the chunk before calling chunk.iloc[]
            relevant_right_indices = [indices[i]-chunk_min for i in relevant_indices]
            
            #Preserve the data types of the chunk (only bother doing this once)
            if n == 0: dct.update(chunk.dtypes.to_dict())
            
            #Get the rows of those indices from the left DataFrame and attach to
            #the chunk. 
            attach = left_df.iloc[relevant_left_indices]
            attach["index"] = relevant_right_indices
            attach.set_index("index", inplace=True)
            chunk = chunk.join(attach, rsuffix=suffix)
            
            #Fix the 'nan' values we just added for non matches when joining to
            #appropriate values for each column's data type
            new_col_names = chunk.columns
            new_datatypes = chunk.dtypes
            for i in range(0, len(chunk.columns)):
                new_dtype = new_datatypes[i]
                old_dtype = dct[new_col_names[i]]
                
                if new_dtype == old_dtype:
                    pass
                else:
                    fillval = "nan"
                    if "int" in str(old_dtype):
                        fillval = -1
                    elif "bool" in str(old_dtype):
                        fillval = False
                    chunk[new_col_names[i]].fillna(fillval, inplace=True)
            
            #Cast everything as the original datatype
            chunk = chunk.apply(lambda x: x.astype(dct[x.name]))
            
            #Save to the new path
            if n == 0:
                chunk.to_hdf(new, key="primary", format="table", mode="w")
            else:
                chunk.to_hdf(new, key="primary", format="table", mode="a", 
                             append=True)
            n = n + 1
        chunks.close()
        
    else:
        raise ValueError("append is not 'left' or 'right'")

def tree_query_multi(tree, ra, dec, rad, algorithm, metric=None):
    """
    Do a tree query with multiprocessing.
    
    Really more of a manager function for tree_search_part.
    
    This might be better implemented with mp.Array(), however this is questionable. 
    
    returns the indices
    """
    n_jobs = mp.cpu_count()
    n_ra = len(ra)
    
    #List of slices for the left indices, to divide up the jobs
    group_bounds = np.linspace(0, n_ra, n_jobs+1, dtype=int)
    group_pairs = [[group_bounds[i], group_bounds[i+1]] for i in range(0, n_jobs)]
    
    #Turn counter and list of processes
    turn = mp.Value('i', 0)
    processes = []
    
    #The thing to return eventually, shape=left, points to right
    indices_empty = np.zeros(n_ra, dtype=int)
    indices_empty.fill(-1)
    indices_empty = indices_empty.tolist()
    
    indices = mp.Manager().list(indices_empty)
    
    
    for i in range(0, n_jobs):
        args = (tree, ra, dec, group_pairs[i], algorithm,
                rad, metric, indices, turn, i)
        processes.append(mp.Process(target=tree_search_part, args=args))
        processes[-1].start()
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    
    
    return indices[:]

def tree_search_part(tree, ra, dec, group, algorithm, rad, metric, indices, turn, n):
    """
    Multiprocessing helper function.
    
    With the given inputs, do a search of the tree and save it to the path.
    """
    #Get the ordered slice of the ra and dec and prepare to search the tree with it
    good_ra, good_dec = ra[group[0]:group[1]], dec[group[0]:group[1]]
    a, b, c = ra_dec_to_xyz(good_ra, good_dec)
    key = np.array([a, b, c]).T
    
    #Do a search with the appropriate algorithm.
    if algorithm == "bayesian":
        raise NotImplemented("algorithm == " + algorithm +
                             " is not implemented")
    elif algorithm == "lowest":
        #Get the indices (shape = left, points to right)
        indices_to_add = tree.query_ball_point(key, rad)
        for i in range(0, len(indices_to_add)):
            #Set the entry to -1 if no entries are found
            if len(indices_to_add[i]) == 0:
                indices_to_add[i] = -1
            #Otherwise get the metric specified and take the lowest value
            #In most cases, this is the most negative magnitude, corresponding
            #to the brightest neighbor within rad. 
            else:
                metrics = [metric[j] for j in indices_to_add[i]]
                indices_to_add[i] = indices_to_add[i][min(range(len(metrics)),
                                                      key=metrics.__getitem__)]
                indices_to_add[i] = indices_to_add[i] + group[0]
    elif algorithm == "closest":
        distances, indices_to_add = tree.query(key, k=1, distance_upper_bound=rad)
        
        for i in range(len(indices_to_add)):
            if indices_to_add[i] == tree.n:
                indices_to_add[i] = -1
    else:
        raise ValueError("algorithm == " + algorithm +
                         " is not valid for multiprocessed tree search")
        
    #We lost the indexing from taking a slice to search the tree, so we need to
    #re-adjust the indices to align with the full dataframe
    indices_to_add = np.array(indices_to_add, dtype=int).tolist()
    
    #Add to the shared array
    indices[group[0]:group[1]] = indices_to_add


def get_ra_dec_file(filename, ra_name, dec_name, chunksize=None, met=None):
    """
    Simply read everything from the file, in chunks, potentially getting the 
    metric as well, as quickly as possible.
    """
    ra, dec, metric = [], [], []
    f = pd.read_hdf(filename, key="primary", chunksize=chunksize, mode="r+")
    if met == None:
        if chunksize == None:
            ra, dec = np.array(f[ra_name]), np.array(f[dec_name])
        else:
            ra, dec = np.array([], dtype=float), np.array([], dtype=float)
            for c in f:
                ra = np.append(ra, c[ra_name])
                dec = np.append(dec, c[dec_name])
    else:
        if chunksize == None:
            ra, dec, metric = np.array(f[ra_name]), np.array(f[dec_name]), np.array(f[met])
        else:
            ra, dec = np.array([], dtype=float), np.array([], dtype=float)
            metric  = np.array([], dtype=float)
            for c in f:
                ra = np.append(ra, c[ra_name])
                dec = np.append(dec, c[dec_name])
                metric = np.append(metric, c[met])
    if met == None:
        return ra, dec
    else:
        return ra, dec, metric
    
    
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

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]
