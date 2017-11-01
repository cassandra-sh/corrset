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
    start_time = int(time.clock())
    #Defining a couple in-method helper functions
    def current_time():
        return (int(time.clock()) - start_time)
    def report(report_string):
        if verbose:
            sys.stdout.flush()
            time = current_time()
            print("")
            print("--- cross_match.file_cross_match() reporting ---")
            print(report_string)
            print("Time is " + str(float(time/60.)) + " minutes from start. ", end="")
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
        #1. Get the left side and build the tree to search with.
        report("Getting tree data and growing tree (left)")
        left_ra, left_dec = get_ra_dec_file(left, left_ran, left_decn)
        a, b, c = ra_dec_to_xyz(left_ra, left_dec)
        left_search_set = np.array([a, b, c]).T
        tree = spatial.KDTree(left_search_set)
        
        #2. Get the right side
        report("Getting tree search data (right)")
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
            
        #3. Feed to the multiprocessing tree search function
        report("Searching tree")
        indices = tree_query_multi(tree, right_ra, right_dec,
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
            if n % int(n_chunks/10) == 0: 
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
        
        #5. Correct the indexing to match the left side and attach.
        chunks.close()
        report("Appending DataFrame")
        frame_to_add.set_index('index')
        left_df = pd.read_hdf(left, key="primary", mode="r+")
        left_df = left_df.join(frame_to_add, rsuffix=suffix)
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
        report("Preparing to save results")
        left_df = pd.read_hdf(left, key="primary", mode="r+")
        
        #5. With the indices of the cross match, iterate through the right side
        #   and append the cross matched left rows to the right, then save.
        report("Going through chunks and saving results")
        chunks = pd.read_hdf(right, key="primary", chunksize=chunksize, mode="r+")
        n = 0
        frame_to_add = 0
        left_indices = range(0, len(indices))
        for chunk in chunks:
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
            
            #Get the rows of those indices from the left DataFrame and attach to
            #the chunk. 
            attach = left_df.iloc[relevant_left_indices]
            attach["index"] = relevant_right_indices
            attach.set_index("index")
            chunk = chunk.join(attach, rsuffix=suffix)
            
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
    group_bounds = np.linspace(0, n_ra, n_jobs+1, dtype=int)
    group_pairs = [[group_bounds[i], group_bounds[i+1]] for i in range(0, n_jobs)]
    
    turn = mp.Value('i', 0)
    processes = []
    
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
        
    return list(indices)

def tree_search_part(tree, ra, dec, group, algorithm, rad, metric, indices, turn, n):
    """
    Multiprocessing helper function.
    
    With the given inputs, do a search of the tree and save it to the path.
    """
    good_ra, good_dec = ra[group[0]:group[1]], dec[group[0]:group[1]]
    a, b, c = ra_dec_to_xyz(good_ra, good_dec)
    key = np.array([a, b, c]).T
    if algorithm == "bayesian":
        raise NotImplemented("algorithm == " + algorithm +
                             " is not implemented")
    elif algorithm == "lowest":
        indices_to_add = tree.query_ball_point(key, rad)
        for i in range(0, len(indices_to_add)):
            if len(indices_to_add[i]) == 0:
                indices_to_add[i] = -1
            else:
                metrics = [metric[i] for i in indices_to_add[i]]
                indices_to_add[i] = indices_to_add[i][min(range(len(metrics)),
                                                      key=metrics.__getitem__)]
    elif algorithm == "closest":
        indices_to_add, distances = tree.query(key)
    else:
        raise ValueError("algorithm == " + algorithm +
                         " is not valid for multiprocessed tree search")
    
    indices[group[0]:group[1]] = indices_to_add
#    
#    written = False
#    while not written:
#        if turn.value == n: #Wait for your turn...
#            for i in indices_to_add:
#                indices.append(i) #Have to use .append() on this ListProxy object
#            turn.value = turn.value + 1
#            written = True
#        else:
#            time.sleep(3)


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