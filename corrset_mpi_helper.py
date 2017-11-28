#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:11:22 2017

@author: csh4
"""

import pickle
import time
import numpy as np
from mpi4py import MPI
import sys

dir_path = "/scr/depot0/csh4/"
mpi_path=  dir_path + "cats/mpi/holder/h"

start_time = int(time.time())
def current_time():
    return (int(time.time()) - start_time)
    
def two_point_angular_corr_part(ra1, dec1, tree2, bins):
    """
    Does the pair counting for the correlation. 
    
    @params
        ra1, dec1 - One side of the pair counting ra and dec
        tree2     - The other side of the pair counting, as a 
                    sklearn.neighbors.KDTree object
        bins      - The cartesian bins for correlating with
    
    @returns 
        The un-normalized pair counts. Remember to divide by (len(ra1)*len(ra2))
    """
    key1 = np.asarray(ra_dec_to_xyz(ra1, dec1), order='F').T
    return tree2.two_point_correlation(key1, bins)

def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere.
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    ra_to_use = np.array(ra, dtype=float)
    dec_to_use = np.array(dec, dtype=float)
    
    sin_ra = np.sin(ra_to_use * np.pi / 180.)
    cos_ra = np.cos(ra_to_use * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec_to_use * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec_to_use * np.pi / 180.)

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]

def do_job(rank, num):        
    f = 0
    try:
        f = open(mpi_path + str(num), 'rb')
    except FileNotFoundError as e:
        print("File " + mpi_path + str(num) + " not found. e = " + str(e))
        sys.stdout.flush()
        return
    args = pickle.load(f)
    f.close()
    
    coords = args['coords']
    tree   = args['tree']
    bins   = args['bins']
    lenra2 = args['lenra2']      
    lenra1 = args['lenra1']
    path   = args['path']
    
    result = two_point_angular_corr_part(coords[0], coords[1], tree, bins)
    result = np.array(np.diff(result), dtype=float)/(lenra1*lenra2)
    np.save(path, result)
    
    print("code:exit" + str(rank) + "_" + str(num))
    sys.stdout.flush()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #Initialize some values for everyone
    """
    Job list options
    
    "exit" - no more jobs for this process
    n      - number, run this job
    """
    job_list = {}
    total_jobs = 0
    
    if rank == 0:
        for i in range(1, size):
            job_list[i] = i-1
        f = open(mpi_path, 'rb')
        total_jobs = pickle.load(f)['total_jobs']
        print("Total number of jobs = " + str(total_jobs))
        sys.stdout.flush()
    else:
        time.sleep(3)
        
    job_list   = comm.bcast(job_list,   root=0)
    
    #Division of labor
    
    #RANK = 0.  Bourgeoisie
    if rank == 0:
        done = False
        next_job = size - 1
        reqs = {}
        while not done:
            #i.   Check with each proletarian if they have finished, to give 
            #   them more work, as the capitalist has "bought the use of the 
            #   labour-power for a definite period, and he insists on his 
            #   rights" (303)
            for i in range(1, size):
                if i not in reqs:
                    reqs[i] = comm.irecv(source=i, tag=i)
                sys.stdout.flush()
                if reqs[i].test()[0]:           #ii.  If an entry is finished
                    if next_job < total_jobs:   # and there is another job to do
                        job_list[i] = next_job  # assign the job
                        next_job = next_job + 1 # and increase the counter
                                                # and prepare for the result
                        reqs[i] = comm.irecv(source=i, tag=i)
                        
                        print("Job list is " + str(job_list) + " at time " + str(current_time()))
                        sys.stdout.flush()
                    elif job_list[i] == 'exit': # If we're done, do nothing
                        pass    
                    else:                       # If there is not another job,
                        job_list[i] = 'exit'    # give the exit code.
                        print("Job list is " + str(job_list) + " at time " + str(current_time()))
                        sys.stdout.flush()
                    
                    #iii. Send the updated list to the proletarian
                    comm.isend(job_list, dest=i, tag=i) 
                    
            
            
            
            #iv.  Check if all entries are 'exit'
            all_exit = True
            for i in range(1, size):
                all_exit = (all_exit and (job_list[i] == 'exit'))
            
            #v.   Exit if this is the case
            if all_exit:
                done = True
                break
                
            time.sleep(.05)
            

    #RANK != 0. Proletarians
    elif rank != 0:
        jobs_done = []
        done = False
        while not done:
            #i.   Load up the job list and see if we have a job
            job = job_list[rank]
            if job == "exit": #ii.  If we have an exit code, exit
                done = True
            elif job in jobs_done: #iii. If we are waiting for a new job, report
                                   #     that and wait
                req = comm.isend(True, dest=0, tag=rank)
                req.wait()
                req = comm.irecv(source = 0, tag = rank)
                job_list = req.wait()
            else:             #iv.  If we have a job, run it
                jobs_done.append(job)
                do_job(rank, job)
