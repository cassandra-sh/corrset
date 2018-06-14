#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpi_run.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

qualifier -> mpi_prep -> * mpi_run * -> mpi_read -> jackknifer

Program Description:
    File to be queued through a slurm or other computing cluster, which 
    utilizes MPI.
    
    Input (pickled):
        dictionary = {'out'         :'filename_out',
                      'ra1'         : [ra values]  ,
                      'dec1'        : [dec values] ,
                      'ra2'         : [ra values]  ,
                      'dec2'        : [dec values] ,
                      'angular_bins': [bin values]  }
    
    Steps:
        1. Load ra1, dec1
            1.1 Convert angular coordinates to cartesian coordinates
        2. Grow KDTree from ra1, dec1
        3. Query the tree for correlation signal
            3.1 Convert angular bins to cartesian bins
            3.2 Query for pair counts within each distance
            3.3 Take difference between each bin for signal
        4. Save result at path specified by 'out'
    
    Inputs are mediated by a manager, who takes rank 0. 
    
Directory info:
    
         parent
         │
         ├── mpi_run.py
         │
         ├── mpi_jobs/       #inputs
         │   ├── jj
         │   ├── j0
         │   ├── j1
         │   └── ...
         │
         └── mpi_outs/       #where outputs will be saved
             ├── out0.npy
             ├── out1.npy
             └── ...
"""

from sklearn.neighbors import KDTree
from mpi4py            import MPI

import mp_manager
import time
import sys
import pickle
import numpy as np
import os

def current_time(start_time):
    return (int(time.time()) - start_time)

def gen_dirs():
    """
    Make a directory in the directory of this python file to store the outputs
    """
    pass

def ang2cart_bins(D):
    """
    Converts angular bins to cartesian bins
    """
    D_to_use = np.array(D, dtype=float)
    return 2 * np.sin(0.5 * D_to_use * np.pi / 180.)
        
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


def calc_corr(tree, ra, dec, angular_bins):
    """
    With the tree, coordinates, and bins, compute the correlation signal. 
    """
    key = np.asarray(ra_dec_to_xyz(ra, dec), order='F').T
    bins = ang2cart_bins(angular_bins)
    return np.array((tree.two_point_correlation(key, bins)), dtype=float)

def grow_tree(ra, dec):
    """
    Grow a sklearn.neighbors KDTree from the ra and dec 
    (ra and dec are first turned into cart coords)
    """
    return KDTree(np.asarray(ra_dec_to_xyz(ra,dec), order='F').T)

def save_result(path, result):
    """
    Save the given numpy array to the path
    """
    #Save the file
    f = open(path, mode='wb')
    np.save(f, result)
    f.flush()
    f.close()
    
    #Wait to make sure the file was actually saved. 
    #Wait up to 5 seconds. If it fails to save, throw some error. 
    timeout = 0.0
    while not os.path.isfile(path):
        time.sleep(0.5)
        timeout = timeout + 0.5
        if timeout > 5.0:
            raise IOError("Failed to save " + path)
        

def do_job(target, out_dir):
    """
    Does a job from a target input file and then renames the input file to mark
    it as finished
    
    (thus if interrupted, you can pick up where you last ended)
    """
    
    if os.path.isfile(target):
        target_file = open(target, 'rb')
        args = pickle.load(target_file)
        target_file.close()
        tree = grow_tree(args['ra1'], args['dec1'])
        corr = calc_corr(tree, args['ra2'], args['dec2'], args['angular_bins'])
        save_result((out_dir + args['out']), corr)
        os.rename(target, target + "_f")
    else:
        print("Job " + target + " not available.")
        sys.stdout.flush()
        return
        

def ensure_job(target, out_dir):
    """
    Ensure every job is done. Can do as a second pass. Double checks, first that
    the job itself has finished and renamed itself, and second that it has
    produced the expected result file. 
    
    Run in case there is a chance that jobs were dropped or unfinished. 
    (do_job was called but results were not saved for whatever reason)
    """
    
    if os.path.isfile(target):
        print(target.split("/")[-1]  + " was missed. Doing.")
        do_job(target, out_dir)
    elif os.path.isfile(target+"_f"):
        target_file = open(target+"_f", 'rb')
        args = pickle.load(target_file)
        target_file.close()
        if not os.path.isfile(out_dir + args['out']):
            print(target.split("/")[-1] + " had no output. Doing.")
            do_job(target+"_f", out_dir)
        else:
            #print(target.split("/")[-1] + " is good.")
            return
    else:
        print(target.split("/")[-1]  + " not found!!")
        

def proletarian(comm, rank, job_suf=""):
    """
    Routine for rank > 0 cores to run.
    
    Routine:
        Look at job list for currently assigned job
            If the job is to exit, end the process
            If the job is old [finished], report finished and wait for a new job
            If the job is new, run it
            
    Needs to be passed a job number by the bourgeoisie through comm.bcast
    """
    time.sleep(10.00)
    
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    job_dir = parent + ("/mpi_jobs"+job_suf+"/")
    out_dir = parent + ("/mpi_outs"+job_suf+"/")
    
    job_list =  {}
    job_list =  comm.bcast(job_list,   root=0)
    jobs_done = []
    done = False
    while not done:
        #i.   Load up the job list and see if we have a job
        target = job_list[rank]
        if target == "exit": #ii.  If we have an exit code, exit
            done = True
        elif target in jobs_done: #iii. If we are waiting for a new job, report
                                  #     that and wait
            req = comm.isend(True, dest=0, tag=rank)
            req.wait()
            req = comm.irecv(source = 0, tag = rank)
            job_list = req.wait()
        else:             #iv.  If we have a job, run it
            jobs_done.append(target)
            do_job((job_dir+"j"+str(target)), out_dir)
            ensure_job((job_dir+"j"+str(target)), out_dir)


def bourgeoisie(comm, size, job_suf="", short_first = False):
    """
    Routine for the rank = 0 core to run
    """
    #Get directory information for loading/saving
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    job_dir = parent + ("/mpi_jobs"+job_suf+"/")
    out_dir = parent + ("/mpi_outs"+job_suf+"/")
    
    #Get out the meta information, and solve for job order (largest first)
    f = open((job_dir+"jj"), 'rb')
    job_info = pickle.load(f)
    total_jobs = job_info['n_jobs']
    job_order = np.argsort(job_info['time_index']).tolist()
    f.close()
    
    #Read out total number
    print("Total number of jobs = " + str(total_jobs))
    sys.stdout.flush()
    
    #Develop job list and broadcast to proletarians
    job_list = {}
    if short_first:
        for n in range(1, size):
            job_list.update({str(n+1) : job_order.pop(0)})
    else:
        for n in range(1, size):
            job_list.update({str(n+1) : job_order.pop()})
    job_list = comm.bcast(job_list, root=0)
    
    #Make some progress trackers
    start_time = int(time.time())
    times = []
    usage = []
    progress = []
    jobs_going = size-1
    jobs_done = 0
    reqs = {}
    
    #Start waiting for job finishes to give updated task
    done = False
    while not done:
        #i.   Check with each proletarian if they have finished. 
        #     Send them the updated job list. 
        for i in range(1, size):
            if i not in reqs:
                reqs[i] = comm.irecv(source=i, tag=i)
            sys.stdout.flush()
            if reqs[i].test()[0]:           #ii.  If an entry is finished
                if len(job_order) > 0   :   # and there is another job to do
                    if short_first:
                        job_list[i] = job_order.pop()  # assign the job
                    else:
                        job_list[i] = job_order.pop(0)
                    jobs_done = jobs_done + 1
                    reqs[i] = comm.irecv(source=i, tag=i)                        
                    print("Job list is " + str(job_list) + " at time " + str(current_time(start_time)))
                    print("Using " + str(jobs_going) + " of " + str(size-1) + " workers")
                    sys.stdout.flush()
                elif job_list[i] == 'exit': # If we're done, do nothing
                    pass    
                else:                       # If there is not another job,
                    job_list[i] = 'exit'    # give the exit code.
                    jobs_done = jobs_done + 1
                    jobs_going = jobs_going - 1
                    print("Job list is " + str(job_list) + " at time " + str(current_time(start_time)))
                    print("Using " + str(jobs_going) + " of " + str(size-1) + " workers")
                    sys.stdout.flush()
                
                #iii. Send the updated list to the proletarian
                comm.isend(job_list, dest=i, tag=i) 
                
        #Note the current resource usage.
        progress.append(jobs_done)
        usage.append(jobs_going)
        times.append(current_time(start_time))
        
        #iv.  Check if all entries are 'exit'
        if jobs_going == 0:
            done = True
            #Save the resource usage record. 
            np.save(out_dir + "_going.npy", np.array(usage, dtype=int))
            np.save(out_dir + "_done.npy", np.array(progress, dtype=int))
            np.save(out_dir + "_times.npy", np.array(times, dtype=int))
            
        time.sleep(1.0)

def main(job_suf="", use_mpi = True, short_first = False):
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            bourgeoisie(comm, size, job_suf = job_suf, short_first=short_first)
        else:
            proletarian(comm, rank, job_suf = job_suf)
            
    else:    
        start_time = int(time.time())
        print("Running code locally. Finding directory information")
        print("Time is " + str(current_time(start_time)))
        directory = os.path.dirname(os.path.realpath(__file__))
        parent = os.path.dirname(directory)
        job_dir = parent + ("/mpi_jobs"+job_suf+"/")
        out_dir = parent + ("/mpi_outs"+job_suf+"/")
        
        print("Loading jobs")
        print("Time is " + str(current_time(start_time)))
        f = open((job_dir+"jj"), 'rb')
        job_info = pickle.load(f)
        job_order = np.argsort(job_info['time_index']).tolist()
        f.close()
        
        print("Sending job order to multiprocessing queue")
        print("Time is " + str(current_time(start_time)))
        args = []
        if short_first:
            while len(job_order) > 0:
                args.append(((job_dir+"j"+str(job_order.pop())), out_dir,))
        else:
            while len(job_order) > 0:
                args.append(((job_dir+"j"+str(job_order.pop(0))), out_dir,))
        
        print("Spawning processes to do jobs.")
        print("Time is " + str(current_time(start_time)))
        mp_manager.queue(do_job, args, cpu_throttle = False)
        
        print("Spawning processes to ensure all jobs are done.")
        print("Time is " + str(current_time(start_time)))
        mp_manager.queue(ensure_job, args, cpu_throttle = False)
        
        print("Done.")
        print("Time is " + str(current_time(start_time)))
        
if __name__ == "__main__":
    main(use_mpi = False)
