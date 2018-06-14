#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp_manager.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

Handles running a multiprocessing queue
"""

import multiprocessing as mp
import gc
import psutil
import time

def queue(targs, args, n_thread_override = None, verbose=True,
          cpu_throttle = False):
    """
    Manage a queue of processes that can be done in parallel.
    
    @params
        targs - one method or a list of methods of length len(args)
        args  - tuples of the method arguments corresponding to targs
        
        n_thread_override - run on this many theads instead of one per cpu
        verbose           - print the fraction of jobs completed every 30 secs
        
        cpu_throttle      - add processes based on cpu utilization - if it is
                            less than half, add a process. n_thread_override or
                            the number of cores is used as the minimum number
                            of jobs at a given time. I recommend leaving 
                            n_thread_override as None if using cpu_throttle
                            
                            WARNING: you have to frontload the heavy jobs. Only
                            do this if the jobs have a low cpu impact or the
                            ones with a large cpu impact come first. 

                            basically, enter true if you want to be 
                            2fast2furious. Pretty much not worth using. 
    """
    if type(targs) == list:
        pass
    else:
        targs = [targs for arg in args]
    
    if verbose:
        print("queue starting with " + str(len(args)) + " jobs.")
    
    
    n_cores = mp.cpu_count()
    
    if n_thread_override != None:
        n_cores = n_thread_override
        
    target_list = [targs[i] for i in range(len(args))]
    argument_list = [args[i] for i in range(len(args))]
    procs = []
    for i in range(len(args)):
        procs.append(mp.Process(target=target_list[i],args=argument_list[i]))
    
    if len(procs) < n_cores:
        for proc in procs:
            proc.daemon = True
            proc.start()
        for proc in procs:
            proc.join()
    elif cpu_throttle == False:
        jobs  = []
        for core in range(n_cores):
            procs[-1].daemon = True
            procs[-1].start()
            jobs.append(procs.pop())
        
        time_elapsed = 0
        while len(procs) > 0:
            for j in range(len(jobs)):
                if not jobs[j].is_alive():
                    jobs[j] = procs.pop()
                    jobs[j].daemon = True
                    jobs[j].start()
                if len(procs) == 0:
                    break
            time.sleep(1)
            time_elapsed = time_elapsed + 1
            
            if verbose:
                if time_elapsed % 30 == 0:
                    print("Time elapsed is " + str(time_elapsed))
                    print("Jobs finished = " + str(len(args)-len(procs)))
                    print("Jobs remaining = " + str(len(procs)))
            
        for proc in jobs:
            proc.join()
    
    elif cpu_throttle == True:
        jobs  = []
        
        for core in range(n_cores):
            procs[-1].daemon = True
            procs[-1].start()
            jobs.append(procs.pop())
        
        time.sleep(10.0)
        
        extra = []
        time_elapsed = 0
        mode = 0
        counter = 0
        start_time = time.time()
        while len(procs) > 0:
            
            # In cpu_throttle mode 0, maintain exactly the specified number of
            # threads.
            if mode == 0:
                for j in range(len(jobs)):
                    if len(procs) == 0:
                        break
                    elif not jobs[j].is_alive():
                        jobs[j] = procs.pop()
                        jobs[j].daemon = True
                        jobs[j].start()
                    
                time.sleep(1)
                time_elapsed = time_elapsed + 1
                percent = psutil.cpu_percent()
                
                # For roughly half the time, monitor the usage.
                #
                # If cpu usage drops below 50% for more than 10 seconds, while
                # the number of threads is that specified, go to mode 1
                if percent < 50.0 and time_elapsed % 30 < 15:
                    less = True
                    for i in range(10):
                        for j in range(len(jobs)):
                            if len(procs) == 0:
                                break
                            elif not jobs[j].is_alive():
                                jobs[j].terminate()
                                jobs[j] = procs.pop()
                                jobs[j].daemon = True
                                jobs[j].start()
                        time.sleep(1)
                        time_elapsed = time_elapsed + 1
                        less = less and psutil.cpu_percent() < 50.0
                            
                    if less:
                        mode = 1
                        counter = time_elapsed
                        print("Going to mode 1 at time "+str(int(time_elapsed)))
                        
                elif time_elapsed % 30 == 0 and verbose:
                    
                    for proc in extra:
                        if not proc.is_alive():
                            proc.terminate()
                            extra.remove(proc)
                    
                    jobs_running = 4 + len(extra)
                    
                    if time_elapsed % 300 == 0:
                        print("cpu throttled mode 0")
                        print("Time elapsed is " + str(int(time_elapsed)))
                        print("CPU usage is " + str(percent) + "%")
                        print("Jobs running at the moment: " + str(n_cores))
                        print("Jobs finished = " + str(len(args)-len(procs)))
                        print("Jobs remaining = " + str(len(procs)))
                        print("")
                    gc.collect()
            
            # In mode 1, add jobs as long as the cpu percent is below 50
            elif mode == 1:                
                cpu_percent = psutil.cpu_percent()
                vm = psutil.virtual_memory()
                mem_percent = 1 - vm.available / vm.total
                time_elapsed = time.time() - start_time
                
                if time_elapsed > counter and verbose:
                    counter = counter + 30
                    n_old = 0
                    for j in range(len(jobs)):
                        if jobs[j].is_alive():
                            n_old = n_old + 1
                    
                    jobs_running = n_old + len(extra)
                    print("Time elapsed is " + str(int(time_elapsed)))
                    print("CPU usage is " + str(percent) + "%")
                    print("Jobs running at the moment: " + str(jobs_running))
                    print("Jobs finished = " + str(len(args)-len(procs)))
                    print("Jobs remaining = " + str(len(procs)))
                    print("")
                    gc.collect()
      
                elif (cpu_percent < 80.0 and
                      mem_percent < 50.0 and
                      jobs_running < 100):
                    if len(procs) > 15:
                        for i in range(10):
                            proc = procs.pop()
                            proc.daemon = True
                            proc.start()
                            extra.append(proc)
                        time.sleep(0.5)
                    else:
                        proc = procs.pop()
                        proc.daemon = True
                        proc.start()
                        extra.append(proc)
            
                for proc in extra:
                    if not proc.is_alive():
                        proc.terminate()
                        extra.remove(proc)
                gc.collect()
    
        for proc in jobs:
            proc.join()
        

def test(n):
    time.sleep(n)
    print("Done!")

def main():
    args = []
    for num in range(20):
        args.append((num,))
    
    queue(test, args)
        
if __name__ == '__main__':
    main()
