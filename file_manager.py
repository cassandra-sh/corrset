#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_manager.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

Program Description:
    Some functions for reading ini/config files and maintaining the catalog
    directory
"""

import configparser
import os
import shutil
import glob

def read_config_file(path):
    """
    Assuming a config file of a given format, read out the parameters as a
    dictionary. Format:    
    _______________________________
    
    #comments
    #where line begins with
    #number sign
    
    NAME : value
    NAME : space separated values
    _______________________________
    
    You have to hardcode later to interpret each parameter. Just gets the string
    after each = sign
    
    @params  the file path to read
    @returns the dictionary of the read parameters. You have to split and
             interpret them
    """
    config = configparser.ConfigParser()
    config.optionxform=str
    config.read(path)
    dct = {}
    for section in config.sections():
        dct = {**dct, **config[section]}
    return dct

def ensure_dir(file_path):
    """
    Ensure the given directory exists
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def build_directory(path):
    """
    Builds the directory 
            
         ├── corrset
         │   ├── file_manager.py
         │   └── ...
         ├── cats
         │   ├── raw           #Convenience folder for holding input stuff
         │   ├── matrices      #Output of mpi_read,  input of jackknifer
         │   ├── temp          #Used by qualifier for intermediate products
         │   └── ready         #Output of qualifier, input of mpi_prep 
         ├── mpi_jobs          #Output of mpi_prep,  input of mpi_run
         └── mpi_outs          #Output of mpi_run,   input of mpi_read
    
    For storing intermediate products in the chain
    
    qualifier -> mpi_prep -> mpi_run -> mpi_read -> jackknifer
    """
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir  = os.path.dirname(current_dir)
    
    ensure_dir(parent_dir + "/cats/")
    ensure_dir(parent_dir + "/cats/raw/")
    ensure_dir(parent_dir + "/cats/matrices/")
    ensure_dir(parent_dir + "/cats/temp/")
    ensure_dir(parent_dir + "/cats/ready/")
    ensure_dir(parent_dir + "/mpi_jobs/")
    ensure_dir(parent_dir + "/mpi_outs/")

        
def empty_dir(path):
    """
    Empties the directory at the given path. 
    
    Makes a new directory if path does not lead to a directory. 
    """
    if os.path.exists(path):
        files = glob.glob(path + "/*")
        for f in files:
            os.remove(f)
    else:
        os.makedirs(path)
