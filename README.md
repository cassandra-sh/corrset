Goals before leaving:

-write a "get started" guide for how to run a correlation through the pipeline 

-turn existing scripts into OOP modules if at all possible 

-add more documentation, especially on things which are not used as much anymore 



Readme:

pre_qualifier -> qualifier -> mpi_prep -> mpi_run -> mpi_read -> jackknifer
    
Each part has the following purpose:
        
[pre pre qualifier stuff]

        -cms             - cross matches a bunch of stuff, applies flags, transfers between
                           HDFs, DataFrames, and fits files, handles Venice and MANGLE masks
                           
        -hsc_random_gen  - generates a random catalog for the HSC footprint using various
                           masks as well as the getFullAreaHealPixels.py script provided by
                           the HSC weak lensing folks
                           
        -file_manager    - some helper functions for maintaining directories
        
        -fits2hdf        - turns memory inefficient astropy fits tables into pandas compatible
                           HDFs. The problem is that astropy fits tables don't chunk read well,
                           and that is an absolutely necessary functionality for this project.
        
pre_qualifier

        implements any and all quality cuts determined appropriate for the 
        correlation inputs and saves the results in a way where the qualifier
        can read them.
        
        The saved outputs of pre_qualifier must be plugged into qualifier.ini
        for qualifier to read the outputs
    
qualifier

        -sorts data catalog by pixel, redshift
        
        -gets out pixel and redshift distribution of data (not used anymore)
        
        -normalizes random catalog to large scale distribution of data
        
        -sorts random catalog by pixel
    
        relies on hdf_sorter and qualifier.ini 
        
        hdf_sorter contains a bunch of memory conservative sorting and 
        normalizing functions for use in qualifier
        
        qualifier.ini CONTAINS ALL OF THE META INFORMATION FOR THIS ROUND OF
        CORRELATIONS! INCLUDING HOW MANY CORRELATIONS, WHAT THE ANGULAR BINS ARE
        WHAT THE FILE PATHS ARE, EVERYTHING!
    
mpi_prep

        -Doesn't actually use MPI anymore
        
        -Using the output of qualifier, prepares pixel-pixel pairs of information
        between the relevant data sets and pickles them for mpi_run to process.
    
mpi_run

        -Uses a multiprocessing scheme to unpickle, run, and repickle, pixel-pixel
        correlation terms. Divide and conquer. 
        
        relies on mp_manager, which manages the Python multiprocessing queue
    
mpi_read

        -reads out all of the correlation information into CountMatrix objects
    
        relies on CountMatrix, which contains count information
    
jackknifer

        -Handles the CountMatrix objects to plot a crapload of stuff about the
        correlation
        
        relies on quick_figs (ironically named) which is basically a bunch of
        scripts to display different results. This script is mostly hard coded,
        as it does a lot of the same basic pre_qualifier stuff, even using some
        fits files
        
        also relies on camb_model_new for generating models


Dependencies

-astroML

-pycamb

-astropy

-pandas

-pymangle

-healpy

