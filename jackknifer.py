#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jackknifer.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

qualifier -> mpi_prep -> mpi_run -> mpi_read -> * jackknifer *


Program Description:
    jackknifer will (first draft) take a set of count matrices and meta info
    and use it to compute the correlation signal.
    
    Steps:
        1. Figure out the result desired from meta.hdf (qualifier.py output)
        
        2. Read the relevant count matrix objects
        
        3. Add up the signal, normalize by the number of objects
        
        4. Compute the Landy-Szalay auto or cross correlation

Directory info:        
    
         parent
         │
         ├── corrset
         │   ├── jackknifer.py   #This file
         │   └── ...
         └── cats
             ├── raw
             ├── matrices        #Inputs for jackknifer.py go here
             └── ready
         
"""

from itertools           import combinations
#from scipy.optimize      import curve_fit
from matplotlib          import rc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import CountMatrix       as cm
import pandas            as pd
import numpy             as np
import camb_model_new
import file_manager
import matplotlib
import quick_figs
#import gen_pairs
import pickle
import shutil
import psutil
import camb
import glob
import time
import sys
import os
import gc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def current_time(start_time):
    return (int(time.time()) - start_time)



def report(report_value, start_time):
    sys.stdout.flush()
    time = current_time(start_time)
    print("")
    print("**jackknifer.py reporting: ", end="")
    print(str(report_value))
    print("Time is " + str(time) + " seconds from start. ", end="")
    print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
    sys.stdout.flush()
    gc.collect()
    
def load_cms(i, zbin, job_suf=""):
    """
    Load the count matrices associated with a given term in the correlation
    
    After having run mpi_read, CountMatrix objects have been prepared in the
    /cats/matrices_suf/ directory.
    
    @params
        i  The row number of the correlation to load (same index as meta hdf)
        job_suf - if this job has a unique suffix
        
    @returns
        the 3(auto)-4(cross) correlation terms, in each z bin. Thus, the 
        dimensionality of returned result is (3/4, n_zbins)
    """
    #Getting count matrix directory 
    directory = os.path.dirname(os.path.realpath(__file__))
    parent    = os.path.dirname(directory)
    cm_dir    = parent + ("/cats/matrices"+job_suf+"/")
    
    #Getting meta information
    meta_st  = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    
    pix_pops = meta_st.get('pop')
    corrs    = meta_st.get('corrs')      
    zhists   = meta_st.get('zhists')     
    zedges   = meta_st.get('zedges')            
    other    = meta_st.get('other')
    abins    = np.array(meta_st.get('abins')['abins'].tolist(), dtype=float)
    nside    = other['nside'][0]
    meta_st.close()

    #Get information associated with this correlation
    corr_type   = corrs['type'][i]
    
    #Autocorrelation routine
    if corr_type == "auto":
        
        #Determine file names
        d_pref = ("C"+str(i)+"D0")
        
        #Get the redshift distribution information to include in the
        #count matrices
        z_hist, z_edges = [], []
        if d_pref in zhists:
            z_hist  = zhists[d_pref]
            z_edges = zedges[d_pref]
        
        
        #Get population specific to this z bin
        dpop = pix_pops[("C"+str(i)+"D0_zbin"+str(zbin))]
        rpop = pix_pops[("C"+str(i)+"R0_zbin"+str(zbin))]
        
        #Determine names for the count matrices
        ddcm = ("C"+str(i)+"D0D0_zbin"+str(zbin)+"mat")
        drcm = ("C"+str(i)+"D0R0_zbin"+str(zbin)+"mat")
        rrcm = ("C"+str(i)+"R0R0_zbin"+str(zbin)+"mat")
        
        #Build count matrices
        ddcm = cm.CountMatrix(nside = nside, filename = (cm_dir+ddcm),
                              tabname = ddcm, abins = abins,
                              pop1= dpop.values, pop2 = dpop.values,
                              z_hist1 = z_hist, z_edges1 = z_edges)
        
        drcm = cm.CountMatrix(nside = nside, filename = (cm_dir+drcm),
                              tabname = drcm, abins = abins,
                              pop1 = dpop.values, pop2 = rpop.values)
        
        rrcm = cm.CountMatrix(nside = nside, filename = (cm_dir+rrcm),
                              tabname = rrcm, abins = abins,
                              pop1 = rpop.values, pop2 = rpop.values)
        
        gc.collect()

        #LOAD CountMatrix objects
        ddcm.load()
        gc.collect()
        
        drcm.load()
        gc.collect()
        
        rrcm.load()        
        gc.collect()
            
        #Add to the list
        return ddcm, drcm, rrcm
            
                    
    #Cross correlation routine
    elif corr_type == "cross":
        
        #Determine file names
        d1_pref = ("C"+str(i)+"D1")
        d2_pref = ("C"+str(i)+"D2")
        
        
        #Get the redshift distribution information to include in the
        #count matrices
        z_hist1, z_edges1, z_hist2, z_edges2 = [], [], [], []
        if d1_pref in zhists:
            z_hist1  = zhists[d1_pref]
            z_edges1 = zedges[d1_pref]
            z_hist2  = zhists[d2_pref]
            z_edges2 = zedges[d2_pref]
            
        
        #Get population specific to this z bin
        d1pop = pix_pops[("C"+str(i)+"D1_zbin"+str(zbin))]
        r1pop = pix_pops[("C"+str(i)+"R1_zbin"+str(zbin))]
        d2pop = pix_pops[("C"+str(i)+"D2_zbin"+str(zbin))]
        r2pop = pix_pops[("C"+str(i)+"R2_zbin"+str(zbin))]
        
        #Determine names for the count matrices
        d1d2cm = ("C"+str(i)+"D1D2_zbin"+str(zbin)+"mat")
        d1r2cm = ("C"+str(i)+"D1R2_zbin"+str(zbin)+"mat")
        d2r1cm = ("C"+str(i)+"D2R1_zbin"+str(zbin)+"mat")
        r1r2cm = ("C"+str(i)+"R1R2_zbin"+str(zbin)+"mat")
        
        #Build count matrices
        d1d2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1d2cm),
                                tabname = d1d2cm, abins = abins,
                                pop1= d1pop.values, pop2= d2pop.values,
                                z_hist1=z_hist1, z_hist2 = z_hist2,
                                z_edges1=z_edges1, z_edges2=z_edges2)
        
        d1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1r2cm),
                                tabname = d1r2cm, abins = abins,
                                pop1= d1pop.values, pop2= r2pop.values)
        
        d2r1cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d2r1cm),
                                tabname = d2r1cm, abins = abins,
                                pop1= d2pop.values, pop2= r1pop.values)
        
        r1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+r1r2cm),
                                tabname = r1r2cm, abins = abins,
                                pop1= r1pop.values, pop2= r2pop.values)
        

            
        #LOAD CountMatrix objects
        d1d2cm.load()
        d1r2cm.load()
        d2r1cm.load()
        r1r2cm.load()
     
            
        #Add to the list
        return d1d2cm, d1r2cm, d2r1cm, r1r2cm
                

def auto_signal(cms, n=16, dump=False):
    """
    Given the count matrices in one z bin for one autocorrelation, compute the
    signal and inverse variance weighted jackknife resample error
    
    @returns signal, errors
        lists of float, length = number of angular bins
    """
    dd, dr, rr = cms[0], cms[1], cms[2]
    parts, areas = rr.partition()

    #
    # Get every combination with one area removed 
    #
    parts_tupled = [tuple(parts[i]) for i in range(len(parts))]
    perms = set()
    for perm in combinations(parts_tupled, 1):
        perms.add(perm)
    parts = []
    for perm in perms:
        part_to_add = []
        for tup in perm:
            for index in tup:
                part_to_add.append(index)
        parts.append(part_to_add)
              
    
    rr_nocut  = rr.get_signal()
    signal = (dd.get_signal() - 2*dr.get_signal() + rr.get_signal()) / rr.get_signal()

    dd_s = [dd.get_signal(cut=part) for part in parts]
    dr_s = [dr.get_signal(cut=part) for part in parts]
    rr_s = [rr.get_signal(cut=part) for part in parts]
    
    sys.stdout.flush()
    gc.collect()
    
    if dump:
        dd = None
        dr = None
        rr = None
        gc.collect()
        
    sig_cut = [((dd_s[i]-2*dr_s[i]+rr_s[i])/rr_s[i]) for i in range(len(dd_s))]
    errs   = np.sqrt(np.sum(np.array([(rr_s[i]/rr_nocut)*((sig_cut[i] - signal)**2) for i in range(len(dd_s))]), axis=0))
    
    gc.collect()
    
    return signal, errs

def cross_signal(cms, n=16, dump=False):
    """
    Given the count matrices in one z bin for one crosscorrelation, compute the
    signal and inverse variance weighted jackknife resample error
    
    @returns signal, errors
        lists of float, length = number of angular bins
    """
    d1d2, d1r2, d2r1, r1r2 = cms[0], cms[1], cms[2], cms[3]
    parts, areas = r1r2.partition()

    #
    # Get every combination with one area removed 
    #
    parts_tupled = [tuple(parts[i]) for i in range(len(parts))]
    perms = set()
    for perm in combinations(parts_tupled, 1):
        perms.add(perm)
    parts = []
    for perm in perms:
        part_to_add = []
        for tup in perm:
            for index in tup:
                part_to_add.append(index)
        parts.append(part_to_add)
             
    r1r2_nocut  = r1r2.get_signal()
    signal = (d1d2.get_signal() - d2r1.get_signal() - d1r2.get_signal() + r1r2.get_signal()) / r1r2.get_signal()
       
    d1d2_s = [d1d2.get_signal(cut=part) for part in parts]
    d1r2_s = [d1r2.get_signal(cut=part) for part in parts]
    d2r1_s = [d2r1.get_signal(cut=part) for part in parts]
    r1r2_s = [r1r2.get_signal(cut=part) for part in parts]
    
    if dump:
        d1d2 = None
        d1r2 = None
        d2r1 = None
        r1r2 = None
        gc.collect()
    
    sig_cut = [((d1d2_s[i]-d2r1_s[i]-d1r2_s[i]+r1r2_s[i])/r1r2_s[i]) for i in range(len(d1d2_s))]
    errs   = np.sqrt(np.sum(np.array([(r1r2_s[i]/r1r2_nocut)*((sig_cut[i] - signal)**2) for i in range(len(d1d2_s))]), axis=0))
    
    return signal, errs


def signals(i, corrs, nz, job_suf=""):
    """
    Get the signal and error in each redshift bin associated with the given
    correlation term (given by index)
    
    @params
        i     - row number of the correlation term from the meta hdf to use
        corrs - the "corrs" component of the meta hdf, to figure out if auto or
                cross
    
    @returns
        the signal and error for the correlation, in shape (n_zbins, 2), where
        the 0 term is the signal and the 1 term is the error. 
    """
    if corrs['type'][i] == 'auto':
        to_ret = []
        for j in range(nz):
            cms = load_cms(i, j, job_suf=job_suf)
            to_ret.append(auto_signal(cms))
            cms = []
            gc.collect()
        return to_ret
    else:
        to_ret = []
        for j in range(nz):
            cms = load_cms(i, j, job_suf=job_suf)
            to_ret.append(cross_signal(cms))
            cms = []
            gc.collect()
        return to_ret
    
def dither(dat, interval=0.01):
    return dat + interval*(0.5 - np.random.ranf(len(dat)))

def norm(x, y):
    return x, y/(np.trapz(y, x=x))
    
def main(job_suf="", save=True, load=False, email=True, log=True, justplotcorr=False):
    start_time = int(time.time())
    report("Beginning jackknifer.main()", start_time)
    
    #
    # Getting output directory, clearing CountMatrix output directory 
    #
    directory = os.path.dirname(os.path.realpath(__file__))
    parent    = os.path.dirname(directory)
    
    #
    # Getting meta information
    #
    meta_st  = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    
    corrs    = meta_st.get('corrs')
    names    = corrs['str'].tolist()
    zbins    = meta_st.get('zbins')['zbins'].tolist()
    nz       = len(zbins)-1
    abins    = np.array(meta_st.get('abins')['abins'].tolist(), dtype=float)
    
    meta_st.close()
    
    #
    # Either generate new results or get old ones
    #
    all_signals = []
    all_errors  = []
    savename = parent + "/cats/jackknife"+job_suf+".pickle"
    if load:
        f = open(savename, 'rb')
        out = pickle.load(f)
        all_signals, all_errors = out[0], out[1]
        f.close()
    else:
        for i in range(len(corrs)):
            report(("Reading signal for signal " + str(i)), start_time)
            signals_out = signals(i, corrs, nz, job_suf=job_suf)
            
            signal  = [signals_out[j][0] for j in range(len(signals_out))]
            errors  = [signals_out[j][1] for j in range(len(signals_out))]
            
            all_signals.append(signal)
            all_errors.append(errors)
            
            gc.collect()
            
        if save:
            out=[all_signals, all_errors]
            f = open(savename, 'wb')
            pickle.dump(out, f)
            f.close()
    
    
    #
    # Develop a color scheme for plotting, with one unique color per correlation
    #
    report("Plotting basic results", start_time)
    color_options = ['dimgrey', 'green', 'r', 'b', 'm', 'k']
    color_dict = {}
    for name in names:
        if name in color_dict:
            pass
        else:
            color_dict[name] = color_options.pop()
    
    
    #
    # Get the cosmology information set up
    #
    params, results = 0,0
    params = camb.CAMBparams()
    params.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    params.InitPower.set_params(ns=0.965)
    results = camb.get_results(params)
    results.calc_power_spectra()
    
    
    def ang2comoving(angle, redshift):
        comoving_dist = results.comoving_radial_distance(redshift)
        return comoving_dist*np.tan(np.array(angle)*np.pi/180.0)
    

    
    
    #
    # Figure out cosmological distance scales
    #
    zbins = np.array(zbins, dtype=float)
    a_bin_middles = (abins[1:] + abins[:-1]) / 2
    z_bin_middles = (zbins[1:] + zbins[:-1]) / 2
    xlabels = []
    xvalues = []
    for z_mid in z_bin_middles:
        xlabels.append(("(Mpc) scale at z = " + str(z_mid)))
        xvalues.append(ang2comoving(a_bin_middles, z_mid))
  
    #
    # Plot the main result
    #
    ax0 = None
    ax  = None
    
    matplotlib.rcParams.update({'font.size': 13})  
    fig = plt.figure(figsize=(16,8,))
    for z in range(len(zbins)-1):
        if z == 0:
            ax0 = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(z+1)))
            ax = ax0
        else:
            ax = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(z+1)),
                                 sharey=ax0)# , sharex=ax0)        
        
        #
        # Plot each correlation and errorbars
        #
        
        
        for n in range(len(corrs)):
            sig =  all_signals[n][z]
            err = all_errors[n][z]
            plt.errorbar(xvalues[z]*(1.0 + n*0.05), sig, yerr=err, ms=6.0,
                         fmt='o', label=(str(names[n])),
                         color=color_dict[names[n]], capsize=4.0, elinewidth=1.5,
                         capthick=1.5)   
            
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        
        axl = ax.axis()
        axl_y = [axl[2], axl[3]]
        
        plt.xlabel(xlabels[z])
        ax.grid('on')
        plt.title("Redshift bin = ["+str(zbins[z])+", "+str(zbins[z+1])+")", y=1.2)

        #
        # Get the legend
        #
        if z == 0:
            fig.subplots_adjust(bottom=0.25, top=0.8)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, shadow=False, ncol=1)
        
        #
        # Adjust the axis
        #
        factor = 3.0 
        plt.gca().set_ylim([min(axl_y)/factor,max(axl_y)*1.5])
        
        #
        # Add angular scale on top
        #
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xvalues[z])
        ax2.set_xticklabels(np.around(a_bin_middles*60,decimals=2))
        ax2.set_xlabel("Angle (arcmin)")        
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_xaxis().set_tick_params(which='minor', size=0)
        ax2.get_xaxis().set_tick_params(which='minor', width=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=-45 ) 
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    plt.tight_layout()


    if justplotcorr:
        return

    #
    # Save a picture
    #
    outfile_pdf = directory + "/output_jackknifer.pdf"
    fig.savefig(outfile_pdf)

    #
    # Save the results to a log directory
    #    
    if log:
        logfiles = glob.glob(parent+"/jackknife_logs/*/")
        nums = [0]
        for logfile in logfiles: nums.append(int(logfile.split("/")[-2]))
        file_manager.ensure_dir(parent+"/jackknife_logs/"+str(max(nums)+1)+"/")
        for file in files:
            name = file.split('/')[-1]
            shutil.copyfile(file, (parent+"/jackknife_logs/"+str(max(nums)+1)+"/"+name))
    
    
    #
    # Print a ton of other plots out
    #
    
    
    #
    # AGN PDZs (pseudo and real) by type and redshift bin, precision by redshift and color
    #
    agn_a_pdzs_x = []
    agn_a_pdzs_y = []
    
    agn_1_pdzs_x = []
    agn_1_pdzs_y = []
    
    agn_2_pdzs_y = []
    agn_2_pdzs_x = []
    
    agn_sz_pdzs_y = []
    agn_sz_pdzs_x = []
    
    gal_pdzs_x = []
    gal_pdzs_y = []
    
    gal_szo_pdzs_x = []
    gal_szo_pdzs_y = []
    
    PDZ = False
    if PDZ or not load:
        plt.figure(num=1, figsize=(10,8))
        for z in range(len(zbins)-1):
            plt.subplot(221+z)
            report(("Plotting AGN PDZ for zbin " + str(z)), start_time)
            zmin = zbins[z]
            zmax = zbins[z+1]
            
            plt.title("AGN PDZs for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
                
            xf, yf, xr, yr, zmua, zsia, imaga  = quick_figs.agn_pdz(zmin=zmin, zmax=zmax)
            xf, yf = norm(xf,yf)
            xr, yr = norm(xr,yr)
            
            plt.figure(1)
            plt.plot(xf, yf, label='AGN - all types, pseudo', color='black', linestyle=":")
            plt.plot(xr, yr, label='AGN - all types, real', color='black')
            
            agn_a_pdzs_x.append(xr)
            agn_a_pdzs_y.append(yr)
            
            xf, yf, xr, yr, zmu2, zsi2, imag2 = quick_figs.agn_pdz(type=2, zmin=zmin, zmax=zmax)
            xf, yf = norm(xf,yf)
            xr, yr = norm(xr,yr)
            
            plt.figure(1)
            plt.plot(xf, yf, label='AGN - Type 2, pseudo', color='red', linestyle=":")
            plt.plot(xr, yr, label='AGN - Type 2, real', color='red')
            
            agn_2_pdzs_x.append(xr)
            agn_2_pdzs_y.append(yr)
            
            xf, yf, xr, yr, zmu1, zsi1, imag1 = quick_figs.agn_pdz(type=1, zmin=zmin, zmax=zmax)
            xf, yf = norm(xf,yf)
            xr, yr = norm(xr,yr)
            
            plt.figure(1)
            plt.plot(xf, yf, label='AGN - Type 1, pseudo', color='blue', linestyle=":")
            plt.plot(xr, yr, label='AGN - Type 1, real', color='blue')
            
            agn_1_pdzs_x.append(xr)
            agn_1_pdzs_y.append(yr)
            
            xf, yf, xr, yr, zmu1, zsi1, imag1 = quick_figs.agn_pdz(type=3, zmin=zmin, zmax=zmax)
            xf, yf = norm(xf,yf)
            xr, yr = norm(xr,yr)
            
            plt.figure(1)
            plt.plot(xr, yr, label='AGN - specz only', color='green')
            
            agn_sz_pdzs_x.append(xr)
            agn_sz_pdzs_y.append(yr)
            
            
            plt.figure(1)
            plt.axvline(zmin, color='black', linewidth=1)
            plt.axvline(zmax, color='black', linewidth=1)
            plt.grid('on')
            plt.legend()
            plt.xlabel("Redshift")
            plt.ylabel("Probability")
            plt.axis([0.0, 3.0, 0.0, 6.5])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        plt.tight_layout()
            
        for z in range(len(zbins)-1):
            PHOTOZ_PRECISION = False
            if PHOTOZ_PRECISION:
                report(("Plotting AGN photo-z precision for zbin " + str(z)), start_time)
                plt.figure(figsize=(10,8))
                plt.subplot("121")
                quick_figs.contour_from_hist(dither(zsi1), zmu1, color='blue', lab='Type 1', bins=[60,20], levs=[0.1, 0.3, 0.6])
                quick_figs.contour_from_hist(dither(zsi2), zmu2, color='red', lab='Type 2', bins=[60,20], levs=[0.1, 0.3, 0.6])
                plt.xlabel("Photo-z standard deviation")
                plt.ylabel("Photo-z best")
                plt.legend()
                xmin, xmax, ymin, ymax = plt.axis()
                plt.axis([0.0, 0.5, ymin, ymax])
                plt.subplot("122")
                quick_figs.contour_from_hist(dither(zsi1), imag1, color='blue', lab='Type 1', bins=[60,30], levs=[0.1, 0.3, 0.6])
                quick_figs.contour_from_hist(dither(zsi2), imag2, color='red', lab='Type 2', bins=[60,30], levs=[0.1, 0.3, 0.6])
                plt.xlabel("Photo-z standard deviation")
                plt.ylabel("Photo-z imag")
                plt.suptitle("AGN Photo-z color/precision for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
                xmin, xmax, ymin, ymax = plt.axis()
                plt.axis([0.0, 0.5, ymin, ymax])
                plt.show()
                    
                plt.figure(figsize=(10,8))
                plt.subplot("121")
                plt.scatter(dither(zsi1), zmu1, color='blue', s=1, label='Type 1')
                plt.scatter(dither(zsi2), zmu2, color='red', s=1, label='Type 2')
                plt.xlabel("Photo-z standard deviation")
                plt.ylabel("Photo-z best")
                plt.legend()
                xmin, xmax, ymin, ymax = plt.axis()
                plt.axis([0.0, 0.5, ymin, ymax])
                plt.subplot("122")
                plt.scatter(dither(zsi1), imag1, color='blue', s=1)
                plt.scatter(dither(zsi2), imag2, color='red', s=1)
                plt.xlabel("Photo-z standard deviation")
                plt.ylabel("Photo-z imag")
                plt.suptitle("AGN Photo-z color/precision for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
                xmin, xmax, ymin, ymax = plt.axis()
                plt.axis([0.0, 0.5, ymin, ymax])
                plt.show()
            
        
        #
        # galaxy pseudo PDZs by redshift bin, precision by redshift and color
        # 
        plt.figure(figsize=(10,8))
        for z in range(len(zbins)-1):
            plt.subplot(221+z)
            
            ##Doing all the galaxies with pseudo pdzs from photoz
            report(("Plotting galaxy PDZ for zbin " + str(z)), start_time)
            plt.title("Galaxy pseudo-PDZ for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
            zmin = zbins[z]
            zmax = zbins[z+1]
            
            x, y, zmu, zsi, imag = quick_figs.gal_pseudo_pdz(zmin=zmin, zmax=zmax)
            x, y = norm(x,y)
            
            gal_pdzs_x.append(x)
            gal_pdzs_y.append(y)
            plt.plot(x,y, color='green')
            

            ##Making pseudo pdz for the szo only sample
            df = pd.read_hdf(corrs['D0'][5], key='primary')
            zlist = np.array(df['redshift'].values)
            z_good = zlist[np.logical_and(zlist>zbins[z], zlist<zbins[z+1])]
            
            
            xrange = x
            y = quick_figs.gauss_sum(xrange, z_good, 0.01*np.ones(len(z_good)))
            x, y = norm(x, y)
            
            gal_szo_pdzs_x.append(x)
            gal_szo_pdzs_y.append(y)
            plt.plot(x,y, color='dimgrey')
            
            
            plt.axvline(zmin, color='black', linewidth=1)
            plt.axvline(zmax, color='black', linewidth=1)
            plt.grid('on')
            plt.xlabel("Redshift")
            plt.ylabel("Probability")
            plt.axis([0.0, 3.0, 0.0, 6.5])
        
            
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        plt.tight_layout()
            
        for z in range(len(zbins)-1):
            PHOTOZ_PRECISION = False
            if PHOTOZ_PRECISION:
                report(("Plotting galaxy photo-z precision for zbin " + str(z)), start_time)
                plt.figure(figsize=(10,8))
                plt.subplot("121")
                plt.hist2d(zsi, zmu, bins=[600,15], cmap='pink_r', norm=colors.LogNorm())
                plt.colorbar()
                plt.gca().set_xscale("log", nonposx='clip')
                plt.ylabel("Photo-z best")
                plt.xlabel("Photo-z standard deviation")

                plt.subplot("122")
                plt.hist2d(zsi, imag, bins=[600,400], cmap='pink_r', norm=colors.LogNorm())
                plt.colorbar()
                plt.gca().set_xscale("log", nonposx='clip')
                plt.xlabel("Photo-z standard deviation")
                plt.ylabel("Photo-z imag")
                plt.suptitle("Galaxy Photo-z color/precision for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
            
                axl = plt.gca().axis()
                axl_x = [axl[0], axl[1]]
                axl_y = [axl[2], axl[3]]
                plt.axis([min(axl_x),max(axl_x),max(axl_y), 22.0])
                
                plt.show()
        
        
        pdz_savename = parent + "/cats/jackknife_PDZS_"+job_suf+".pickle"
        if save:
            out = [agn_a_pdzs_x,
                   agn_a_pdzs_y,
                   agn_1_pdzs_x,
                   agn_1_pdzs_y,
                   agn_2_pdzs_x,
                   agn_2_pdzs_y,
                   agn_sz_pdzs_x,
                   agn_sz_pdzs_y,
                   gal_pdzs_x,
                   gal_pdzs_y,
                   gal_szo_pdzs_x,
                   gal_szo_pdzs_y]
            f = open(pdz_savename, 'wb')
            pickle.dump(out, f)
            f.close()
        
        
        #
        # galaxy real PDZs for GAMA09H by redshift bin, precision by redshift and color
        #     
        GAMA09H = False
        if GAMA09H:
            for z in range(len(zbins)-1):
                report(("Plotting galaxy PDZ in GAMA09H for zbin " + str(z)), start_time)
                zmin = zbins[z]
                zmax = zbins[z+1]
                
                plt.figure(figsize=(10,8))
                plt.title("Galaxy PDZs for z = ["+str(zbins[z])+", "+str(zbins[z+1])+") in GAMA09H")
                xf, yf, xr, yr = quick_figs.gal_pdz(zmin=zmin, zmax=zmax)
                xf, yf = norm(xf,yf)
                xr, yr = norm(xr,yr)
                plt.plot(xf,yf, color='green', linestyle=':', label='Pseudo-PDZ')
                plt.plot(xr,yr, color='green', label='Real-PDZ')
                plt.axvline(zmin, color='black', linewidth=1)
                plt.axvline(zmax, color='black', linewidth=1)
                plt.grid('on')
                plt.xlabel("Redshift")
                plt.ylabel("Number of Galaxies")
                plt.legend()
                plt.show()
    
    
    pdz_savename = parent + "/cats/jackknife_PDZS_"+job_suf+".pickle"
    if load:
        f = open(pdz_savename, 'rb')
        out = pickle.load(f)
        agn_a_pdzs_x  = out[0]
        agn_a_pdzs_y  = out[1]
        agn_1_pdzs_x  = out[2]
        agn_1_pdzs_y  = out[3]
        agn_2_pdzs_x  = out[4]
        agn_2_pdzs_y  = out[5]
        agn_sz_pdzs_x = out[6]
        agn_sz_pdzs_y = out[7]
        gal_pdzs_x    = out[8]
        gal_pdzs_y    = out[9]
        gal_szo_pdzs_x= out[10]
        gal_szo_pdzs_y= out[11]
        f.close()
                
    COLORPLOTS = False
    if COLORPLOTS:
        report("Making colorplots ", start_time)
        #
        # Color-color plots
        #
        for z in range(len(zbins)-1):
            quick_figs.colorcolor_zbin(zmin=zbins[z], zmax=zbins[z+1])
        
        #
        # Color versus redshift
        #
        quick_figs.colorz(zmin=0.0, zmax=3.0)
     
    
    #
    # MODEL STUFF
    #
    
    #
    # 0. Make lists to store model results in
    #
    bias_pdf_x     = [[] for n in range(6)]
    bias_pdf_y     = [[] for n in range(6)]
    bias_bests     = [[] for n in range(6)]
    bias_pdf_names = [[] for n in range(6)]
    model_angles   = [[] for n in range(6)]
    model_dists    = [[] for n in range(6)]
    model_values   = [[] for n in range(6)]
    
    bias0 = 0
    
    #
    # 1. Make or load every model.
    #
    if not load:
        
        chis = []
        
        for z in range(len(zbins)-1):
            chis.append([])
            for n in range(6):
                
                report(("Making model for z bin " + str(z) + " and corr " + str(n)), start_time)
                
                #
                # 1.1 Get the signal and error associated with this correlation
                #
                sig =  all_signals[n][z]
                err = all_errors[n][z]
                
                #
                # 1.15 Figure out what of this signal we are going to use in fitting (clipping small angles)
                #
                n_to_cut = 3
                
                sig_to_use = sig[n_to_cut:]
                err_to_use = err[n_to_cut:]
                abi_to_use = a_bin_middles[n_to_cut:]
                
                #
                # 1.2 Get the PDZs associated with this correlation
                #
                pdz2_x, pdz2_y = gal_pdzs_x[z], gal_pdzs_y[z]
                
                pdz1_x, pdz1_y = [], []
                if n == 0: #GALAXY-GALAXY
                    pdz1_x, pdz1_y = gal_pdzs_x[z], gal_pdzs_y[z]
                elif n == 1: #ALL TYPE AGN
                    pdz1_x, pdz1_y = agn_a_pdzs_x[z], agn_a_pdzs_y[z]
                elif n == 2: #TYPE 1 AGN
                    pdz1_x, pdz1_y = agn_1_pdzs_x[z], agn_1_pdzs_y[z]
                elif n == 3: #TYPE 2 AGN
                    pdz1_x, pdz1_y = agn_2_pdzs_x[z], agn_2_pdzs_y[z]
                elif n == 4:
                    pdz1_x, pdz1_y = agn_sz_pdzs_x[z], agn_sz_pdzs_y[z]
                elif n == 5:
                    pdz1_x, pdz1_y = gal_szo_pdzs_x[z], gal_szo_pdzs_y[z]
                    pdz2_x, pdz2_y = gal_szo_pdzs_x[z], gal_szo_pdzs_y[z]
                    
                
                #
                # 1.3 Generate the actual model
                #
                xmod, ymod, best, xbias, ybias, chisq = camb_model_new.corr_bias(abi_to_use, sig_to_use, err_to_use,
                                                                                 pdz1_x, pdz1_y, pdz2_x, pdz2_y)
                true_best = 0
                if n == 0 or n == 5:
                    bias0 = best
                    true_best = best
                else:
                    true_best = (best**2) / bias0
                
                #
                # 1.35 Print the model results
                #
                print("zbin: ", z, " and corr ", names[n])
                print("best bias is ", true_best)
                print("chi square is ", chisq)
                chis[z].append(chisq)
                print("")





                #
                # 1.4 Adjust angles to physical units
                #
                z_mid = ( zbins[z] + zbins[z+1] ) / 2
                xmod_comoving = ang2comoving(xmod, z_mid)
                print(z_mid)
                print(xmod)
                print(xmod_comoving)
                
                #
                # 1.5 Add the results to the list
                #
                bias_pdf_x[n].append(xbias)
                bias_pdf_y[n].append(ybias)
                bias_bests[n].append(best)
                bias_pdf_names[n].append(str(names[n])+" zbin " + str(z))
                model_angles[n].append(xmod)
                model_dists[n].append(xmod_comoving)
                model_values[n].append(ymod)
                
                #
                # 1.6 Save all results
                #
                if save:
                    mod_savename = parent + "/cats/jackknife_MODELS_"+job_suf+".pickle"
                    out = [bias_pdf_x,
                           bias_pdf_y,
                           bias_bests,
                           bias_pdf_names,
                           model_angles,
                           model_dists,
                           model_values]
                    f = open(mod_savename, 'wb')
                    pickle.dump(out, f)
                    f.close()
        
        print(chis)
        print("chi squares: (row = z bins cols = types")
        for c in range(len(names)):
            print(str(names[c]), end="")
        print("")
        for z in range(len(chis)):
            print("zbin = ["+str(zbins[z])+", "+str(zbins[z+1])+")", end="  ")
            for c in range(len(chis[z])):
                print(chis[z][c], end="   ")
            print("")

    #
    # 1.7 Or load all results
    #      
    elif load:
        mod_savename = parent + "/cats/jackknife_MODELS_"+job_suf+".pickle"
        f = open(mod_savename, 'rb')
        out = pickle.load(f)
        f.close()
        
        bias_pdf_x     = out[0]
        bias_pdf_y     = out[1]
        bias_bests     = out[2]
        bias_pdf_names = out[3]
        model_angles   = out[4]
        model_dists    = out[5]
        model_values   = out[6]
    
    
    def plotwhich(l = range(6), baseline=True):
        fig = plt.figure(figsize=(16,8,))
        for z in range(len(zbins)-1):
            if z == 0:
                ax0 = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(z+1)))
                ax = ax0
            else:
                ax = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(z+1)), sharey=ax0)        
            if baseline:
                plt.plot(xvalues[z], all_signals[0][z], color='black', label='Galaxy-galaxy autocorrelation signal')
            for n in l:
                sig =  all_signals[n][z]
                err = all_errors[n][z]
                xmod_comoving = model_dists[n][z]
                ymod = model_values[n][z]
                best = bias_bests[n][z]
                plt.errorbar(xvalues[z]*(1.0 + n*0.05), sig, yerr=err, ms=6.0,
                             fmt='o', label=(str(names[n])), 
                             color=color_dict[names[n]], capsize=4.0, elinewidth=1.5,
                             capthick=1.5)
                plt.plot(xmod_comoving, np.array(ymod)*(best**2), color=color_dict[names[n]],
                         label=(str(names[n])+" CAMB model (best fit)"))
                plt.plot(xmod_comoving, np.array(ymod), color=color_dict[names[n]],
                         label=(str(names[n])+" CAMB model (unbiased)"), linestyle=":")
            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')
            plt.xlabel(xlabels[z])
            
            axl = ax.axis()
            axl_y = [axl[2], axl[3]]
            
            ax.grid('on')
            plt.title("Redshift bin = ["+str(zbins[z])+", "+str(zbins[z+1])+")", y=1.2)
            if z == 0:
                fig.subplots_adjust(bottom=0.25, top=0.8)
                plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), fancybox=True, shadow=False, ncol=3)
            factor = 3.0 
            plt.gca().set_ylim([min(axl_y)/factor,max(axl_y)*1.5])
            ax2 = ax.twiny()
            ax2.set_xscale('log')
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xvalues[z])
            ax2.set_xticklabels(np.around(a_bin_middles*60,decimals=2))
            ax2.set_xlabel("Angle (arcmin)")        
            ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_xaxis().set_tick_params(which='minor', size=0)
            ax2.get_xaxis().set_tick_params(which='minor', width=0)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=-45 )
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        plt.tight_layout()
    
    
    # 2. Make various plots of correlation signal/model

    # 2.1 Plot only the galaxy-galaxy autocorrelation
    plotwhich(l=[0], baseline=False)
    
    # 2.1 Plot only the galaxy-galaxy autocorrelation
    plotwhich(l=[0, 5], baseline=False)
    
    # 2.2 Plot galaxy-galaxy autocorrelation and all agn cross correlation
    plotwhich(l=[0,1], baseline=False)
    
    # 2.25 Plot galaxy-galaxy autocorrelation and all agn cross correlation
    plotwhich(l=[4,1], baseline=False)
    
    # 2.3 Plot only both types of AGN 
    plotwhich(l=[2,3], baseline=False)
    
    # 2.4 Plot both types of AGN with galaxy-galaxy autocorrelation [as black line]
    plotwhich(l=[2,3], baseline=True)
    
    
    #
    # 3. Plot all the bias fits in a nice 3x4 subplot plot
    #
    f, axarr = plt.subplots(1, len(zbins)-1)
    
    for z in range(len(zbins)-1):
        ax = axarr[z]
        best0 = bias_bests[0][z]
        for n in range(6):
            ax.step((bias_pdf_x[n][z]**2)/best0,
                    bias_pdf_y[n][z],
                    color=color_dict[names[n]], label=names[n], linewidth=2)
        ax.set(xlabel=r"b (bias)")
        ax.set_title(str("Redshift bin = ["+str(zbins[z])+", "+str(zbins[z+1])+")"))
        if z == 0:
            ax.set(ylabel="P")
            f.subplots_adjust(bottom=0.25, top=0.8)
            ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.1), fancybox=True, shadow=False, ncol=2)
        ax.axis([0,8,0,2])
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main(save=True, load=False, email=False, log=True, justplotcorr=False)
