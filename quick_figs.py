#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:06:56 2018

@author: csh4
"""

from   scipy  import spatial
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import matplotlib
from astropy.io import fits
import gc
import os
import glob
from scipy.spatial import KDTree
from matplotlib.colors import LogNorm
import pymangle
from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



def file_len(path):
    store = pd.HDFStore(path)
    nrows = store.get_storer('primary').nrows
    store.close()
    return nrows
    
#    #Figure out the number of lines in a file
#    with open(path) as f:
#        for i, l in enumerate(f):
#            pass
#    return i + 1

wise_mangles = ["/u/csh4/Dropbox/research/agn/polygons/wise_mask_allwise_stars.ply",
                "/u/csh4/Dropbox/research/agn/polygons/wise_mask_allsky_pix.ply"]

def wise_mask(ra, dec, mangle_files=wise_mangles):
    """
    Takes a list of RA and DEC and returns the indices which are permissible
    
    Uses equatorial coordinates, converts them to galactic, does the thing, and
    returns the indices.
    
    Dependencies: numpy, astropy.coordinates.SkyCoord, mangle
    
    performance: medium (probably max)
    
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
    coordinates = SkyCoord(ra=np.array(ra, dtype=float)*u.degree,
                           dec=np.array(dec, dtype=float)*u.degree,
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
    return np.logical_not(ins_true)


bad_patches = []
bad_spots = [[245.77, 245.98, 42.68, 42.87],
             [247.28, 247.52, 42.86, 43.04],
             [133.92, 134.31,  2.27,  2.75],
             [210.00, 210.57,  0.26,  0.59],
             [ 35.40, 38.50,  -2.80, -1.70]]#,
#             [332.50, 333.8,   1.50,  2.20],
#             [135.60, 137.06, -1.70,  0.05],
#             [135.00, 137.20,  3.85,  4.70]]

def in_spots(ra, dec, spots=bad_spots):
    flag = np.zeros(len(ra), dtype=float)
    
    for spot in spots:
        in_spot = np.logical_and(np.logical_and(np.greater(ra, spot[0]),
                                                np.less(ra, spot[1])),
                                 np.logical_and(np.greater(dec, spot[2]),
                                                np.less(dec, spot[3])))
        flag = np.logical_or(flag, in_spot)
    
    return flag
  
     
only_patch = [0.0, 360.0, -30.0, 30.0]
small_patch_only = True

def use_only(ra, dec):
    if not small_patch_only:
        return np.zeros(np.shape(ra), dtype=bool)
    ra_outside = np.logical_or(np.greater(ra, only_patch[1]), np.less(ra, only_patch[0]))
    dec_outside = np.logical_or(np.greater(dec, only_patch[3]), np.less(dec, only_patch[2]))
    return np.logical_or(ra_outside, dec_outside)


def in_bad_patches(ra, dec):
    is_bad = np.zeros(np.shape(ra), dtype=bool)
    for patch in bad_patches:
        ra_inside = np.logical_and(np.greater(ra, patch[0]), np.less(ra, patch[1]))
        dec_inside = np.logical_and(np.greater(dec, patch[2]), np.less(dec, patch[3]))
        ra_dec_bad = np.logical_and(ra_inside, dec_inside)
        is_bad = np.logical_or(is_bad, ra_dec_bad)
    return is_bad

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


def fancyhist(x1, y1, x2, y2, xlabel, ylabel):
    f = plt.figure()
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])
    gs.update(wspace=0.0, hspace=0.0, right=0.78)
    
    n_xbins = 10
    n_ybins = 10
    
    y_min = np.amin([y1, y2])
    y_max = np.amax([y1, y2])
    x_min = np.amin([x1, x2])
    x_max = np.amax([x1, x2])
    
    xbins=np.linspace(x_min, x_max, n_xbins+1)
    ybins=np.linspace(y_min, y_max, n_ybins+1)
    
    main = plt.subplot(gs[1])
    main.tick_params(labelbottom=False, labelleft=False)
    hist1 = plt.hist2d(x1, y1, cmap="Reds", alpha = 0.5, bins=[xbins, ybins])
    hist2 = plt.hist2d(x2, y2, cmap="Blues", alpha = 0.5, bins=[xbins, ybins])
    
    box = main.get_position()
    cbar1 = f.add_axes([box.x0 + box.width + 0.01, box.y0, 0.025, box.height])
    plt.title("Red stuff")
    plt.colorbar(hist1[3], cax = cbar1)
    
    cbar2 = f.add_axes([box.x0 + box.width + 0.08, box.y0, 0.025, box.height])
    plt.title("Blue stuff")
    plt.colorbar(hist2[3], cax = cbar2)
    
    
    left = plt.subplot(gs[0], sharey=main)
    plt.hist(y1, orientation=u'horizontal', color="Red", bins=ybins)
    plt.hist(y2, orientation=u'horizontal', color="Blue", bins=ybins)
    plt.ylabel(ylabel)
    plt.xlabel("N")
    left.xaxis.set_label_position("top")
    left.xaxis.tick_top()
    
    bottom = plt.subplot(gs[3], sharex=main)
    plt.hist(x1, color="Red", bins=xbins)
    plt.hist(x2, color="Blue", bins=xbins)
    plt.xlabel(xlabel)
    plt.ylabel("N")
    bottom.yaxis.tick_right()
    bottom.yaxis.set_label_position("right")

    plt.show()
    
    
def and_them(arrs):
    out = np.ones(len(arrs[0]), dtype=bool)
    for arr in arrs:
        out = np.logical_and(out, arr)
    return out

def or_them(arrs):
    out = np.zeros(len(arrs[0]), dtype=bool)
    for arr in arrs:
        out = np.logical_or(out, arr)
    return out

def inbin(x, lo, hi, addition):
    good = np.logical_and(np.greater_equal(x, lo),
                          np.less(x, hi))
    good = np.logical_and(good, addition)
    return np.where(good)[0]

def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-((x-mu)**2))/(2*(sigma**2)))

def gauss_sum(x, mu, sigma):
    out = np.zeros(len(x))
    for i in range(len(mu)):
        out = out + gauss(x, mu[i], sigma[i])
    return out


def colorz(zmin=0.0, zmax=1.0):
    matplotlib.rcParams.update({'font.size': 12})
    
    
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    
    agn_target = parent + "/cats/raw/agn_err.hdf"
    agn = pd.read_hdf(agn_target, key="primary")
    
    z_best = agn['ZBEST_PEAK'].tolist()
    for i in np.where(np.logical_not(np.isnan(agn['SPECZ'])))[0]:
        z_best[i] = agn['SPECZ'][i]
        
    #has_match = np.logical_not(np.isnan(agn['frankenz_best_hsc2']))
    z_good = inbin(z_best, zmin, zmax, np.ones(len(z_best), dtype=bool))
    
    condition = z_good #and_them([has_match,z_good])
    reqs = np.where(condition)[0]
    
    #w2 = np.array([agn['W2'][i]             for i in reqs], dtype=float)
    w3 = np.array([agn['W3'][i]             for i in reqs], dtype=float)
    g  = np.array([agn['G_PSFFLUX_MAG'][i]  for i in reqs], dtype=float)
    z  = np.array([z_best[i]                for i in reqs], dtype=float)
    
    f = plt.figure()
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])
    gs.update(wspace=0.0, hspace=0.0, right=0.78, bottom=0.15)
    
    main = plt.subplot(gs[1])
    main.tick_params(labelbottom=False, labelleft=True)

    hist = plt.hist2d((g-w3), z,
               bins=[150,800], cmap='pink_r', norm=colors.PowerNorm(gamma=0.33), vmin=0)
    #plt.axvline(6.7, color='black', linewidth=1)
    plt.ylabel(r"$\textrm{Redshift}$")
    plt.title("AGN Color by redshift")
    box = main.get_position()
    cbar1 = f.add_axes([box.x0 + box.width + 0.01, box.y0, 0.025, box.height])
    plt.title(r"$N$")
    plt.colorbar(hist[3], cax = cbar1)#, ticks=[100, 80, 60, 40, 30, 20, 10, 5, 1, 0])
    
    bottom = plt.subplot(gs[3], sharex=main)
    plt.hist(g-w3, color="#A56B6B", bins=80, histtype='step', linewidth=2)
    #plt.axvline(6.0, color='black', linewidth=1)
    plt.xlabel(r"$\textrm{g}_{\textrm{psf}} - \textrm{W3}$")
    plt.ylabel(r"$N$")
    bottom.yaxis.tick_left()
    bottom.yaxis.set_label_position("left")
    plt.show()
    
    
    
    
def colorcolor():
    matplotlib.rcParams.update({'font.size': 12})
    
    
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    
    agn_target = parent + "/cats/raw/agn_err.hdf"
    agn = pd.read_hdf(agn_target, key="primary")
    
    has_match = np.logical_not(np.isnan(agn['frankenz_best_hsc2']))
    
    condition = and_them([has_match,])
    reqs = np.where(condition)[0]
    
    w2 = np.array([agn['W2'][i]             for i in reqs], dtype=float)
    w3 = np.array([agn['W3'][i]             for i in reqs], dtype=float)
    g  = np.array([agn['G_PSFFLUX_MAG'][i]  for i in reqs], dtype=float)
    
    f = plt.figure()
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])
    gs.update(wspace=0.0, hspace=0.0, right=0.78, bottom=0.15)
    
    main = plt.subplot(gs[1])
    main.tick_params(labelbottom=False, labelleft=True)

    hist = plt.hist2d((g-w3), (w2-w3),
               bins=[70,40], cmap='pink_r', norm=colors.PowerNorm(gamma=0.33), vmin=0)
    plt.axvline(6.0, color='black', linewidth=1)
    plt.ylabel(r"$\textrm{W2} - \textrm{W3}$")

    box = main.get_position()
    cbar1 = f.add_axes([box.x0 + box.width + 0.01, box.y0, 0.025, box.height])
    plt.title(r"$N$")
    plt.colorbar(hist[3], cax = cbar1)#, ticks=[100, 80, 60, 40, 30, 20, 10, 5, 1, 0])
    
    bottom = plt.subplot(gs[3], sharex=main)
    plt.hist(g-w3, color="#A56B6B", bins=80, histtype='step', linewidth=2)
    plt.axvline(6.0, color='black', linewidth=1)
    plt.xlabel(r"$\textrm{g}_{\textrm{psf}} - \textrm{W3}$")
    plt.ylabel(r"$N$")
    bottom.yaxis.tick_left()
    bottom.yaxis.set_label_position("left")
    plt.show()
    
    
def colorcolor_zbin(zmin=0.0, zmax=1.0):
    matplotlib.rcParams.update({'font.size': 12})
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    agn_target = parent + "/cats/raw/agn_err.hdf"
    agn = pd.read_hdf(agn_target, key="primary")
    z_best = agn['ZBEST_PEAK'].tolist()
    for i in np.where(np.logical_not(np.isnan(agn['SPECZ'])))[0]:
        z_best[i] = agn['SPECZ'][i]
    
    has_match = np.logical_not(np.isnan(agn['frankenz_best_hsc2']))
    z_good = np.logical_and(np.greater_equal(np.array(z_best), zmin),
                            np.less(np.array(z_best), zmax))
    condition = and_them([has_match, z_good])
    reqs = np.where(condition)[0]
    
    w2 = np.array([agn['W2'][i]             for i in reqs], dtype=float)
    w3 = np.array([agn['W3'][i]             for i in reqs], dtype=float)
    g  = np.array([agn['G_PSFFLUX_MAG'][i] for i in reqs], dtype=float)
    
    f = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])
    gs.update(wspace=0.0, hspace=0.0, right=0.78, bottom=0.15)
    main = plt.subplot(gs[1])
    main.tick_params(labelbottom=False, labelleft=True)
    hist = plt.hist2d((g-w3), (w2-w3),
               bins=[100,40], cmap='pink_r', norm=colors.PowerNorm(gamma=0.33))
    plt.axvline(6.7, color='black', linewidth=1)
    plt.ylabel(r"$\textrm{W2} - \textrm{W3}$")
    plt.title("Redshift bin = [" + str(zmin) + ", " + str(zmax) + ")")
    box = main.get_position()
    cbar1 = f.add_axes([box.x0 + box.width + 0.01, box.y0, 0.025, box.height])
    plt.title(r"$N$")
    plt.colorbar(hist[3], cax = cbar1)#, ticks=[100, 80, 60, 40, 30, 20, 10, 5, 1, 0])
    bottom = plt.subplot(gs[3], sharex=main)
    plt.hist(g-w3, color="#A56B6B", bins=50, histtype='step', linewidth=2)
    plt.axvline(6.7, color='black', linewidth=1)
    plt.xlabel(r"$\textrm{g}_{\textrm{psf}} - \textrm{W3}$")
    plt.ylabel(r"$N$")
    bottom.yaxis.tick_left()
    bottom.yaxis.set_label_position("left")
    plt.show()

def agn_pseudo_pdz(zmin=0.0, zmax=1.0, type=0, matched=True,
                       xrange=np.linspace(0.0, 4.0, num=500)):
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    agn_target = parent + "/cats/raw/agn_err.hdf"
    agn = pd.read_hdf(agn_target, key="primary")
    gmw2 = agn['G_PSFFLUX_MAG'] - agn['W3']
    
    type1flag = agn['P_T1']
    type2flag = agn['P_T2']
    
    criteria = np.ones(len(agn), dtype=bool)
    
    if matched == True:
        has_match = np.logical_not(np.isnan(agn['frankenz_best_hsc2']))
        criteria  = np.logical_and(has_match, criteria)
        
    if type == 1:
        criteria = np.logical_and(criteria, np.greater(type1flag, 0.5))
    elif type == 2:
        criteria = np.logical_and(criteria, np.greater(type2flag, 0.5))
    
    
    andyz_uperr    = np.abs(agn['ZBEST_UP83']-agn['ZBEST_PEAK'])
    andyz_loerr    = np.abs(agn['ZBEST_PEAK']-agn['ZBEST_LO17'])
    z_bstd = ((andyz_uperr+andyz_loerr)/2.0).tolist()
    z_best = agn['ZBEST_PEAK'].tolist()
    
    for i in np.where(np.logical_not(np.isnan(agn['SPECZ'])))[0]:
        z_best[i] = agn['SPECZ'][i]
        z_bstd[i] = 0.01
        
    z_best = np.array(z_best)
    z_good = and_them([np.greater_equal(z_best, zmin),
                                np.less(z_best, zmax)])
    criteria = np.logical_and(z_good, criteria)
    good = np.where(criteria)[0]
    z_best = z_best.tolist()
    mean_good  = np.array([z_best[i] for i in good])
    stdev_good = np.array([z_bstd[i] for i in good])
    
    return xrange, gauss_sum(xrange, mean_good, stdev_good)

def contour_from_hist(x, y, color, lab=None, bins=[20,20], levs=[0.1, 0.3, 0.6]):
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    max_value = (np.amax(counts))
    levels = np.array([levs[0]*max_value, levs[1]*max_value, levs[2]*max_value])
    
    plt.contour(xedges[:-1], yedges[:-1], counts.T, levels,
                linewidths=(2, 2, 3), linestyles=('dotted', 'dashed', 'solid'),
                alpha=0.5, colors=color, label=lab)


f2_exists = False

def agn_pdz(zmin=0.0, zmax=1.0, type=0, lumcut=True,
                       xrange=np.linspace(0.0, 4.0, num=500)):
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    agn_target = parent + "/cats/raw/agn_err.hdf"
    agn = pd.read_hdf(agn_target, key="primary")
    #gmw2 = agn['G_PSFFLUX_MAG'] - agn['W3']
    
    type1flag = agn['P_T1']
    type2flag = agn['P_T2']
    
    
    criteria = np.ones(len(agn), dtype=bool)
        
    if type == 1:
        criteria = np.logical_and(criteria, np.greater(type1flag, 0.5))
    elif type == 2:
        criteria = np.logical_and(criteria, np.greater(type2flag, 0.5))
    elif type == 3:
        criteria = np.logical_and(criteria, np.logical_not(np.isnan(agn['SPECZ'])))
    
    andyz_uperr    = np.abs(agn['ZBEST_UP83']-agn['ZBEST_PEAK'])
    andyz_loerr    = np.abs(agn['ZBEST_PEAK']-agn['ZBEST_LO17'])
    z_bstd = ((andyz_uperr+andyz_loerr)/2.0).tolist()
    z_best = agn['ZBEST_PEAK'].tolist()
    
    for i in np.where(np.logical_not(np.isnan(agn['SPECZ'])))[0]:
        z_best[i] = agn['SPECZ'][i]
        z_bstd[i] = 0.01
        
    z_best = np.array(z_best)
    z_good = and_them([np.greater_equal(z_best, zmin),
                                np.less(z_best, zmax)])
       
#    npeaks_ok = np.less(agn['NPEAKS'], 1.5)
#    criteria = np.logical_and(criteria, npeaks_ok)
  
    conf_ok = np.greater(agn['ZBEST_CONF'], 0.5)
    criteria = np.logical_and(criteria, conf_ok)
    
    
    l_cut = [43.70, 44.15, 44.35, 44.45, 44.55]
    lz_bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    lum_ok = np.ones(len(criteria), dtype=bool)
    lum = agn['LAGN'].tolist()
    if lumcut:
        z_bin = -1 * np.ones(len(lum), dtype=int)
        for i in range(len(lz_bins)-1):
            inbin = np.where(np.logical_and(np.greater(z_best, lz_bins[i]),
                                               np.less(z_best, lz_bins[i+1])))[0]
            for j in inbin:
                z_bin[j] = i
        in_a_zbin = np.where(z_bin != -1)[0]
        for i in in_a_zbin:
            if lum[i] < l_cut[z_bin[i]]:
                lum_ok[i] = False
    num_ok = len(np.where(lum_ok)[0])
    num_tot = len(lum_ok)
    print(type, zmin, zmax, num_ok/num_tot)
    criteria = np.logical_and(criteria, lum_ok)
    
#    cfok_l = len(conf_ok)
#    cfok_g = len(np.where(np.logical_and(npeaks_ok, conf_ok))[0])
#    frac = cfok_g/cfok_l
#    print(frac)
    
    
    criteria = np.logical_and(z_good, criteria)
    good = np.where(criteria)[0]
    z_best = z_best.tolist()
    mean_good  = np.array([z_best[i] for i in good])
    stdev_good = np.array([z_bstd[i] for i in good])
    
    f = fits.open(parent+"/cats/raw/agn_apz_veniced.fits")
    pdz   = f[1].data['PDZ']
    sz    = f[1].data['SPECZ']
    ra    = f[1].data['RA_HSC']
    dec   = f[1].data['DEC_HSC']
    zbest = f[1].data['ZBEST_PEAK']
    fits_imag  = f[1].data['I_PSFFLUX_MAG']
    
    
    andyz_uperr    = np.abs(f[1].data['ZBEST_UP83']-f[1].data['ZBEST_PEAK'])
    andyz_loerr    = np.abs(f[1].data['ZBEST_PEAK']-f[1].data['ZBEST_LO17'])
    z_bstd = ((andyz_uperr+andyz_loerr)/2.0).tolist()
    
    for i in np.where(np.logical_not(np.isnan(sz)))[0]:
        zbest[i] = sz[i]
        z_bstd[i] = 0.01
        
    criteria = np.ones(len(ra), dtype=bool)
    
    mask_ok = wise_mask(ra, dec)
    z_ok    = np.logical_and(np.greater_equal(np.array(zbest), zmin),
                             np.less(np.array(zbest), zmax))
    
    criteria = np.logical_and(criteria, z_ok)
    criteria = np.logical_and(criteria, mask_ok)
        
#    npeaks_ok = np.less(f[1].data['NPEAKS'], 1.5)
#    criteria = np.logical_and(criteria, npeaks_ok)
    
    conf_ok = np.greater(f[1].data['ZBEST_CONF'], 0.5)
    criteria = np.logical_and(criteria, conf_ok)
    
    lum_ok = np.ones(len(criteria), dtype=bool)
    lum = f[1].data['LAGN'].tolist()
    if lumcut:
        z_bin = -1 * np.ones(len(lum), dtype=int)
        for i in range(len(lz_bins)-1):
            inbin = np.where(np.logical_and(np.greater(zbest, lz_bins[i]),
                                               np.less(zbest, lz_bins[i+1])))[0]
            for j in inbin:
                z_bin[j] = i
        in_a_zbin = np.where(z_bin != -1)[0]
        for i in in_a_zbin:
            if lum[i] < l_cut[z_bin[i]]:
                lum_ok[i] = False
    criteria = np.logical_and(criteria, lum_ok)
    
    type1flag = f[1].data['P_T1']
    type2flag = f[1].data['P_T2']
    if type == 1:
        criteria = np.logical_and(criteria, np.greater(type1flag, 0.5))
    elif type == 2:
        criteria = np.logical_and(criteria, np.greater(type2flag, 0.5))
    elif type == 3:
        criteria = np.logical_and(criteria, np.logical_not(np.isnan(sz)))
    
    zrange = np.arange(0, 3.2025, 0.025)
    pdz_true = np.zeros(len(zrange), dtype=float)
    indices = np.where(criteria)[0]
    
    z_mu  = []
    z_sig = []
    imag  = []
    goodlums = []
    
    for i in indices:
        if np.isnan(sz[i]):
            arrsum = np.trapz(pdz[i], x=zrange)
            to_add = np.array(pdz[i], dtype=float)/arrsum
            pdz_true = pdz_true + to_add
        else:
            pdz_true = pdz_true + gauss(zrange, sz[i], 0.01)
        z_mu.append(zbest[i])
        z_sig.append(z_bstd[i])
        imag.append(fits_imag[i])
        goodlums.append(lum[i])
    
    global f2_exists
    if f2_exists == False:
        plt.subplots(num=2, nrows=2, ncols=2)
        plt.suptitle("Luminosity histogram for different AGN types")
        f2_exists = True
    
    if type == 1 or type == 2:
        color = ''
        if type == 1:
            color = 'blue'
        else:
            color = 'red'
        f = plt.figure(2)
        axs = f.axes
        num = 0
        if zmin == 0.5:
            num = 1
        elif zmin == 0.7:
            num = 2
        elif zmin == 0.9:
            num = 3
        plt.sca(axs[num])
        plt.hist(goodlums,histtype='step',bins=50, color=color, range=(42,47))
        plt.title("ZBIN " + str(zmin) + ", " + str(zmax))
    
    return xrange, gauss_sum(xrange, mean_good, stdev_good), zrange, pdz_true, z_mu, z_sig, imag


def gal_pseudo_pdz(zmin=0.0, zmax=1.0, xrange=np.linspace(0.0, 4.0, num=500),
                   imax = 24.0):
    hsc_zdist_f  = "/scratch/csh4/cats/hsc_25_zdist_f.hdf"
    
    xax = xrange
    yax = np.zeros(len(xrange), dtype=float)

    z_mu = []
    z_sig = []
    imag = []
    
    chunks = pd.read_hdf(hsc_zdist_f, chunksize=2500000)
    for chunk in chunks:
        chunk_ra   = chunk['ra'           ].values
        chunk_dec  = chunk['dec'          ].values
        chunk_z    = chunk['frankenz_best'].values
        chunk_std  = chunk['frankenz_std' ].values
        chunk_imag = chunk['imag_kron'    ].values
        
        #Figure out flags
        patches_ok = np.logical_and(np.logical_not(use_only(chunk_ra, chunk_dec)),
                                    np.logical_not(in_bad_patches(chunk_ra, chunk_dec)))
        wise_mask_ok = np.greater(chunk['wise_flag'].values, 0)
        sig_ok = np.greater(chunk_std, 0.0)
        z_good = np.logical_and(np.greater(chunk_z,zmin),np.less(chunk_z,zmax))
        #arcturus_ok = np.greater(chunk['flag'].tolist(), 0)
        spot_good = np.logical_not(in_spots(chunk_ra, chunk_dec))
        extended_good = np.greater(chunk['iclassification_extendedness'].values, 0.3)
        flag_ok = np.ones(len(chunk_ra), dtype=bool)
        imag_ok = np.less_equal(chunk_imag, imax)
    
        all_ok = and_them([patches_ok, wise_mask_ok, flag_ok, spot_good,
                            extended_good, sig_ok, imag_ok, z_good])
        indices_ok = np.where(all_ok)[0]
    
        mean  = np.array([chunk_z[i]   for i in indices_ok])
        stdev = np.array([chunk_std[i] for i in indices_ok])
        yax = yax + gauss_sum(xax, mean, stdev)
        
        for i in indices_ok:
            z_mu.append(chunk_z[i])
            z_sig.append(chunk_std[i])
            imag.append(chunk_imag[i])
        
        gc.collect()
    
    return xax, yax, z_mu, z_sig, imag
 
def gal_pdz(zmin=0.0, zmax=1.0, xrange=np.linspace(0.0, 4.0, num=500),
                   imax = 24.0):
    hsc_zdist_f  = "/scratch/csh4/cats/hsc_25_zdist_f.hdf"
    
    xax_fake = xrange
    yax_fake = np.zeros(len(xrange), dtype=float)
    
    arr = np.zeros(601, dtype=float)

    edge_file = fits.open("/scratch/csh4/py/codes/ver2/cats/pdz/PHOTOZ/pz_pdf_bins_frankenz.fits")
    edges = edge_file[1].data['BINS']
    edge_file.close()

    n_valid = 0
    n_chunks = 0
    
    
    chunks = pd.read_hdf(hsc_zdist_f, chunksize=5000000)
    for chunk in chunks:
        chunk_ra   = chunk['ra'           ].values
        chunk_dec  = chunk['dec'          ].values
        chunk_z    = chunk['frankenz_best'].values
        chunk_std  = chunk['frankenz_std' ].values
        chunk_imag = chunk['imag_kron'    ].values
        chunk_ids  = chunk['object_id'    ].values
        
        #Figure out flags
        patches_ok = np.logical_and(np.logical_not(use_only(chunk_ra, chunk_dec)),
                                    np.logical_not(in_bad_patches(chunk_ra, chunk_dec)))
        wise_mask_ok = np.greater(chunk['wise_flag'].values, 0)
        sig_ok = np.greater(chunk_std, 0.0)
        z_good = np.logical_and(np.greater(chunk_z,zmin),np.less(chunk_z,zmax))
        #arcturus_ok = np.greater(chunk['flag'].tolist(), 0)
        spot_good = np.logical_not(in_spots(chunk_ra, chunk_dec))
        extended_good = np.greater(chunk['iclassification_extendedness'].values, 0.3)
        flag_ok = np.ones(len(chunk_ra), dtype=bool)
        imag_ok = np.less_equal(chunk_imag, imax)
    
        all_ok = and_them([patches_ok, wise_mask_ok, flag_ok, spot_good,
                            extended_good, sig_ok, imag_ok, z_good])
        indices_ok = np.where(all_ok)[0]
    
        if len(indices_ok) == 0:
            pass
        else:        
            mean  = np.array([chunk_z[i]    for i in indices_ok])
            stdev = np.array([chunk_std[i]  for i in indices_ok])
            ids   = np.array([chunk_ids[i]  for i in indices_ok])
            
            ids = ids.reshape([len(ids), 1])
            tree = KDTree(ids)
            
            for file in glob.glob('/scratch/csh4/py/codes/ver2/cats/pdz/PHOTOZ/GAMA09H/*.fits'):
                f = fits.open(file)
                pz_ids = f[1].data['object_id']
                pz_arrs = f[1].data['P(z)']
                d, i = tree.query(pz_ids.reshape((len(pz_ids), 1)), distance_upper_bound=0.5)
                valid = np.where(d == 0)[0]
                for j in valid:
                    if np.isnan(pz_arrs[j][0]):
                        pass
                    else:
                        #
                        # Normalize the array
                        #
                        arrsum = np.trapz(pz_arrs[j], x=edges)
                        to_add = np.array(pz_arrs[j], dtype=float)/arrsum
                        
                        #
                        # Add to the existing pdz sum
                        #
                        arr = arr + to_add
                        yax_fake = yax_fake + gauss(xax_fake, mean[i[j]], stdev[i[j]])
                        
                        n_valid = n_valid + 1
            
        gc.collect()
        n_chunks = n_chunks + 1
    
    return xax_fake, yax_fake, edges, arr

def norm(x, y):
    return x, y/(np.trapz(y, x=x))
    
if __name__ == "__main__":
    zbins=np.array([0.3,0.5,0.7,0.9,1.1])
    f = plt.figure(1, figsize=(10,8))
    f2 = plt.subplots(num=2, nrows=2, ncols=2)
    plt.suptitle("Luminosity histogram for different AGN types")
    f2_exists = True
    
    for z in range(len(zbins)-1):
        f.add_subplot(221+z)
        zmin = zbins[z]
        zmax = zbins[z+1]
    
        plt.title("AGN PDZs for z = ["+str(zbins[z])+", "+str(zbins[z+1])+")")
        
        xf, yf, xr, yr, zmua, zsia, imaga  = agn_pdz(zmin=zmin, zmax=zmax)
        xf, yf = norm(xf,yf)
        xr, yr = norm(xr,yr)
        
        plt.figure(1)
        plt.plot(xf, yf, label='AGN - all types, pseudo', color='black', linestyle=":")
        plt.plot(xr, yr, label='AGN - all types, real', color='black')
    
        xf, yf, xr, yr, zmu2, zsi2, imag2 = agn_pdz(type=2, zmin=zmin, zmax=zmax)
        xf, yf = norm(xf,yf)
        xr, yr = norm(xr,yr)
        plt.figure(1)
        plt.plot(xf, yf, label='AGN - Type 2, pseudo', color='red', linestyle=":")
        plt.plot(xr, yr, label='AGN - Type 2, real', color='red')
    
        xf, yf, xr, yr, zmu1, zsi1, imag1 = agn_pdz(type=1, zmin=zmin, zmax=zmax)
        xf, yf = norm(xf,yf)
        xr, yr = norm(xr,yr)
        plt.figure(1)
        plt.plot(xf, yf, label='AGN - Type 1, pseudo', color='blue', linestyle=":")
        plt.plot(xr, yr, label='AGN - Type 1, real', color='blue')
    
        xf, yf, xr, yr, zmu1, zsi1, imag1 = agn_pdz(type=3, zmin=zmin, zmax=zmax)
        xf, yf = norm(xf,yf)
        xr, yr = norm(xr,yr)
        plt.figure(1)
        plt.plot(xr, yr, label='AGN - specz only', color='green')
    
        plt.axvline(zmin, color='black', linewidth=1)
        plt.axvline(zmax, color='black', linewidth=1)
        plt.grid('on')
        plt.legend()
        plt.xlabel("Redshift")
        plt.ylabel("Probability")
        plt.axis([0.0, 3.0, 0.0, 6.5])
    plt.show()
