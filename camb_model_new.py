#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:38:07 2018

Handler for the big nasty integrals pertaining to generating correlation models
from CAMB.

The model desired here is:
    
    omega_{DM}(theta) = pi times integral of (z = 0 to infinity)
                           times integral of (k = 0 to infinity)
                           times the dimensionless power spectrum
                               (which is equal to the power spectrum times
                                k^3 divided by 2 pi squared)
                           times the zeroth order bessel function of the first kind,
                               whose input is k times theta times the comoving 
                               distance along the line of sight
                           times the derivative of the comoving distance with respect
                               to redshift
                           times the PDZ of one sample, times the PDZ of the other sample
                           dz dk
    All detailed in DiPompeo et al 2017/

@author: csh4
"""

from scipy import interpolate
import os
import camb
import pickle
import scipy
import numpy       as np
import pandas      as pd
import matplotlib.pyplot as plt
import mp_manager  as mpm


"""
Initializing the CAMB parameters and getting the matter power interpolator
"""
params, results, PK = 0,0,0
params = camb.CAMBparams()
params.set_cosmology()#H0=67.5, ombh2=0.022, omch2=0.122)
params.InitPower.set_params()#ns=0.965)

results = camb.get_results(params)
results.calc_power_spectra()
c = 2.998 * ( 10 ** 5 ) #km/s
PK = camb.get_matter_power_interpolator(params, k_hunit=False, hubble_units=False,
                                        nonlinear=True, kmax=1000, return_z_k = True)


"""
BEGIN HELPER FUNCTIONS (which mostly go into the integrand)
"""

def p_spec(z, k):
    """
    matter power spectrum as function of z
    """
    return PK[0].P(z, k)    

def p_spec_nodims(z, k):
    """
    dimensionless power spectrum at redshift z, wavenumber k
    """
    return p_spec(z,k)/(2*(np.pi**2))

def comoving_distance(z):
    """
    comoving distance along line of sight
    """
    return results.comoving_radial_distance(z)

def dzdchi(z):
    """
     (H0 /c)[Ωm (1 + z)**3 + ΩΛ ]**1/2 = H(z)/c
    """
    return results.hubble_parameter(z)/c

def bessel0(x):
    """
    zeroth order bessel function of 1st kind
    """
    return scipy.special.jn(0, x)

def deg_2_rad(theta):
    """
    Convert degrees to radians
    """
    return np.pi*theta/(180.0)

def and_them(arrs):
    """
    And the arrays of booleans given
    """
    out = np.ones(len(arrs[0]), dtype=bool)
    for arr in arrs:
        out = np.logical_and(out, arr)
    return out

def or_them(arrs):
    """
    Or the arrays of booleans given
    """
    out = np.zeros(len(arrs[0]), dtype=bool)
    for arr in arrs:
        out = np.logical_or(out, arr)
    return out

def gauss(x, mu, sigma):
    """
    Return the probability associated with a normal distribution of a given
    mean and sigma for some given x value
    """
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-((x-mu)**2))/(2*(sigma**2)))
            
def inbin(x, lo, hi):
    """
    Return true if x is between lo and hi
    """
    good = np.logical_and(np.greater_equal(x, lo),
                             np.less_equal(x, hi))
    return good

def norm(x, y):
    """
    Normalize the distibution to 1.
    """
    return y/(np.trapz(y, x=x))





"""
BEGIN INTEGRATION FUNCTIONS
"""
    
def k_integrand(z, k, theta):
    """
    the k part of the integrand from DiPompeo et al 2017
    """
    return (k*
            p_spec_nodims(z,k)*
            bessel0(k*
                    deg_2_rad(theta)*
                    comoving_distance(z)))


"""
BEGIN K_INTEGRAL LOGGING
"""

class kstore:
    """
    This class basically pre-generates the k integrand values for some given 
    parameter set of k and z and theta, and holds them in memory as a numpy 
    interpolator object (as a function of z and theta)
    
    Thus, you can quickly generate models for a range of z and theta values. 
    """
    def __init__(self, gen = False, def_name="/kstore.hdf"):
        """
        Load or generate the k_integral values for an ensemble of z and theta
        values
        
        Make a 2d interpolator
        """
        self.krange = np.linspace(0.0, 100.0, num=200000)
        self.zrange = np.arange(0, 4.0, 0.01)
        self.trange = np.logspace(np.log10(0.0025),np.log10(1.3),num=50)
        self.name = def_name
        
        self.df = pd.DataFrame()
        if gen:
            self.gen_self()
        else:
            try:
                self.load()
            except FileNotFoundError:
                print("Could not find kstore = " + def_name + ". Generating new one.")
                self.gen_self()
            
        zvals = self.df.values
        self.f = interpolate.interp2d(self.zrange, self.trange, zvals.T)
    
    def k_int(self, z, theta):
        """
        Retrieve a value from the interpolator
        """
        return self.f(z, theta) 
    
    def z_integrand_internal(self, z, pdz1_x, pdz1_y, pdz2_x, pdz2_y, theta):
        """
        Helper function for z_integral. What is being integrated over. 
        Details in DiPompeo et al. 2017
        """
        return (np.pi*
                np.interp(z, pdz1_x, pdz1_y)*
                np.interp(z, pdz2_x, pdz2_y)*
                dzdchi(z)*
                self.k_int(z, theta))
            
    def z_integral(self, pdz1_x, pdz1_y, pdz2_x, pdz2_y, theta):
        """
        Integrate the k_integral over z space given some PDZs and a theta value
        
        Uses the PDZs.
        """
        zvals = []
        for z in self.zrange:
            zvals.append(self.z_integrand_internal(z, pdz1_x, pdz1_y,
                                                      pdz2_x, pdz2_y, theta)[0])
        return np.trapz(np.array(zvals), x=self.zrange)
    
    def load(self):
        """
        Load the internal data frame from disk
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        parent    = os.path.dirname(directory)
        self.df = pd.read_hdf(parent+self.name, key='primary')
    
    def save(self):
        """
        Save the internal data frame to disk
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        parent    = os.path.dirname(directory)
        self.df.to_hdf(parent+self.name, key='primary', format='table')
    
    def gen_self(self):
        """
        Generate k_integral values for an enseble of z and theta values
        
        Save in the internal data frame
        """
        dct = {}
        for t in self.trange:
            k_ints = []
            for z in self.zrange:
                k_ints.append(np.trapz(k_integrand(z, self.krange, t), 
                                       x=self.krange))
            dct.update({t:k_ints})
        self.df = pd.DataFrame.from_dict(dct)
        self.save()
        

"""
CREATE A KSTORE OBJECT FOR MODEL FITTING

automatically generates a new integrand set if it does not exist. Alternatively,
pass in gen = True

I know this is sloppy. The model fitting functions should probably be methods
within the class kstore, so you don't have this scripted element. But it works ok. 
"""
kst = kstore(gen = False)

"""
BEGIN MODEL FITTING FUNCTIONS
"""
  
def mod(pdz1_x, pdz1_y, pdz2_x, pdz2_y, theta):
    """
    Generate a model value from the pdzs and a theta value. 
    
    Poorly named helper function
    """
    return kst.z_integral(pdz1_x, pdz1_y, pdz2_x, pdz2_y, theta)

def model(pdz1_x, pdz1_y, pdz2_x, pdz2_y, 
          trange = np.logspace(np.log10(0.0025),np.log10(1.3),num=50)):
    """
    Given some PDZs, generate a model and return. 
    """
    modvals = [mod(pdz1_x, pdz1_y, pdz2_x, pdz2_y, t) for t in trange]
    return trange, modvals
     

def likelihood(x_data, y_data, yerr, x_model, y_model):
    """
    Generate the likelihood of a given model being correct 
    (via product of normal distributions)
    
    Your standard bayesian model fitting routine
    """
    y_mod_interp = np.interp(x_data, x_model, y_model)
    prob = 1.0
    for i in range(len(y_data)):
        prob = prob * gauss(y_data[i], y_mod_interp[i], yerr[i])
    return prob

def chisquare(y1, y2, yerr):
    """
    Figure out the chi square value for the model, given some data points,
    model points, and errors
    
    @params
        y1 - data points
        y2 - model points
        yerr - data point errorbars
    """
    chisq= 0.0
    for i in range(len(y1)):
        chisq = chisq + ((y1[i] - y2[i])**2)/(yerr[i]**2)
    return chisq
        
def bias_fit(x_data, y_data, yerr, x_model, y_model):
    """
    Given a model, find the best fit bias measurement to some data, using a 
    bog-standard gaussian product likelihood function
    
    Recall, if you have two sets of data (say D1 and D2) and you are running
    the correlation, the model is multiplied by the product of the two biases,
    b1 and b2
    
    So what you are getting here is the square root of the product of the 
    biases b1 and b2. If this is an autocorrelation, this is the bias itself.
    
    If this is a cross correlation, you must square the bias received, divide
    it by the bias of one data set (as a known) to get the bias of the other
    set. 
    
    @params
        x_data  - The central angles of the correlation measurements
        y_data  - The values of the correlation measurements
        yerr    - The errorbars of the correlation measurements
        x_model - The angular coordinates of the model
        y_model - The values of the models

    @returns
        brange      - The range of bias values tested
        likelihoods - The probability associated with each bias value
        chisq       - The not reduced chi square value associated with the best
                      fit bias value
        best        - The best fit bias value
                        (i.e. square this and multiply it by the base model for
                         the best fitting model)
    """
    brange = np.linspace(0.0, 8.0, num=400)
    likelihoods = np.array([likelihood(x_data, y_data, yerr,
                                       x_model, np.array(y_model)*(b**2)) for b in brange], dtype=float)
    likelihoods = likelihoods/np.trapz(likelihoods, x=brange)
    
    best = brange[np.argmax(likelihoods)]
    y_mod_interp = np.interp(x_data, x_model, y_model)
    chisq = chisquare(y_data, np.array(y_mod_interp)*(best**2), yerr)
    
    return brange, likelihoods, chisq, best


def corr_bias(x_data, y_data, yerr, pdz1_x, pdz1_y, pdz2_x, pdz2_y):
    """
    Given a correlation measurement and associated PDZs, generate a model and
    fit as a bias to the measurement. Return:
        1) the model [unbiased]    (x and y float arrays)
        2) best fit bias (float)
        3) the bias PDF  (x and y float arrays)
    
    @params
        x_data  - The central angles of the correlation measurements
        y_data  - The values of the correlation measurements
        yerr    - The errorbars of the correlation measurements
        pdz1_x  - PDZ 1 redshift range to generate models from
        pdz1_y  - PDZ 1 probability values to generate models from
        pdz2_x  - PDZ 2 redshift range to generate models from
        pdz2_y  - PDZ 2 probability values to generate models from
    
    pdz1_x and pdz2_x, pdz1_y and pdz2_y should be the same for an autocorrelation

    @returns
        xmod   - the angular range associated with the generated model
        ymod   - the value of the model at each angle
        best   - The best fit bias value
                        (i.e. square this and multiply it by the base model for
                         the best fitting model)
        xbias  - The range of bias values tested
        ybias  - The probability associated with each bias value
        chisq  - The not reduced chi square value associated with the best
                 fit bias value
    """
    xmod, ymod = model(pdz1_x, pdz1_y, pdz2_x, pdz2_y)
    xbias, ybias, chisq, best = bias_fit(x_data, y_data, yerr, xmod, ymod)
    return xmod, ymod, best, xbias, ybias, chisq 

def corr_bias_plot(x_data, y_data, yerr, pdz1_x, pdz1_y, pdz2_x, pdz2_y, title=""):
    """
    Plot the correlation and bias given some data and PDZs (sort of a test function)
    """
    xmod, ymod, best, xbias, ybias, chisq = corr_bias(x_data, y_data, yerr, pdz1_x, pdz1_y, pdz2_x, pdz2_y)
    
    plt.figure()        
    plt.errorbar(x_data, y_data, yerr=yerr, ms=6.0, fmt='o', capthick=1.5,
                 capsize=4.0, elinewidth=1.5, label='Data')
    plt.plot(xmod, ymod, label="Model (no bias)")
    plt.plot(xmod, ymod*(best**2), label='Model (best bias of '+str(best)+')')
    plt.legend()
    plt.xlabel("Degrees")
    plt.ylabel("Correlation")
    plt.loglog()
    plt.title(title + " correlation and model")
    
    plt.figure()
    plt.step(xbias, ybias, color='black')
    plt.xlabel("Bias")
    plt.ylabel("Probability")
    plt.title(title + " bias fit")
    
    plt.show()
    


"""
Some old stuff from before I used OOP for model generation. OOP is certainly
the better way of doing things
"""

#def k_integral(z, theta):
#    """
#    The k integral itself 
#    """
#    krange = np.linspace(0.0, 100.0, num=200000)
#    #krange = np.linspace(0.0, 5.0, num=100)
#    #kvals  = np.array([k_integrand(z, k, theta) for k in krange], dtype=float)
#    kvals = k_integrand(z, krange, theta)
#    return np.trapz(kvals, x=krange)

#def z_integrand(z, theta, pdz1_x, pdz1_y, pdz2_x, pdz2_y):
#    return (np.pi*
#            np.interp(z, pdz1_x, pdz1_y)*
#            np.interp(z, pdz2_x, pdz2_y)*
#            dzdchi(z)*
#            k_integral(z, theta))
#
#def z_integral(theta, pdz1_x, pdz1_y, pdz2_x, pdz2_y):
#    zrange = np.arange(0, 4.0, 0.001)
#    zvals  = np.array([z_integrand(z, theta, pdz1_x, pdz1_y, pdz2_x, pdz2_y) for z in zrange])
#    #zvals = z_integrand(zrange, theta, pdz1_x, pdz1_y, pdz2_x, pdz2_y)
#    return np.trapz(zvals, x=zrange)

    
    
if __name__ == "__main__":
    print("done")