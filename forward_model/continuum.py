#!/usr/bin/env python
#
# Nov. 29 2017
# @Dino Hsu
#
# Make a continuum function
#
# 
# Refer to Apogee_tool
#

import nirspec_pip as nsp
import apogee_tools.forward_model as apmdl
import splat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.special import wofz
import copy


def continuum(data, mdl, **kwargs):
    
    """
    This function returns a continuum corrected model.
    Ported from apogee_tools
    
    Parameters
    ----------
    data    : an instance of the Spectrum object
              the data used in the fitting as a polynomial
    mdl     : an instance of the Model object
              the model being corrected
    deg     : a float value 
              the degree of the fitting polynomial. The default vaule is 5.
    
    Returns
    -------
    mdlcont : an instance of the Model object
              A continuum corrected model object from the fitted 
              polynomial
    """
    
    deg = kwargs.get('deg', 5)
    prop = kwargs.get('prop',False)

    flux_in_rng = np.where((mdl.wave > data.wave[0]) & (mdl.wave < data.wave[-1]))[0]
    mdl_wave = mdl.wave[flux_in_rng]
    mdl_flux = mdl.flux[flux_in_rng]

    mdl_flux_vals = [x for x in mdl_flux if str(x) != 'nan']
    mdl_flux = mdl_flux/max(mdl_flux_vals)
    mdl_flux = [1 if str(x) == 'nan' else x for x in mdl_flux]

    mdl_int = np.interp(data.wave, mdl_wave, mdl_flux)
    mdlcont = nsp.Model(wave=mdl_wave, flux=mdl_flux)

    mdldiv  = data.flux/mdl_int

    #find mean and stdev of mdldiv
    mdldiv_vals = [x for x in mdldiv if str(x) != 'nan']
    mean_mdldiv = np.mean(mdldiv_vals)
    std_mdldiv  = np.std(mdldiv_vals)
    
    #replace nans and outliers with average value
    mdldiv = np.array([mean_mdldiv if str(x) == 'nan' else x for x in mdldiv])
    mdldiv[mdldiv <= mean_mdldiv - 2*std_mdldiv] = mean_mdldiv
    mdldiv[mdldiv >= mean_mdldiv + 2*std_mdldiv] = mean_mdldiv
    
    pcont  = np.polyfit(data.wave, mdldiv, deg)
    
    #mdlcont.flux *= np.polyval(pcont, mdlcont.wave)
    mdl.flux *= np.polyval(pcont, mdl.wave)/max(mdl_flux_vals)

    if prop is True:
        return mdl, np.polyval(pcont, mdl.wave)/max(mdl_flux_vals)
    else:
        return mdl

def _continuumFit(wave, a, b, c):
    return a*wave**2 + b*wave + c

def linear_fit(x,a,b):
    return a*x + b

def gaus_absorption_only(x,x0,sigma,a,b):
    return (1 - b*np.e**(-(x-x0)**2/(2*sigma**2)))*a

def gaus_absorption_spec(x, x0, sigma, scale, a, b, c, d):
    """
    This function is to fit continuum of the spectra 
    with one absorption feature.
    
    """
    gaussian_absorption = 1 - scale*np.e**(-(x-x0)**2/(2*sigma**2))
    return gaussian_absorption * (a*x**2 + b*x + c) + d

def voigt_profile(x, x0, amp, gamma, scale, a, b, c, d):
    """
    Return a spectral line absorption with a second order 
    polynomial and the Voigt line shape absorption at x0 
    with Lorentzian component HWHM gamma and Gaussian component
    HWHM alpha (the latter is absorbed in the amp parameter.

    """
    #sigma = alpha / np.sqrt(2 * np.log(2))

    voigt_absorption = (1 - scale * np.real(wofz((x-x0 + 1j*gamma)*amp)))
    return voigt_absorption * (a*x**2 + b*x + c) + d

def continuumTelluric(data, model=None, order=None):
    """
    Return a continnum telluric standard data.
    Default: return a telluric flux of mean 1.

    Parameters
    ----------
    data:  spectrum object
           The input telluric data to be continuum
           corrected

    model: (optional) model object
           The telluric model to obtain the mean flux
           Instead of 1 as in default, it returns a constant
           shift by the difference between the mean flux of 
           the telluric data and that of the telluric model

    Returns
    -------
    data: spectrum object
          continuum corrected telluric data

    Examples
    --------
    >>> import nirspec_pip as nsp
    >>> nsp.continuumTelluric(data)

    >>> nsp.continuumTelluric(data,model)

    """
    if model is None:
        wavelow = data.wave[0]
        wavehigh = data.wave[-1]
        model = nsp.getTelluric(wavelow,wavehigh)

    if order == 35:
        # O35 has a voigt absorption profile
        popt, pcov = curve_fit(voigt_profile,data.wave[20:-20],
            data.flux[20:-20],
            p0=[21660,2000,0.1,0.1,0.01,0.1,10000,1000])
        const = np.mean(data.flux/voigt_profile(data.wave, *popt))\
        -np.mean(model.flux)
        data.flux = data.flux/voigt_profile(data.wave, *popt) - const
        data.noise = data.noise/voigt_profile(data.wave, *popt)

    elif order == 38:
        # O38 has rich absorption features
        def fit_continuum_O38(x,a,b,**kwargs):
            flux = kwargs.get('flux',data.flux)
            linear = a*x + b
            return flux/linear

        model2 = copy.deepcopy(model)
        model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, 
            flux=model2.flux, vbroad=4.8, 
            rotate=False, gaussian=True)
        model2.flux = np.array(splat.integralResample(xh=model2.wave, 
            yh=model2.flux, xl=data.wave))
        model2.wave = data.wave
        
        popt, pcov = curve_fit(fit_continuum_O38,data.wave,
            model2.flux, p0=[ 8.54253062e+00  , -166000])
        #const = np.mean(data.flux/linear_fit(data.wave, *popt))-np.mean(model.flux)
        #data.flux = data.flux/linear_fit(data.wave, *popt) - const
        data.flux = data.flux/linear_fit(data.wave, *popt)
        data.noise = data.noise/linear_fit(data.wave, *popt)

    elif order == 55:
        popt, pcov = curve_fit(_continuumFit, data.wave, data.flux)
        data.flux = data.flux/_continuumFit(data.wave, *popt)
        data.noise = data.noise/_continuumFit(data.wave, *popt)
        data.flux /= np.max(data.flux)
        data.noise /= np.max(data.flux)
        data.flux /= 0.93
        data.noise /= 0.93

    elif order == 56:
        # select the highest points to fit a polynomial
        x1 = np.max(data.flux[0:100])
        x2 = np.max(data.flux[100:150])
        x3 = np.max(data.flux[200:300])
        x4 = np.max(data.flux[300:400])
        x5 = np.max(data.flux[600:700])
        x6 = np.max(data.flux[700:800])

        a = [float(data.wave[np.where(data.flux == x1)]),
        float(data.wave[np.where(data.flux == x2)]),
        float(data.wave[np.where(data.flux == x3)]),
        float(data.wave[np.where(data.flux == x4)]),
        float(data.wave[np.where(data.flux == x5)]),
        float(data.wave[np.where(data.flux == x6)])]
        
        b = [x1, x2, x3, x4, x5, x6]

        popt, pcov = curve_fit(_continuumFit, a, b)

        data.flux = data.flux/_continuumFit(data.wave,*popt)*0.85
        data.noise = data.noise/_continuumFit(data.wave,*popt)*0.85

    elif order == 59:
        popt, pcov = curve_fit(voigt_profile,
            data.wave, data.flux, 
            p0=[12820,2000,0.1,0.1,0.01,0.1,10000,1000])
        data.flux /= voigt_profile(data.wave, *popt)
        data.noise /= voigt_profile(data.wave, *popt)

    elif order == 65:
        # O65 is best mateched by a gaussian absorption feature
        popt, pcov = curve_fit(gaus_absorption_only,
            data.wave,data.flux, p0=[11660,50,2000,2000])
        const = np.mean(data.flux/gaus_absorption_only(data.wave, 
            *popt))-np.mean(model.flux)
        data.flux = data.flux/gaus_absorption_only(data.wave, *popt) - const
        data.noise /= gaus_absorption_only(data.wave, *popt)

    else:
        # this second order polynomial continnum correction 
        # works for the O33, O34, O36, and O37
        popt, pcov = curve_fit(_continuumFit, data.wave, data.flux)
        const = np.mean(data.flux/_continuumFit(data.wave, 
            *popt))-np.mean(model.flux)
        data.flux = data.flux/_continuumFit(data.wave, *popt) - const
        data.noise = data.noise/_continuumFit(data.wave, *popt)

        #if order == 37:
        #    data.flux -= 0.05

    return data
