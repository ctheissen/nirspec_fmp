import nirspec_pip as nsp
import apogee_tools as ap
import apogee_tools.forward_model as apmdl
import splat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')
import copy
import time
import os

# example params setup
# below is the fitted parameters of J0720-0846 on 2014 Jan 19
params = {'teff':2300,'logg':5.5,'z':-0.00,'en':0.00,'order':33,\
'vsini':6.9,'rv':84.3,'alpha':1.5}

def makeModel(teff,logg,z,vsini,rv,alpha,**kwargs):
	"""
	Return a forward model.

	Parameters
	----------
	params : a dictionary that specifies the parameters such as teff, logg, z.
	data   : an input science data used for continuum correction

	Returns
	-------
	model: a synthesized model
	"""

	# read in the parameters
	#params  = kwargs.get('params')
	#teff  = params['teff']
	#logg  = params['logg']
	#z     = params['z']
	#en    = params['en']
	#order = params['order']
	#vsini = params['vsini']
	#rv    = params['rv']
	#alpha = params['alpha']
	order = kwargs.get('order', 33)
	lsf  = kwargs.get('lsf', 4.5) # instrumental LSF
	tell  = kwargs.get('tell', True) # apply telluric
	data  = kwargs.get('data', None) # for continuum correction and resampling

	# read in a model
	model = nsp.Model(teff=teff, logg=logg, feh=z, order=order)
	# contunuum correction
	if data is not None:
		model = nsp.continuum(data=data, mdl=model)
	# apply vsini
	model.flux = apmdl.rotation_broaden.broaden(wave=model.wave, flux=model.flux, vbroad=vsini, rotate=True)
	# apply rv (including the barycentric correction)
	model.wave = apmdl.rv_function.rvShift(model.wave, rv=rv)
	# apply telluric
	if tell is True:
		apmdl.telluric.BASE = '/Users/dinohsu/projects/apogee_all/apogee_tools/'
		tell_sp = nsp.getTelluric(wavelow=model.wave[0],wavehigh=model.wave[-1])
		model = apmdl.telluric.applyTelluric(mdl=model,tell_sp=tell_sp, alpha=alpha)
	# NIRSPEC LSF
	model.flux = apmdl.rotation_broaden.broaden(wave=model.wave, flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)
	# integral resampling
	if data is not None:
		model.flux = np.array(splat.integralResample(xh=model.wave, yh=model.flux, xl=data.wave))
		model.wave = data.wave

	return model


def returnModelFit(data,theta,**kwargs):
	"""
	Chisquare computing function for MCMC.
	"""
	params  = kwargs.get('params')
	lsf  = kwargs.get('lsf', 4.5) # instrumental LSF
	plot  = kwargs.get('plot', False)

	synth_mdl = makeModel(params=theta, lsf=lsf, data=data)
	chisquare = nsp.chisquare(data,synth_mdl)
	if plot == True:
		plt.figure(figsize=[12,5])
		plt.plot(data.wave, data.flux, label='data', color='k', alpha=.7, linewidth=1)
		plt.plot(synth_mdl.wave, synth_mdl.flux, label=r'$\chi^2=%s$'%(str(chisquare)), color='r', alpha=.7, linewidth=1)
		plt.legend(loc='upper right')
		plt.ylabel(r'$F_{\lambda}$ (counts)', fontsize=15)
		plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=15)
		plt.show()
		plt.close()
	return chisquare

def convolveTelluric(lsf,telluric_data, **kwargs):
	"""
	Return a convolved telluric model given a telluric data and lsf.
	"""
	# get a telluric standard model
	wavelow = telluric_data.wave[0] - 50
	wavehigh = telluric_data.wave[-1] + 50
	telluric_model = nsp.getTelluric(wavelow=wavelow,wavehigh=wavehigh)
	# resample
	telluric_model.flux = np.array(splat.integralResample(xh=telluric_model.wave\
		, yh=telluric_model.flux, xl=telluric_data.wave))
	telluric_model.wave = telluric_data.wave
	# lsf
	telluric_model.flux = apmdl.rotation_broaden.broaden(wave=telluric_model.wave\
		, flux=telluric_model.flux, vbroad=lsf, rotate=False, gaussian=True)
	return telluric_model

def getLSF(telluric_data, **kwargs):
	"""
	Return a best LSF value from a telluric data.
	"""
	lsf_list = []
	test_lsf = np.arange(3.0,10.0,0.1)
	for i in test_lsf:
		data = copy.deepcopy(telluric_data)
		telluric_model = nsp.convolveTelluric(i,data)
		data = nsp.continuumTelluric(data=data, model=telluric_model)
		chisquare = nsp.chisquare(data,telluric_model)
		lsf_list.append([chisquare,i])
	lsf = min(lsf_list)[1]
	return lsf

def getAlpha(telluric_data,lsf):
	"""
	Return a best alpha value from a telluric data.
	"""
	alpha_list = []
	test_alpha = np.arange(0.1,2,0.1)
	for i in test_alpha:
		data = copy.deepcopy(telluric_data)
		telluric_model = nsp.convolveTelluric(lsf,data)
		telluric_model.flux **= i 
		data = nsp.continuumTelluric(data=data, model=telluric_model)
		chisquare = nsp.chisquare(data,telluric_model)
		alpha_list.append([chisquare,i])
	alpha = min(alpha_list)[1]
	return alpha


def initModelFit(data, model, **kwargs):
	"""
	Use the Nelder-Mead "Amoeba" algorithm to obtain the fitted Parameters
	for the forward modeling initialization stage. 
	"""

	return best_params, chisquare

def mcmcModeling(**kwargs):
	"""
	MCMC routine for the forward modeling.
	"""
