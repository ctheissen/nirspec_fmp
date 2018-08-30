import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import os
import nirspec_fmp as nsp
import apogee_tools as ap
import apogee_tools.forward_model as apmdl
import splat

def makeModel(teff,logg,z,vsini,rv,alpha,wave_offset,flux_offset,**kwargs):
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
	order = kwargs.get('order', 33)
	lsf   = kwargs.get('lsf', 6.0)   # instrumental LSF
	tell  = kwargs.get('tell', True) # apply telluric
	data  = kwargs.get('data', None) # for continuum correction and resampling

	# read in a model
	model = nsp.Model(teff=teff, logg=logg, feh=z, order=order)
	
	# apply vsini
	model.flux = apmdl.rotation_broaden.broaden(wave=model.wave, 
		flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
	
	# apply rv (including the barycentric correction)
	#model.wave = apmdl.rv_function.rvShift(model.wave, rv=rv)
	model.wave = rvShift(model.wave, rv=rv)
	
	# apply telluric
	if tell is True:
		model = nsp.applyTelluric(model=model, alpha=alpha, airmass='1.5')
	# NIRSPEC LSF
	model.flux = apmdl.rotation_broaden.broaden(wave=model.wave, 
		flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# add a fringe pattern to the model
	#model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

	# wavelength offset
	model.wave += wave_offset

	# flux offset
	model.flux += flux_offset
	
	# integral resampling
	if data is not None:
		model.flux = np.array(splat.integralResample(xh=model.wave, 
			yh=model.flux, xl=data.wave))
		model.wave = data.wave
		# contunuum correction
		model = nsp.continuum(data=data, mdl=model)

	return model

def rvShift(wavelength, rv):
	"""
	Perfrom the radial velocity correction.

	Parameters
	----------
	wavelength 	: 	numpy array 
					model wavelength (in Angstroms)

	rv 			: 	float
					radial velocity shift (in km/s)

	Returns
	-------
	wavelength 	: 	numpy array 
					shifted model wavelength (in Angstroms)
	"""
	return wavelength * ( 1 + rv / 299792.458)

def applyTelluric(model, alpha=1, airmass='1.5'):
	"""
	Apply the telluric model on the science model.

	Parameters
	----------
	model 	:	model object
				BT Settl model
	alpha 	: 	float
				telluric scaling factor (the power on the flux)

	Returns
	-------
	model 	: 	model object
				BT Settl model times the corresponding model

	"""
	# read in a telluric model
	wavelow  = model.wave[0] - 10
	wavehigh = model.wave[-1] + 10
	telluric_model = nsp.getTelluric(wavelow=wavelow,
		wavehigh=wavehigh, alpha=alpha, airmass=airmass)
	# apply the telluric alpha parameter
	#telluric_model.flux = telluric_model.flux**(alpha)

	#if len(model.wave) > len(telluric_model.wave):
	#	print("The model has a higher resolution ({}) than the telluric model ({})."\
	#		.format(len(model.wave),len(telluric_model.wave)))
	#	model.flux = np.array(splat.integralResample(xh=model.wave, 
	#		yh=model.flux, xl=telluric_model.wave))
	#	model.wave = telluric_model.wave
	#	model.flux *= telluric_model.flux

	#elif len(model.wave) < len(telluric_model.wave):
	## This should be always true
	telluric_model.flux = np.array(splat.integralResample(xh=telluric_model.wave, 
		yh=telluric_model.flux, xl=model.wave))
	telluric_model.wave = model.wave
	model.flux *= telluric_model.flux

	#elif len(model.wave) == len(telluric_model.wave):
	#	model.flux *= telluric_model.flux
		
	return model

def convolveTelluric(lsf,telluric_data,alpha=1):
	"""
	Return a convolved telluric transmission model given a telluric data and lsf.
	"""
	# get a telluric standard model
	wavelow = telluric_data.wave[0] - 50
	wavehigh = telluric_data.wave[-1] + 50
	telluric_model = nsp.getTelluric(wavelow=wavelow,wavehigh=wavehigh)
	telluric_model.flux **= alpha
	# lsf
	telluric_model.flux = apmdl.rotation_broaden.broaden(wave=telluric_model.wave, 
		flux=telluric_model.flux, vbroad=lsf, rotate=False, gaussian=True)
	# resample
	telluric_model.flux = np.array(splat.integralResample(xh=telluric_model.wave, 
		yh=telluric_model.flux, xl=telluric_data.wave))
	telluric_model.wave = telluric_data.wave
	return telluric_model

def getLSF(telluric_data,continuum=True,test=False,save_path=None):
	"""
	Return a best LSF value from a telluric data.
	"""
	lsf_list = []
	test_lsf = np.arange(3.0,13.0,0.1)
	
	data = copy.deepcopy(telluric_data)
	if continuum is True:
		data = nsp.continuumTelluric(data=data)

	for i in test_lsf:
		telluric_model = nsp.convolveTelluric(i,data)
		if telluric_data.order == 59:
			telluric_model.flux **= 3
			# mask hydrogen absorption feature
			data2          = copy.deepcopy(data)
			tell_mdl       = copy.deepcopy(telluric_model)
			mask_pixel     = 450
			data2.wave     = data2.wave[mask_pixel:]
			data2.flux     = data2.flux[mask_pixel:]
			data2.noise    = data2.noise[mask_pixel:]
			tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
			tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

			chisquare = nsp.chisquare(data2,tell_mdl)

		else:
			chisquare = nsp.chisquare(data,telluric_model)
		lsf_list.append([chisquare,i])

		if test is True:
			plt.plot(telluric_model.wave,telluric_model.flux+(i-3)*10+1,
				'r-',alpha=0.5)

	if test is True:
		plt.plot(telluric_data.wave,telluric_data.flux,
			'k-',label='telluric data',alpha=0.5)
		plt.title("Test LSF",fontsize=15)
		plt.xlabel("Wavelength ($\AA$)",fontsize=12)
		plt.ylabel("Transmission + Offset",fontsize=12)
		plt.minorticks_on()
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_lsf_data_mdl.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

		fig, ax = plt.subplots()
		for i in range(len(lsf_list)):
			ax.plot(lsf_list[i][1],lsf_list[i][0],'k.',alpha=0.5)
		ax.plot(min(lsf_list)[1],min(lsf_list)[0],'r.',
			label="best LSF {} km/s".format(min(lsf_list)[1]))
		ax.set_xlabel("LSF (km/s)",fontsize=12)
		ax.set_ylabel("$\chi^2$",fontsize=11)
		plt.minorticks_on()
		plt.legend(fontsize=10)
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_lsf_chi2.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

	lsf = min(lsf_list)[1]

	if telluric_data.order == 61 or telluric_data.order == 62 \
	or telluric_data.order == 63 or telluric_data.order == 64:
		lsf = 5.5
		print("The LSF is obtained from orders 60 and 65 (5.5 km/s).")

	return lsf

def getAlpha(telluric_data,lsf,continuum=True,test=False,save_path=None):
	"""
	Return a best alpha value from a telluric data.
	"""
	alpha_list = []
	test_alpha = np.arange(0.1,7,0.1)

	data = copy.deepcopy(telluric_data)
	if continuum is True:
		data = nsp.continuumTelluric(data=data)

	for i in test_alpha:
		telluric_model = nsp.convolveTelluric(lsf,data,
			alpha=i)
		#telluric_model.flux **= i 
		if data.order == 59:
			# mask hydrogen absorption feature
			data2          = copy.deepcopy(data)
			tell_mdl       = copy.deepcopy(telluric_model)
			mask_pixel     = 450
			data2.wave     = data2.wave[mask_pixel:]
			data2.flux     = data2.flux[mask_pixel:]
			data2.noise    = data2.noise[mask_pixel:]
			tell_mdl.wave  = tell_mdl.wave[mask_pixel:]
			tell_mdl.flux  = tell_mdl.flux[mask_pixel:]

			chisquare = nsp.chisquare(data2,tell_mdl)

		else:
			chisquare = nsp.chisquare(data,telluric_model)
		alpha_list.append([chisquare,i])

		if test is True:
			plt.plot(telluric_model.wave,telluric_model.flux+i*10,
				'k-',alpha=0.5)

	if test is True:
		plt.plot(telluric_data.wave,telluric_data.flux,
			'r-',alpha=0.5)
		plt.rc('font', family='sans-serif')
		plt.title("Test Alpha",fontsize=15)
		plt.xlabel("Wavelength ($\AA$)",fontsize=12)
		plt.ylabel("Transmission + Offset",fontsize=12)
		plt.minorticks_on()
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_alpha_data_mdl.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

		fig, ax = plt.subplots()
		plt.rc('font', family='sans-serif')
		for i in range(len(alpha_list)):
			ax.plot(alpha_list[i][1],alpha_list[i][0],'k.',alpha=0.5)
		ax.plot(min(alpha_list)[1],min(alpha_list)[0],'r.',
			label="best alpha {}".format(min(alpha_list)[1]))
		ax.set_xlabel(r"$\alpha$",fontsize=12)
		ax.set_ylabel("$\chi^2$",fontsize=12)
		plt.minorticks_on()
		plt.legend(fontsize=10)
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_alpha_chi2.png"\
				.format(telluric_data.name,
					telluric_data.order))
		plt.show()
		plt.close()

	alpha = min(alpha_list)[1]

	return alpha

def getFringeFrequecy(tell_data, test=False):
	"""
	Use the Lomb-Scargle Periodogram to identify 
	the fringe pattern.
	"""
	tell_sp  = copy.deepcopy(tell_data)

	## continuum correction
	tell_sp  = nsp.continuumTelluric(data=tell_sp)

	## get a telluric model
	lsf      = nsp.getLSF(tell_sp)
	alpha    = nsp.getAlpha(tell_sp,lsf)
	tell_mdl = nsp.convolveTelluric(lsf=lsf,
		telluric_data=tell_sp,alpha=alpha)

	## fit the fringe pattern in the residual
	pgram_x = np.array(tell_sp.wave,float)[10:-10]
	pgram_y = np.array(tell_sp.flux - tell_mdl.flux,float)[10:-10]
	offset  = np.mean(pgram_y)
	pgram_y -= offset
	mask    = np.where(pgram_y - 1.5 * np.absolute(np.std(pgram_y)) > 0)
	pgram_x = np.delete(pgram_x, mask)
	pgram_y = np.delete(pgram_y, mask)
	pgram_x = np.array(pgram_x, float)
	pgram_y = np.array(pgram_y, float)

	f = np.linspace(0.01,10,100000)

	## Lomb Scargle Periodogram
	pgram = signal.lombscargle(pgram_x, pgram_y, f)

	if test:
		fig, ax = plt.subplots(figsize=(16,6))
		ax.plot(f,pgram, 'k-', label='residual',alpha=0.5)
		ax.set_xlabel('frequency')
		plt.legend()
		plt.show()
		plt.close()

	return f[np.argmax(pgram)]


def initModelFit(data, model, **kwargs):
	"""
	Use the Nelder-Mead "Amoeba" algorithm to obtain the fitted Parameters
	for the forward modeling initialization stage. 
	"""

	return best_params, chisquare

def run_mcmc(**kwargs):
	"""
	MCMC routine for the forward modeling.
	"""
