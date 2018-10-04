import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import nirspec_fmp as nsp
import emcee
import corner
import splat
import copy
import time
import os
import sys


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
	modelset = kwargs.get('modelset', 'btsettl08')
	lsf   = kwargs.get('lsf', 6.0)   # instrumental LSF
	tell  = kwargs.get('tell', True) # apply telluric
	data  = kwargs.get('data', None) # for continuum correction and resampling

	# read in a model
	model = nsp.Model(teff=teff, logg=logg, feh=z, order=order, modelset=modelset)
	
	# wavelength offset
	model.wave += wave_offset

	# apply vsini
	model.flux = nsp.broaden(wave=model.wave, 
		flux=model.flux, vbroad=vsini, rotate=True, gaussian=False)
	
	# apply rv (including the barycentric correction)
	model.wave = rvShift(model.wave, rv=rv)
	
	# apply telluric
	if tell is True:
		model = nsp.applyTelluric(model=model, alpha=alpha, airmass='1.5')
	# NIRSPEC LSF
	model.flux = nsp.broaden(wave=model.wave, 
		flux=model.flux, vbroad=lsf, rotate=False, gaussian=True)

	# add a fringe pattern to the model
	#model.flux *= (1+amp*np.sin(freq*(model.wave-phase)))

	# wavelength offset
	#model.wave += wave_offset
	
	# integral resampling
	if data is not None:
		model.flux = np.array(splat.integralResample(xh=model.wave, 
			yh=model.flux, xl=data.wave))
		model.wave = data.wave
		# contunuum correction
		model = nsp.continuum(data=data, mdl=model)

	# flux offset
	model.flux += flux_offset
	#model.flux **= (1 + flux_exponent_offset)

	return model

def rvShift(wavelength, rv):
	"""
	Perform the radial velocity correction.

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
	wavelow               = telluric_data.wave[0]  - 50
	wavehigh              = telluric_data.wave[-1] + 50
	telluric_model        = nsp.getTelluric(wavelow=wavelow,wavehigh=wavehigh)
	telluric_model.flux **= alpha
	# lsf
	telluric_model.flux = nsp.broaden(wave=telluric_model.wave, flux=telluric_model.flux, 
		vbroad=lsf, rotate=False, gaussian=True)
	# resample
	telluric_model.flux = np.array(splat.integralResample(xh=telluric_model.wave, 
		yh=telluric_model.flux, xl=telluric_data.wave))
	telluric_model.wave = telluric_data.wave
	return telluric_model

def getLSF2(telluric_data, continuum=True, test=False, save_path=None):
	"""
	Return a best LSF value from a telluric data.
	"""
	
	data = copy.deepcopy(telluric_data)

	def bestParams(data, i, alpha, c2, c0):

		data2          = copy.deepcopy(data)
		data2.wave     = data2.wave + c0
		telluric_model = nsp.convolveTelluric(i, data2, alpha=alpha)
		model          = nsp.continuum(data=data2, mdl=telluric_model)
		#plt.figure(2)
		#plt.plot(model.wave, model.flux+c2, 'r-', alpha=0.5)
		#plt.plot(data.wave*c1+c0, data.flux, 'b-', alpha=0.5)
		#plt.close()
		#plt.show()
		#sys.exit()
		return model.flux + c2

	def bestParams2(theta, data):

		i, alpha, c2, c0, c1 = theta 
		data2                = copy.deepcopy(data)
		data2.wave           = data2.wave*c1 + c0
		telluric_model       = nsp.convolveTelluric(i, data2, alpha=alpha)
		model                = nsp.continuum(data=data2, mdl=telluric_model)
		return np.sum(data.flux - (model.flux + c2))**2

	from scipy.optimize import curve_fit, minimize

	popt, pcov = curve_fit(bestParams, data, data.flux, p0=[4.01, 1.01, 0.01, 1.01], maxfev=1000000, epsfcn=0.1)

	#nll = lambda *args: bestParams2(*args)
	#results = minimize(nll, [3., 1., 0.1, -10., 1.], args=(data))
	#popt = results['x']

	data.wave      = data.wave+popt[3]

	telluric_model = nsp.convolveTelluric(popt[0], data, alpha=popt[1])
	model          = nsp.continuum(data=data, mdl=telluric_model)

	#model.flux * np.e**(-popt[2]) + popt[3]
	model.flux + popt[2]

	return popt[0]

def getLSF(telluric_data, alpha=1.0, continuum=True,test=False,save_path=None):
	"""
	Return a best LSF value from a telluric data.
	"""
	lsf_list = []
	test_lsf = np.arange(3.0,13.0,0.1)
	
	data = copy.deepcopy(telluric_data)
	if continuum is True:
		data = nsp.continuumTelluric(data=data)

	data.flux **= alpha
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
		plt.plot(data.wave,data.flux,
			'k-',label='telluric data',alpha=0.5)
		plt.title("Test LSF",fontsize=15)
		plt.xlabel("Wavelength ($\AA$)",fontsize=12)
		plt.ylabel("Transmission + Offset",fontsize=12)
		plt.minorticks_on()
		if save_path is not None:
			plt.savefig(save_path+\
				"/{}_O{}_lsf_data_mdl.png"\
				.format(data.name, data.order))
		#plt.show()
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
				.format(data.name, data.order))
		#plt.show()
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
		data = nsp.continuumTelluric(data=data, order=data.order)

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
	tell_sp  = nsp.continuumTelluric(data=tell_sp, order=tell_sp.order)

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

	#f = np.linspace(0.01,10,100000)
	f = np.linspace(1.0,10,100000)

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

def telluric_mcmc(tell_sp, nwalkers=30, step=400, burn=300, priors=None, moves=2.0, save=True, save_to_path=None):
	"""
	MCMC routine for telluric standard stars to obtain the LSF and alpha. This function utilizes the emcee package.

	Parameters
	----------
	tell_sp 	:	Spectrum object
					telluric spectrum
	nwalkers 	:	int
					number of walkers. default is 30.
	step 		:	int
					number of steps. default is 400
	burn		:	int
					burn in mcmc to compute the best parameters. default is 300.
	priors 		: 	dic
					A prior dictionary that specifies the range of the priors.
					Keys 	'lsf_min'  , 'lsf_max'  : LSF min/max in km/s
							'alpha_min', 'alpha_max': alpha min/max
							'A_min'    , 'A_max'	: flux offset in cnts/s
							'B_min'    , 'B_max'	: wave offset in Angstroms
					If there is no input priors dictionary, a wide range of priors will be adopted.
	moves		:	float
					the stretch scale parameter. default is 2.0 (same as emcee).
	save 		:	boolean
					save the modeled lsf and alpha in the header. default is True.
	save_to_path: 	str
					the path to save the mcmc outputs.

	Examples
	--------
	>>> import nirspec_fmp as nsp
	>>> tell_sp = nsp.Spectrum(name='jan19s0024_calibrated',order=33)
	>>> nsp.telluric_mcmc(tell_sp)

	"""

	## Initial parameters setup
	tell_data_name       = tell_sp.name
	tell_path            = tell_sp.path
	order                = tell_sp.order
	ndim                 = 4
	applymask            = False
	pixel_start          = 10
	pixel_end            = -30

	if priors is None:
		priors =  { 'lsf_min':4.0,  	'lsf_max':10.0,
					'alpha_min':0.3,  	'alpha_max':2.0,
					'A_min':-1.0,    	'A_max':1.0,
					'B_min':-0.5,		'B_max':0.5    	}

	if save_to_path is None:
		save_to_path = './mcmc'

	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)

	tell_sp.wave    = tell_sp.wave[pixel_start:pixel_end]
	tell_sp.flux    = tell_sp.flux[pixel_start:pixel_end]
	tell_sp.noise   = tell_sp.noise[pixel_start:pixel_end]

	data = copy.deepcopy(tell_sp)

	## MCMC functions
	def makeTelluricModel(lsf, alpha, flux_offset, wave_offset0, data=data):
		"""
		Make a telluric model as a function of LSF, alpha, and flux offset.
		"""
		data2               = copy.deepcopy(data)
		data2.wave          = data2.wave + wave_offset0
		#data2.wave          = data2.wave * (1 + wave_offset1) + wave_offset0
		telluric_model      = nsp.convolveTelluric(lsf, data2, alpha=alpha)
		model               = nsp.continuum(data=data2, mdl=telluric_model, deg=2)
		model.flux         += flux_offset

		return model

	## log file
	log_path = save_to_path + '/mcmc_parameters.txt'

	file_log = open(log_path,"w+")
	file_log.write("tell_path {} \n".format(tell_path))
	file_log.write("tell_name {} \n".format(tell_data_name))
	file_log.write("order {} \n".format(order))
	file_log.write("ndim {} \n".format(ndim))
	file_log.write("nwalkers {} \n".format(nwalkers))
	file_log.write("step {} \n".format(step))
	file_log.write("pixel_start {} \n".format(pixel_start))
	file_log.write("pixel_end {} \n".format(pixel_end))
	file_log.write("moves {} \n".format(moves))
	file_log.close()

	## MCMC for the parameters LSF, alpha, and a nuisance parameter for flux offset
	def lnlike(theta, data=data):
		"""
		Log-likelihood, computed from chi-squared.

		Parameters
		----------
		theta
		data

		Returns
		-------
		-0.5 * chi-square + sum of the log of the noise
		"""
		## Parameters MCMC
		lsf, alpha, A, B = theta

		model = makeTelluricModel(lsf, alpha, A, B, data=data)

		chisquare = nsp.chisquare(data, model)

		return -0.5 * (chisquare + np.sum(np.log(2*np.pi*data.noise**2)))

	def lnprior(theta):
		"""
		Specifies a flat prior
		"""
		## Parameters for theta
		lsf, alpha, A, B = theta

		limits =  { 'lsf_min':1.0  ,  'lsf_max':20.0,
					'alpha_min':0.3,  'alpha_max':2.0,
					'A_min':-1.0   ,  'A_max':1.0,
					'B_min':-0.5   ,  'B_max':0.5    }

		if  limits['lsf_min']   < lsf  < limits['lsf_max'] \
		and limits['alpha_min'] < alpha < limits['alpha_max']\
		and limits['A_min']     < A     < limits['A_max']\
		and limits['B_min']     < B     < limits['B_max']:
			return 0.0

		return -np.inf

	def lnprob(theta, data):
		lnp = lnprior(theta)

		if not np.isfinite(lnp):
			return -np.inf
		return lnp + lnlike(theta, data)

	pos = [np.array([priors['lsf_min']   + (priors['lsf_max']    - priors['lsf_min'] )  * np.random.uniform(), 
					 priors['alpha_min'] + (priors['alpha_max']  - priors['alpha_min']) * np.random.uniform(),
					 priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
					 priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform()]) for i in range(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves)

	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()
	print('total time: ',(time2-time1)/60,' min.')
	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
	print(sampler.acceptance_fraction)

	np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])
	
	samples = sampler.chain[:, :, :].reshape((-1, ndim))

	np.save(save_to_path + '/samples', samples)

	# create walker plots
	sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
	samples = np.load(save_to_path + '/samples.npy')

	ylabels = ["LSF (km/s)", "alpha", "flux offset", "wave offset0"]

	## create walker plots
	plt.rc('font', family='sans-serif')
	plt.tick_params(labelsize=30)
	fig = plt.figure(tight_layout=True)
	gs = gridspec.GridSpec(ndim, 1)
	gs.update(hspace=0.1)

	for i in range(ndim):
		ax = fig.add_subplot(gs[i, :])
		for j in range(nwalkers):
			ax.plot(np.arange(1,int(step+1)), sampler_chain[j,:,i],'k',alpha=0.2)
		ax.set_ylabel(ylabels[i])
	fig.align_labels()
	plt.minorticks_on()
	plt.xlabel('nstep')
	plt.savefig(save_to_path+'/walker.png', dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

	# create array triangle plots
	triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
	#print(triangle_samples.shape)

	# create the final spectra comparison
	lsf_mcmc, alpha_mcmc, A_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
		zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

	# add the summary to the txt file
	file_log = open(log_path,"a")
	file_log.write("*** Below is the summary *** \n")
	file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
	file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
	file_log.write("lsf_mcmc {} km/s\n".format(str(lsf_mcmc)))
	file_log.write("alpha_mcmc {}\n".format(str(alpha_mcmc)))
	file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
	file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
	#file_log.write("C_mcmc {}\n".format(str(C_mcmc)))
	file_log.close()

	print(lsf_mcmc, alpha_mcmc, A_mcmc, B_mcmc)

	if '_' in tell_sp.name:
		tell_data_name = tell_sp.name.split('_')[0]

	## triangular plots
	plt.rc('font', family='sans-serif')
	fig = corner.corner(triangle_samples, 
		labels=ylabels,
		truths=[lsf_mcmc[0], 
		alpha_mcmc[0],
		A_mcmc[0],
		B_mcmc[0]],
		quantiles=[0.16, 0.84],
		label_kwargs={"fontsize": 20})
	plt.minorticks_on()
	fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

	data2               = copy.deepcopy(data)
	data2.wave          = data2.wave + B_mcmc[0]
	telluric_model      = nsp.convolveTelluric(lsf_mcmc[0], data, alpha=alpha_mcmc[0])
	model, pcont        = nsp.continuum(data=data, mdl=telluric_model, deg=2, tell=True)
	model.flux         += A_mcmc[0]

	plt.figure(figsize=(20,8))
	plt.plot(model.wave, model.flux, 'r-', alpha=0.5)
	plt.plot(model.wave, np.polyval(pcont, model.wave) + A_mcmc[0], c='crimson', alpha=0.5)
	plt.plot(data.wave, data.flux, 'k-', alpha=0.5)
	plt.plot(data.wave, data.flux-(model.flux+A_mcmc[0]),'k-', alpha=0.5)
	plt.figtext(0.89,0.86,"{} O{}".format(tell_data_name, order),
		color='k',
		horizontalalignment='right',
		verticalalignment='center',
		fontsize=12)
	plt.figtext(0.89,0.83,"${0}^{{+{1}}}_{{-{2}}}/{3}^{{+{4}}}_{{-{5}}}/{6}^{{+{7}}}_{{-{8}}}$".format(\
		round(lsf_mcmc[0],2),
		round(lsf_mcmc[1],2),
		round(lsf_mcmc[2],2),
		round(alpha_mcmc[0],2),
		round(alpha_mcmc[1],2),
		round(alpha_mcmc[2],2),
		round(A_mcmc[0],2),
		round(A_mcmc[1],2),
		round(A_mcmc[2],2)),
		color='r',
		horizontalalignment='right',
		verticalalignment='center',
		fontsize=12)
	plt.figtext(0.89,0.80,r"$\chi^2$ = {}, DOF = {}".format(\
		round(nsp.chisquare(data,model)), len(data.wave-ndim)),
		color='k',
		horizontalalignment='right',
		verticalalignment='center',
		fontsize=12)
	plt.fill_between(data.wave, -data.noise, data.noise, alpha=0.5)
	plt.tick_params(labelsize=15)
	plt.ylabel('Flux (counts/s)',fontsize=15)
	plt.xlabel('Wavelength ($\AA$)',fontsize=15)
	plt.savefig(save_to_path+'/telluric_spectrum.png',dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

	if save is True:
		data_path = tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits'
		with fits.open(data_path) as hdulist:
			hdulist[0].header['LSF']   = lsf_mcmc[0]
			hdulist[0].header['ALPHA'] = alpha_mcmc[0]
			try:
				hdulist.writeto(data_path,overwrite=True)
			except FileNotFoundError:
				hdulist.writeto(data_path)

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
