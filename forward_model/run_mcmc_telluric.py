import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.gridspec as gridspec
from astropy.io import fits
import emcee
#from schwimmbad import MPIPool
from multiprocessing import Pool
import nirspec_fmp as nsp
import corner
import os
import sys
import time
import copy
import argparse
import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description="Run the forward-modeling routine for telluric files",\
	usage="run_mcmc_telluric.py order date_obs tell_data_name tell_path save_to_path")

parser.add_argument("order",metavar='o',type=int,
    default=None, help="order", nargs="+")

parser.add_argument("date_obs",type=str,
    default=None, help="source name", nargs="+")

parser.add_argument("tell_data_name",type=str,
    default=None, help="telluric data name", nargs="+")

parser.add_argument("tell_path",type=str,
    default=None, help="telluric data path", nargs="+")

parser.add_argument("save_to_path",type=str,
    default=None, help="output path", nargs="+")

parser.add_argument("-ndim",type=int,
    default=4, help="number of dimension; default 4")

parser.add_argument("-nwalkers",type=int,
    default=50, help="number of walkers of MCMC; default 50")

parser.add_argument("-step",type=int,
    default=400, help="number of steps of MCMC; default 400")

parser.add_argument("-burn",type=int,
    default=300, help="burn of MCMC; default 300")

parser.add_argument("-moves",type=float,
    default=2.0, help="moves of MCMC; default 2.0")

parser.add_argument("-priors",type=dict,
    default=None, help="priors of MCMC; default None")

parser.add_argument("-pixel_start",type=int,
    default=10, help="starting pixel index for the science data; default 10")

parser.add_argument("-pixel_end",type=int,
    default=-40, help="ending pixel index for the science data; default -40")

parser.add_argument("-applymask",type=bool,
    default=False, help="apply a simple mask based on the STD of the average flux; default is False")

parser.add_argument("-save",type=bool,
    default=True, help="save the fitted LSF and telluric alpha to the fits file header; default is True")

args = parser.parse_args()

######################################################################################################


order                  = int(args.order[0])
date_obs               = str(args.date_obs[0])
tell_data_name         = str(args.tell_data_name[0])
tell_path              = str(args.tell_path[0])
save_to_path           = str(args.save_to_path[0])
ndim, nwalkers, step   = int(args.ndim), int(args.nwalkers), int(args.step)
burn                   = int(args.burn)
moves                  = float(args.moves)
priors                 = args.priors
applymask              = args.applymask
pixel_start, pixel_end = int(args.pixel_start), int(args.pixel_end)
save                   = args.save

if order == 35: applymask = True

tell_data_name2 = tell_data_name + '_calibrated'
tell_sp         = nsp.Spectrum(name=tell_data_name2, order=order, path=tell_path, applymask=applymask)

###########################################################################################################
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

"""

## Initial parameters setup
tell_data_name       = tell_sp.name
tell_path            = tell_sp.path
order                = tell_sp.order
ndim                 = 4
applymask            = False
pixel_start          = 10
pixel_end            = -30

## add a pixel label for plotting
length1     = len(tell_sp.oriWave)
pixel       = np.delete(np.arange(length1),tell_sp.mask)
pixel       = pixel[pixel_start:pixel_end]

if priors is None:
	priors =  { 'lsf_min':3.0,  	'lsf_max':6.0,
				'alpha_min':0.3,  	'alpha_max':3.0,
				'A_min':-1.0,    	'A_max':1.0,
				'B_min':-0.02,		'B_max':0.02    	}

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
	
	if data.order == 35:
		from scipy.optimize import curve_fit
		popt, pcov = curve_fit(nsp.voigt_profile,data2.wave[10:-10], data2.flux[10:-10], 
			p0=[21660,2000,0.1,0.1,0.01,0.1,10000,1000], maxfev=10000)
		model               = nsp.continuum(data=data2, mdl=telluric_model, deg=2)
		model.flux         *= nsp.voigt_profile(data2.wave, *popt)
	else:
		model               = nsp.continuum(data=data2, mdl=telluric_model, deg=2)
	
	model.flux             += flux_offset

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

	limits =  { 'lsf_min':2.0  	,  'lsf_max':10.0,
				'alpha_min':0.3 ,  'alpha_max':4.0,
				'A_min':-500.0 	,  'A_max':500.0,
				'B_min':-0.04   ,  'B_max':0.04    }

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

with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves, pool=pool)
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

ylabels = ["$\Delta \\nu_{inst}$ (km/s)", "$\\alpha$", "$F_{\lambda}$ offset", "$\lambda$ offset ($\AA$)"]

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
#plt.show()
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
#plt.show()
plt.close()

data2               = copy.deepcopy(data)
data2.wave          = data2.wave + B_mcmc[0]
telluric_model      = nsp.convolveTelluric(lsf_mcmc[0], data, alpha=alpha_mcmc[0])
model, pcont        = nsp.continuum(data=data, mdl=telluric_model, deg=2, tell=True)
model.flux         += A_mcmc[0]

plt.tick_params(labelsize=20)
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(111)
ax1.plot(model.wave, model.flux, c='C3', ls='-', alpha=0.5)
ax1.plot(model.wave, np.polyval(pcont, model.wave) + A_mcmc[0], c='C1', ls='-', alpha=0.5)
ax1.plot(data.wave, data.flux, 'k-', alpha=0.5)
ax1.plot(data.wave, data.flux-(model.flux+A_mcmc[0]),'k-', alpha=0.5)
ax1.minorticks_on()
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
	round(nsp.chisquare(data,model)), round(len(data.wave-ndim)/3)),
	color='k',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.fill_between(data.wave, -data.noise, data.noise, alpha=0.5)
plt.tick_params(labelsize=15)
plt.ylabel('Flux (counts/s)',fontsize=15)
plt.xlabel('Wavelength ($\AA$)',fontsize=15)

ax2 = ax1.twiny()
ax2.plot(pixel, data.flux, color='w', alpha=0)
ax2.set_xlabel('Pixel',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xlim(pixel[0], pixel[-1])
ax2.minorticks_on()
	
plt.savefig(save_to_path+'/telluric_spectrum.png',dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

###################################################################################################
## mask the bad data
outlier_rejection = 3

mask_region = np.where( np.abs( data.flux-  model.flux ) > outlier_rejection*np.std( data.flux - model.flux ) )

data.wave = data.wave[mask_region]
data.flux = data.flux[mask_region]

## start based on the previous results

pos = [np.array([	lsf_mcmc[0]   + ( np.max( lsf_mcmc[1], 		lsf_mcmc[2] ) )  * np.random.normal(), 
				 	alpha_mcmc[0] + ( np.max( alpha_mcmc[1], 	alpha_mcmc[2] ) )  * np.random.normal(), 
				 	A_mcmc[0]   + ( np.max( A_mcmc[1], A_mcmc[2] ) )  * np.random.normal(), 
				 	B_mcmc[0]   + ( np.max( B_mcmc[1], B_mcmc[2] ) )  * np.random.normal(), ]) for i in range(nwalkers)]

## run mcmc

with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), a=moves, pool=pool)
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
sampler_chain = np.load(save_to_path + '/sampler_chain2.npy')
samples = np.load(save_to_path + '/samples2.npy')

ylabels = ["$\Delta \\nu_{inst}$ (km/s)", "$\\alpha$", "$F_{\lambda}$ offset", "$\lambda$ offset ($\AA$)"]

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
plt.savefig(save_to_path+'/walker2.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# create array triangle plots
triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
#print(triangle_samples.shape)

# create the final spectra comparison
lsf_mcmc, alpha_mcmc, A_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
	zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))

# add the summary to the txt file
file_log = open(log_path,"a")
file_log.write("*** Below is the summary 2 *** \n")
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
fig.savefig(save_to_path+'/triangle2.png', dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

data2               = copy.deepcopy(data)
data2.wave          = data2.wave + B_mcmc[0]
telluric_model      = nsp.convolveTelluric(lsf_mcmc[0], data, alpha=alpha_mcmc[0])
model, pcont        = nsp.continuum(data=data, mdl=telluric_model, deg=2, tell=True)
model.flux         += A_mcmc[0]

plt.tick_params(labelsize=20)
fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(111)
ax1.plot(model.wave, model.flux, c='C3', ls='-', alpha=0.5)
ax1.plot(model.wave, np.polyval(pcont, model.wave) + A_mcmc[0], c='C1', ls='-', alpha=0.5)
ax1.plot(data.wave, data.flux, 'k-', alpha=0.5)
ax1.plot(data.wave, data.flux-(model.flux+A_mcmc[0]),'k-', alpha=0.5)
ax1.minorticks_on()
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
	round(nsp.chisquare(data,model)), round(len(data.wave-ndim)/3)),
	color='k',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.fill_between(data.wave, -data.noise, data.noise, alpha=0.5)
plt.tick_params(labelsize=15)
plt.ylabel('Flux (counts/s)',fontsize=15)
plt.xlabel('Wavelength ($\AA$)',fontsize=15)

ax2 = ax1.twiny()
ax2.plot(pixel, data.flux, color='w', alpha=0)
ax2.set_xlabel('Pixel',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xlim(pixel[0], pixel[-1])
ax2.minorticks_on()
	
plt.savefig(save_to_path+'/telluric_spectrum2.png',dpi=300, bbox_inches='tight')
#plt.show()
plt.close()


###################################################################################################

if save is True:
	data_path = tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits'
	with fits.open(data_path) as hdulist:
		hdulist[0].header['LSF']       = lsf_mcmc[0]
		hdulist[0].header['ALPHA']     = alpha_mcmc[0]
		#hdulist[0].header['WFIT0NEW'] += A_mcmc[0]
		try:
			hdulist.writeto(data_path, overwrite=True)
		except FileNotFoundError:
			hdulist.writeto(data_path)
