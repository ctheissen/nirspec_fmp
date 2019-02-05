import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from astropy.io import fits
from astropy import units as u
import sys
import os
import warnings
import copy
import nirspec_fmp as nsp
warnings.filterwarnings("ignore")


class Spectrum():
	"""
	The spectrum class for reading the reduced NIRSPEC data by NSDRP.

	Parameters
	----------
	name : str
	       The data filename before the order number.
	       Ex. name='jan19s0022'
	order: int
	       The order of the spectra.
	path : str
	       The directory where the reduced data is.

	Returns
	-------
	flux : numpy.ndarray
	       The flux of the spectrum.
	wave : numpy.ndarray
	       The wavelength of the spectrum
	noise: numpy.ndarray
	       The noise of the spectrum
	sky  : numpy.ndarray
	       The sky emission
	plot : matplotlib plot
	       plot the spectrum (with noise on/off option)

	Examples
	--------
	>>> import nirspec_pip as nsp
	>>> path = '/path/to/reducedData'
	>>> data = nsp.Spectrum(name='jan19s0022', order=33, path=path)
	>>> data.plot()

	"""
	def __init__(self, **kwargs):
		self.name = kwargs.get('name')
		self.order = kwargs.get('order')
		self.path = kwargs.get('path')
		self.applymask = kwargs.get('applymask',False)
		#self.manaulmask = kwargs('manaulmask', False)

		if self.path == None:
			self.path = './'

		fullpath = self.path + '/' + self.name + '_' + str(self.order) + '_all.fits'

		hdulist = fits.open(fullpath, ignore_missing_end=True)

		#The indices 0 to 3 correspond to wavelength, flux, noise, and sky
		self.header= hdulist[0].header
		self.wave  = hdulist[0].data
		self.flux  = hdulist[1].data
		self.noise = hdulist[2].data
		try:
			self.sky = hdulist[3].data
		except IndexError:
			print("No sky line data.")
			self.sky = np.zeros(self.wave.shape) 
		self.mask  = []

		# define a list for storing the best wavelength shift
		self.bestshift = []

		# store the original parameters
		self.oriWave  = hdulist[0].data
		self.oriFlux  = hdulist[1].data
		self.oriNoise = hdulist[2].data

		if self.applymask:
			# set up masking criteria
			self.avgFlux = np.mean(self.flux)
			self.stdFlux = np.std(self.flux)

			self.smoothFlux = self.flux
			# set the outliers as the flux below 
			self.smoothFlux[self.smoothFlux <= self.avgFlux-2*self.stdFlux] = 0
		
			self.mask  = np.where(self.smoothFlux <= 0)
			self.wave  = np.delete(self.wave, list(self.mask))
			self.flux  = np.delete(self.flux, list(self.mask))
			self.noise = np.delete(self.noise, list(self.mask))
			self.sky   = np.delete(self.sky, list(self.mask))
			self.mask  = self.mask[0]

	def customMask(self, custom_mask):
		"""
		Mask the pixels by a self-defined list.
		"""
		## combine the list and remove the duplicates
		self.mask  =  list(set().union(self.mask, custom_mask))

		self.wave  = np.delete(self.oriWave, list(self.mask))
		self.flux  = np.delete(self.oriFlux, list(self.mask))
		self.noise = np.delete(self.oriNoise, list(self.mask))

		return self


	def maskBySigmas(self, sigma=2):
		"""
		Mask the outlier data points by sigmas.
		"""
		# set up masking criteria
		self.avgFlux = np.mean(self.flux)
		self.stdFlux = np.std(self.flux)

		self.smoothFlux = self.flux
		# set the outliers as the flux below 
		self.smoothFlux[self.smoothFlux <= self.avgFlux - sigma * self.stdFlux] = 0
		
		self.mask  = np.where(self.smoothFlux <= 0)
		self.wave  = np.delete(self.wave, list(self.mask))
		self.flux  = np.delete(self.flux, list(self.mask))
		self.noise = np.delete(self.noise, list(self.mask))
		self.sky   = np.delete(self.sky, list(self.mask))
		self.mask  = self.mask[0]

	def plot(self, **kwargs):
		"""
		Plot the spectrum.
		"""
		#xlim   = kwargs.get('xrange', [self.wave[0], self.wave[-1]])
		#ylim   = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
		items  = kwargs.get('items', ['spec','noise'])
		title  = kwargs.get('title')
		mask   = kwargs.get('mask', True)
		save   = kwargs.get('save', False)
		output = kwargs.get('output', str(self.name) + '.png')
		
		plt.figure(figsize=(16,6))
		plt.rc('font', family='sans-serif')
		## Plot masked spectrum
		if ('spectrum' in items) or ('spec' in items):
			if "_" in self.name:
				plot_name = self.name.split("_")[0]
			else:
				plot_name = self.name
			if mask:
				plt.plot(self.wave, self.flux, color='k', 
					alpha=.8, linewidth=1, 
					label="{} O{}".format(plot_name,self.order))
			if not mask:
				plt.plot(self.oriWave, self.oriFlux, color='k', 
					alpha=.8, linewidth=1, 
					label="{} O{}".format(plot_name,self.order))

		## Plot spectrum noise
		if 'noise' in items:
			if mask:
				plt.fill_between(self.wave, -self.noise, self.noise,
					color='gray', linewidth=1, alpha=.6)
			elif not mask:
				plt.fill_between(self.oriWave, -self.oriNoise, 
					self.oriNoise,
					color='gray', linewidth=1, alpha=.6)

		plt.legend(fontsize=12)
		#plt.xlim(xlim)
		#plt.ylim(ylim)    
    
		plt.xlabel('Wavelength [$\AA$]', fontsize=18)
		plt.ylabel('Flux (cnts/s)', fontsize=18)
		plt.minorticks_on()
		plt.tick_params(axis='both', labelsize=18)

		if title != None:
			plt.title(title, fontsize=20)

		if save == True:
			plt.savefig(output)

		plt.show()
		plt.close()

	def writeto(self, save_to_path, method='ascii',
		tell_sp=None):
		"""
		Save the data as an ascii or a fits file.

		Parameters
		----------
		save_to_path 	:	str
							the path to save the output file

		method 			: 	'ascii' or 'fits'
							the output file format, either in
							a single ascii file or several fits
							files labeled in the order of 
							wavelength


		Optional Parameters
		-------------------
		tell_sp 		: 	Spectrum object
							the telluric data for the corresponding
							wavelength calibration

		Returns
		-------
		ascii or fits 	: 	see the method keyword
							The wavelength is in microns


		"""
		#pixel = np.delete(np.arange(1024),list(self.mask))
		pixel = np.arange(len(self.oriWave))
		## create the output mask array 0=good; 1=bad
		if self.applymask:
			mask = np.zeros((len(self.oriWave),),dtype=int)
			np.put(mask,self.mask.tolist(),int(1))
		else:
			mask = np.zeros((len(self.oriWave),),dtype=int)

		if method == 'fits':
			#fullpath = self.path + '/' + self.name + '_' + str(self.order) + '_all.fits'
			#hdulist = fits.open(fullpath, ignore_missing_end=True)
			#hdulist.writeto(save_to_path)
			#hdulist.close()
			save_to_path2 = save_to_path + self.header['FILENAME'].split('.')[0]\
			+ '_O' + str(self.order)
			## wavelength
			hdu1 = fits.PrimaryHDU(self.wave/10000, header=self.header)
			save_to_path2_1 = save_to_path2 + '_wave.fits'
			hdu1.writeto(save_to_path2_1)
			## flux
			hdu2 = fits.PrimaryHDU(self.flux, header=self.header)
			save_to_path2_2 = save_to_path2 + '_flux.fits'
			hdu2.writeto(save_to_path2_2)
			## uncertainty
			hdu3 = fits.PrimaryHDU(self.noise, header=self.header)
			save_to_path2_3 = save_to_path2 + '_uncertainty.fits'
			hdu3.writeto(save_to_path2_3)
			## pixel
			hdu4 = fits.PrimaryHDU(pixel, header=self.header)
			save_to_path2_4 = save_to_path2 + '_pixel.fits'
			hdu4.writeto(save_to_path2_4)
			## mask
			hdu5 = fits.PrimaryHDU(mask, header=self.header)
			save_to_path2_5 = save_to_path2 + '_mask.fits'
			hdu5.writeto(save_to_path2_5)

			if tell_sp is not None:
				tell_sp2 = copy.deepcopy(tell_sp)
				# the telluric standard model
				wavelow = tell_sp2.wave[0] - 20
				wavehigh = tell_sp2.wave[-1] + 20
				tell_mdl = nsp.getTelluric(wavelow=wavelow,wavehigh=wavehigh)
				# continuum correction for the data
				tell_sp2 = nsp.continuumTelluric(data=tell_sp2, 
					model=tell_mdl,order=tell_sp2.order)
				# telluric flux
				hdu6 = fits.PrimaryHDU(tell_sp.flux, header=tell_sp.header)
				save_to_path2_6 = save_to_path2 + '_telluric_flux.fits'
				hdu5.writeto(save_to_path2_6)
				# telluric uncertainty
				hdu7 = fits.PrimaryHDU(tell_sp.noise, header=tell_sp.header)
				save_to_path2_7 = save_to_path2 + '_telluric_uncertainty.fits'
				hdu5.writeto(save_to_path2_7)
				# telluric model
				hdu8 = fits.PrimaryHDU(tell_mdl.flux, header=tell_sp.header)
				save_to_path2_8 = save_to_path2 + '_telluric_model.fits'
				hdu5.writeto(save_to_path2_8)
				

		elif method == 'ascii':
			save_to_path2 = save_to_path + self.header['FILENAME'].split('.')[0]\
			+ '_O' + str(self.order) + '.txt'

			if tell_sp is None:
				df = pd.DataFrame(data={'1_wavelength':list(self.oriWave/10000),
					'2_flux':list(self.oriFlux),
					'3_uncertainty':list(self.oriNoise),
					'4_pixel':list(pixel),
					'5_mask':list(mask)})
			
			elif tell_sp is not None:
				tell_sp2 = copy.deepcopy(tell_sp)
				tell_sp2 = nsp.continuumTelluric(data=tell_sp2
					,order=self.order)
				lsf0 = nsp.getLSF(tell_sp2)
				tell_sp2.flux = tell_sp2.oriFlux
				tell_sp2.wave = tell_sp2.oriWave
				tell_mdl = nsp.convolveTelluric(lsf0, tell_sp2)

				df = pd.DataFrame(data={'wavelength':list(self.oriWave/10000),
					'flux':list(self.oriFlux),
					'uncertainty':list(self.oriNoise),
					'telluric_flux':list(tell_sp.oriFlux),
					'telluric_uncertainty':list(tell_sp.oriNoise),
					'telluric_model':list(tell_mdl.flux),
					'pixel':list(pixel),
					'mask':list(mask)})


			df.to_csv(save_to_path2, index=None, sep='\t', mode='a',
				header=True, columns=['wavelength', 'flux', 'uncertainty', 
				'telluric_flux', 'telluric_uncertainty', 'telluric_model',
				'pixel', 'mask'])


	def coadd(self, sp, method='pixel'):
		"""
		Coadd individual extractions, either in pixel space or
		wavelength space.

		Parameters
		----------
		sp 		: 	Spectrum object
					spectrum to be coadded

		method 	: 	'pixel' or 'wavelength'
					coadd based on adding pixels or wavelength
					If 'wavelength', the second spectrum would be
					10x supersample and then cross correlated
					to be optimally shifted and coadded

		Returns
		-------
		self 	: 	Spectrum object
					coadded spectra

		"""
		if method == 'pixel':
			w1 = 1/self.oriNoise**2
			w2 = 1/sp.oriNoise**2
			self.oriFlux = (self.oriFlux*w1 + sp.oriFlux*w2)/(w1 + w2)
			self.oriNoise = np.sqrt(1/(w1 + w2))
			## set up masking criteria
			self.avgFlux = np.mean(self.oriFlux)
			self.stdFlux = np.std(self.oriFlux)
			self.smoothFlux = self.oriFlux
			## set the outliers as the flux below 
			if self.applymask:
				self.smoothFlux[self.smoothFlux <= self.avgFlux-2*self.stdFlux] = 0
				self.mask = np.where(self.smoothFlux <= 0)
			else:
				self.mask = []
			self.wave  = np.delete(self.oriWave, list(self.mask))
			self.flux  = np.delete(self.oriFlux, list(self.mask))
			self.noise = np.delete(self.oriNoise, list(self.mask))

		elif method == 'wavelength':
			self_supers = copy.deepcopy(self)
			g = interpolate.interp1d(self.wave, self.flux)
			sp_supers = copy.deepcopy(sp)
			f = interpolate.interp1d(sp.wave, sp.flux)
			## 10x supersample the average difference of 
			## the wavelength
			#step0 = np.mean(np.diff(self.wave))/10
			#self_supers.wave = np.arange(self.wave[0],
			#	self.wave[-1],step0)
			self_supers.flux = g(self_supers.wave)
			self_supers.oriWave = np.arange(self.oriWave[0],
				self.oriWave[-1],(self.oriWave[-1]-self.oriWave[0])/10240)
			g1 = interpolate.interp1d(self.oriWave, self.oriFlux)
			self_supers.oriFlux = g1(self_supers.oriWave)

			#step = np.mean(np.diff(sp.wave))/10
			#sp_supers.wave = np.arange(sp.wave[0],sp.wave[-1],step)
			#sp_supers.flux = f(sp_supers.wave)
			sp_supers.oriWave = np.arange(sp.oriWave[0],
				sp.oriWave[-1],(sp.oriWave[-1]-sp.oriWave[0])/10240)
			f1 = interpolate.interp1d(sp.oriWave, sp.oriFlux)
			sp_supers.oriFlux = f1(sp_supers.oriWave)

			## calculate the max cross correlation value
			def xcorr(a0,b0,shift):
				"""
				Shift is the index number after supersampling 
				both of the spectra.
				"""
				a = copy.deepcopy(a0)
				b = copy.deepcopy(b0)

				## shift the wavelength of b
				length = b.oriFlux.shape[0]
				if shift >= 0:
					mask_a = np.arange(0,shift,1)
					a.oriFlux = np.delete(a.oriFlux,mask_a)
					mask_b = np.arange(length-1,length-shift-1,-1)
					b.oriFlux = np.delete(b.oriFlux,mask_b)

				elif shift < 0:
					mask_a = np.arange(length-1,length+shift-1,-1)
					a.oriFlux = np.delete(a.oriFlux,mask_a)
					mask_b = np.arange(0,-shift,1)
					b.oriFlux = np.delete(b.oriFlux,mask_b)

				## shift the wavelength of b
				#b.wave += shift * step
				## discard the points where the wavelength values
				## are larger
				#condition = (a.wave > b.wave[0]) & (a.wave < b.wave[-1])
				
				#a.flux = a.flux[np.where(condition)]
				#a.wave = a.wave[np.where(condition)]
				## resampling the telluric model
				#b.flux = np.array(nsp.integralResample(xh=b.wave, 
				#	yh=b.flux, xl=a.wave))
				
				return np.inner(a.oriFlux, b.oriFlux)/\
				(np.average(a.oriFlux)*np.average(b.oriFlux))/a.oriFlux.shape[0]

			xcorr_list = []
			## mask the ending pixels
			self_supers2 = copy.deepcopy(self_supers)
			sp_supers2 = copy.deepcopy(sp_supers)
			self_supers2.wave = self_supers2.wave[1000:-1000]
			self_supers2.flux = self_supers2.flux[1000:-1000]
			sp_supers2.wave = sp_supers2.wave[1000:-1000]
			sp_supers2.flux = sp_supers2.flux[1000:-1000]
			for shift in np.arange(-10,10,1):
				xcorr_list.append(xcorr(self_supers2,sp_supers2,shift))

			## dignostic plot for cc result
			fig, ax = plt.subplots()
			ax.plot(np.arange(-10,10,1),np.array(xcorr_list),'k-')
			plt.show()
			plt.close()

			step = np.absolute(np.mean(np.diff(sp_supers.wave)))
			bestshift = np.arange(-10*step,10*step,step)[np.argmax(xcorr_list)]
			sp_supers.oriWave += bestshift
			## discard the points where the wavelength values
			## are larger
			condition = (self.oriWave > sp_supers.oriWave[0])\
			& (self.oriWave < sp_supers.oriWave[-1])

			self.oriFlux = self.oriFlux[np.where(condition)]
			self.oriWave = self.oriWave[np.where(condition)]
			self.oriNoise = self.oriNoise[np.where(condition)]
			sp_supers.oriNoise = sp_supers.oriNoise[np.where(condition)]
			sp_supers.oriFlux = np.array(nsp.integralResample(xh=sp_supers.oriWave, 
				yh=sp_supers.oriFlux, xl=self.oriWave))

			w1 = 1/self.oriNoise**2
			w2 = 1/sp_supers.oriNoise**2
			self.oriFlux = (self.oriFlux*w1 + sp_supers.oriFlux*w2)/(w1 + w2)
			self.oriNoise = np.sqrt(1/(w1 + w2))
			## set up masking criteria
			self.avgFlux = np.mean(self.oriFlux)
			self.stdFlux = np.std(self.oriFlux)
			self.smoothFlux = self.oriFlux
			## set the outliers as the flux below 
			self.smoothFlux[self.smoothFlux <= self.avgFlux-2*self.stdFlux] = 0
			self.mask = np.where(self.smoothFlux <= 0)
			self.wave  = np.delete(self.oriWave, list(self.mask))
			self.flux  = np.delete(self.oriFlux, list(self.mask))
			self.noise = np.delete(self.oriNoise, list(self.mask))

		return self

	def updateWaveSol(self, tell_sp, length1=1024):
		"""
		Return a new wavelength solution given a wavelength 
		calibrated telluric spectrum.

		Parameters
		----------
		tell_sp 	: 	Spectrum object
						the calibrated telluric spectra
		"""
		wfit0 = tell_sp.header['WFIT0NEW']
		wfit1 = tell_sp.header['WFIT1NEW']
		wfit2 = tell_sp.header['WFIT2NEW']
		wfit3 = tell_sp.header['WFIT3NEW']
		wfit4 = tell_sp.header['WFIT4NEW']
		wfit5 = tell_sp.header['WFIT5NEW']
		c3    = tell_sp.header['c3']
		c4    = tell_sp.header['c4']

		self.wave = np.delete(nsp.waveSolution(np.arange(length1),
			wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4, order=self.order), list(self.mask))
		self.oriWave = nsp.waveSolution(np.arange(length1),
			wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4, order=self.order)

		return self









