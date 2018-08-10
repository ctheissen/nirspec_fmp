#!/usr/bin/env python
#
# Nov. 22 2017
# @Dino Hsu
#
# Generate a spectrum class by importing the reduced
# NIRSPEC data
#
# Refer to the Jessica's Apogee_tools
#

import nirspec_pip as nsp
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from astropy import units as u
import warnings
import copy
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
		self.sky   = hdulist[3].data
		self.mask  = []

		# define a list for storing the best wavelength shift
		self.bestshift = []

		# store the original parameters
		self.oriWave  = hdulist[0].data
		self.oriFlux  = hdulist[1].data
		self.oriNoise = hdulist[2].data

		#set up masking criteria
		self.avgFlux = np.mean(self.flux)
		self.stdFlux = np.std(self.flux)

		self.smoothFlux = self.flux
		#set the outliers as the flux below 
		self.smoothFlux[self.smoothFlux <= self.avgFlux-2*self.stdFlux] = 0
		self.mask = np.where(self.smoothFlux <= 0)
		self.wave  = np.delete(self.wave, list(self.mask))
		self.flux  = np.delete(self.flux, list(self.mask))
		self.noise = np.delete(self.noise, list(self.mask))
		self.sky   = np.delete(self.sky, list(self.mask))
		self.mask = self.mask[0]

	def plot(self, **kwargs):
		"""
		Plot the spectrum.
		"""
		xlim = kwargs.get('xrange', [self.wave[0], self.wave[-1]])
		ylim = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
		items  = kwargs.get('items', ['spec'])
		title  = kwargs.get('title')
		save   = kwargs.get('save', False)
		output = kwargs.get('output', str(self.name) + '.pdf')

		plt.figure(figsize=(16,4))
		#Plot masked spectrum
		if ('spectrum' in items) or ('spec' in items):
			plt.plot(self.wave, self.flux, color='k', alpha=.8, linewidth=1, label=self.name)

		#Plot spectrum noise
		if 'noise' in items:
			plt.plot(self.wave, self.noise, color='c', linewidth=1, alpha=.6)

		plt.legend(loc='upper right', fontsize=12)
        
        
		plt.xlim(xlim)
		plt.ylim(ylim)    
    
		minor_locator = AutoMinorLocator(5)
		#ax.xaxis.set_minor_locator(minor_locator)
		# plt.grid(which='minor') 
    
		plt.xlabel(r'$\lambda$ [$\mathring{A}$]', fontsize=18)
		plt.ylabel(r'$Flux$', fontsize=18)
		#plt.ylabel(r'$F_{\lambda}$ [$erg/s \cdot cm^{2}$]', fontsize=18)
		if title != None:
			plt.title(title, fontsize=20)
		plt.tight_layout()

		if save == True:
			plt.savefig(output)

		plt.show()
		plt.close()

	def writeto(self, save_to_path):
		"""
		Save as a new fits file.
		"""
		fullpath = self.path + '/' + self.name + '_' + str(self.order) + '_all.fits'
		hdulist = fits.open(fullpath, ignore_missing_end=True)
		hdulist.writeto(save_to_path)
		hdulist.close()


	def coadd(self, sp, method='pixel'):
		"""
		Coadd individual extractions, either in pixel space or
		wavelength space.
		usage: method='pixel' or 'wave'
		"""
		if sp is None:
			print("Please select another spectra.")
		if method == 'pixel':
			coadd = copy.deepcopy(sp)
			w1 = 1/self.oriNoise**2
			w2 = 1/sp.oriNoise**2
			#sp.wave = sp.wave
			coadd.oriFlux = (self.oriFlux*w1 + sp.oriFlux*w2)/(w1+w2)
			coadd.oriNoise = np.sqrt(1/(w1 + w2))
			#set up masking criteria
			coadd.avgFlux = np.mean(coadd.oriFlux)
			coadd.stdFlux = np.std(coadd.oriFlux)
			coadd.smoothFlux = coadd.oriFlux
			#set the outliers as the flux below 
			coadd.smoothFlux[coadd.smoothFlux <= coadd.avgFlux-2*coadd.stdFlux] = 0
			coadd.mask = np.where(coadd.smoothFlux <= 0)
			coadd.wave  = np.delete(coadd.oriWave, list(coadd.mask))
			coadd.flux  = np.delete(coadd.oriFlux, list(coadd.mask))
			coadd.noise = np.delete(coadd.oriNoise, list(coadd.mask))

		return coadd

	def updateWaveSol(self, tell_sp):
		"""
		Return a new wavelength solution given a wavelength calibrated telluric spectrum.
		"""
		wfit0 = tell_sp.header['WFIT0NEW']
		wfit1 = tell_sp.header['WFIT1NEW']
		wfit2 = tell_sp.header['WFIT2NEW']
		wfit3 = tell_sp.header['WFIT3NEW']
		wfit4 = tell_sp.header['WFIT4NEW']
		wfit5 = tell_sp.header['WFIT5NEW']
		c3    = tell_sp.header['c3']
		c4    = tell_sp.header['c4']
		self.wave = np.delete(nsp.waveSolution(np.arange(1024)+1,wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4\
			,order=self.order), list(self.mask))
		return self











