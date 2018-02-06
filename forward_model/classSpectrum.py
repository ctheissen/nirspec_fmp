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

import sys
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from astropy import units as u


class Spectrum():
	"""
	The spectrum class reading the NIRSPEC data with the parameters:
	@Dino Hsu

	input parameters:
	name: the part of the data filename before the order number
	order: the order of the spectra
	path: the path before the filename of the data

	output parameters:
	flux: the flux of the spectrum
	wave: the wavelength of the spectrum
	noise: the noise of the spectrum
	sky: the sky emission
	plot: plot the spectrum (with noise on/off option)

	"""
	def __init__(self, **kwargs):
		self.name = kwargs.get('name')
		self.order = kwargs.get('order')
		self.path = kwargs.get('path')

		if self.path == None:
			self.path = './'

		fullpath = self.path + '/' + self.name + '_' + self.order + '_all.fits'

		hdulist = fits.open(fullpath, ignore_missing_end=True)

		#The indices 0 to 3 correspond to wavelength, flux, noise, and sky
		self.header= hdulist[0].header
		self.wave  = hdulist[0].data
		self.flux  = hdulist[1].data
		self.noise = hdulist[2].data
		self.sky   = hdulist[3].data

		#set up masking criteria
		self.avgFlux = np.mean(self.flux)
		self.stdFlux = np.std(self.flux)

		self.smoothFlux = self.flux
		#set the outliers as the flux below 
		self.smoothFlux[self.smoothFlux <= self.avgFlux-2*self.stdFlux] = 0

		mask = np.where(self.smoothFlux <= 0)

		self.wave  = np.delete(self.wave, list(mask))
		self.flux  = np.delete(self.flux, list(mask))
		self.noise = np.delete(self.noise, list(mask))
		self.sky   = np.delete(self.sky, list(mask))

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









