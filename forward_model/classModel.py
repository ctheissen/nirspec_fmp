#!/usr/bin/env python
#
# Jan. 31 2018
# @Dino Hsu
#
# The model class for the BTSettl models
#
#

import numpy as np
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def construct_model_names(teff, logg, feh, en, order, path=None):
    """
    Return the full name of the BT-Settl model.
    """
    if path is None:
        path = 'models/' + 'NIRSPEC-O' + str(order) + '-RAW/'
    else:
        path = path + 'NIRSPEC-O' + str(order) + '-RAW/'
    full_name = path + 'btsettl08_t'+ str(teff) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
    return full_name

class Model():
    """
    The Model class reads in the BT-SETTL models.
    
    Inputs:
    teff: given from 500 to 3,500 K
    logg: given in two decimal digits. Ex: logg = 3.50
    feh: given in two decimal digits. Ex. feh = 0.00
    en: alpha enhancement. given in two decimal digits. Ex. en = 0.00
    order: given from 29 to 80
    path: the path of the model
    
    Outputs:
    flux: the flux
    wave: the wavelength
    """
    def __init__(self, **kwargs):
        BASE = '/Users/dinohsu/Google Drive/ucsd/research/nirspec/models/'
        self.path  = kwargs.get('path')
        self.teff  = kwargs.get('teff')
        self.logg  = kwargs.get('logg')
        self.feh   = kwargs.get('feh')
        self.en    = kwargs.get('en')
        self.order = kwargs.get('order')

        if self.teff == None:
            self.teff = 3500
        if self.logg == None:
            self.logg = 5.00
        if self.feh  == None:
            self.feh  = 0.00
        if self.en   == None:
            self.en   = 0.00
        if self.order== None:
            self.order= 32
        
        full_name = construct_model_names(self.teff, self.logg, self.feh, self.en, self.order, self.path)
        model = ascii.read(full_name, format='no_header', fast_reader=False)
        self.wave  = model[0][:]*10000 #convert to Angstrom
        self.flux  = model[1][:]
        

    def plot(self, **kwargs):
        """
        Plot the model spectrum.
        """
        name = construct_model_names(self.teff, self.logg, self.feh, self.en, self.order, self.path)
        ylim = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
        title  = kwargs.get('title')
        save   = kwargs.get('save', False)
        output = kwargs.get('output', str(name) + '.pdf')
        
        plt.figure(figsize=(16,4))
        plt.plot(self.wave, self.flux, color='k', alpha=.8, linewidth=1, label=name)
        plt.legend(loc='upper right', fontsize=12)
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

