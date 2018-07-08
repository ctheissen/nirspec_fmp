#!/usr/bin/env python
#
# Jan. 31 2018
# @Dino Hsu
#
# The model class for the BTSettl models
#
#
import warnings
warnings.filterwarnings("ignore")
import splat
import splat.model as spmd
import numpy as np
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def _constructModelName(teff, logg, feh, en, order, path=None):
    """
    Return the full name of the BT-Settl model.
    """
    if path is None:
        path = '/Users/dinohsu/projects/Models/models/btsettl08/' + 'NIRSPEC-O' + str(order) + '-RAW/'
    else:
        path = path + '/NIRSPEC-O' + str(order) + '-RAW/'
    full_name = path + 'btsettl08_t'+ str(teff) + '_g' + '{0:.2f}'.format(float(logg)) + '_z-' + '{0:.2f}'.format(float(feh)) + '_en' + '{0:.2f}'.format(float(en)) + '_NIRSPEC-O' + str(order) + '-RAW.txt'
    return full_name

class Model():
    """
    The Model class reads in the BT-SETTL models.    

    Parameters
    ----------
    1. Read in a BT-Settl model
    teff : int 
          The effective temperature, given from 500 to 3,500 K.
    logg : float
          The log(gravity), given in two decimal digits. 
          Ex: logg=3.50
    feh  : float
           The metalicity, given in two decimal digits. 
           Ex. feh=0.00
    en   : float
           alpha enhancement. given in two decimal digits. 
           Ex. en=0.00
    order: int
           The order of the model, given from 29 to 80
    path : str
           The path to the model

    2. Creat a model instance with given wavelengths and fluxes
    flux : astropy.table.column.Column
           The input flux.
    wave : astropy.table.column.Column
           The input wavelength.

    Returns
    -------
    flux : astropy.table.column.Column
           The flux retrieved from the model.
    wave : astropy.table.column.Column
           The wavelength retrieved from the model.

    Examples
    --------
    >>> import nirspec_pip as nsp
    >>> model = nsp.Model(teff=2300, logg=5.5, order=33, path='/path/to/models')
    >>> model.plot()
    """
    def __init__(self, **kwargs):
        self.path  = kwargs.get('path')
        self.order = kwargs.get('order')

        if self.order != None:
            self.teff  = kwargs.get('teff')
            self.logg  = kwargs.get('logg')
            self.feh   = kwargs.get('feh')
            self.en    = kwargs.get('en')
            if self.teff == None:
                self.teff = 2500
            if self.logg == None:
                self.logg = 5.00
            if self.feh  == None:
                self.feh  = 0.00
            if self.en   == None:
                self.en   = 0.00
            #print('Return a BT-Settl model of the order {0}, with Teff {1} logg {2}, z {3}, Alpha enhancement {4}.'\
            #    .format(self.order, self.teff, self.logg, self.feh, self.en))
        
            #full_name = _constructModelName(self.teff, self.logg, self.feh, self.en, self.order, self.path)
            #model = ascii.read(full_name, format='no_header', fast_reader=False)
            #self.wave  = model[0][:]*10000 #convert to Angstrom
            #self.flux  = model[1][:]
            
            #load the splat.interpolation BTSETTL model
            instrument = "NIRSPEC-O{}-RAW".format(self.order)
            sp = spmd.getModel(instrument=str(instrument),teff=self.teff,logg=self.logg,z=self.feh)
            self.wave = sp.wave.value*10000 #convert to Angstrom
            self.flux = sp.flux.value

        else:
            #print('Return a self-defined model.')
            self.wave   = kwargs.get('wave', [])
            self.flux   = kwargs.get('flux', [])
        

    def plot(self, **kwargs):
        """
        Plot the model spectrum.
        """
        if self.order != None:
            name = str(_constructModelName(self.teff, self.logg, self.feh, self.en, self.order, self.path))
            output = kwargs.get('output', str(name) + '.pdf')
            ylim = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
            title  = kwargs.get('title')
            save   = kwargs.get('save', False)
        
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
        else:
            output = kwargs.get('output'+ '.pdf')
            ylim = kwargs.get('yrange', [min(self.flux)-.2, max(self.flux)+.2])
            title  = kwargs.get('title')
            save   = kwargs.get('save', False)
        
            plt.figure(figsize=(16,4))
            plt.plot(self.wave, self.flux, color='k', alpha=.8, linewidth=1)
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


