import nirspec_pip as nsp
import apogee_tools.forward_model as apmdl
import splat as spt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import copy
import os
from astropy.io import fits
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import time

FULL_PATH  = os.path.realpath(__file__)
BASE = os.path.split(os.path.split(os.path.split(FULL_PATH)[0])[0])[0]

def waveSolution(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4, **kwargs):
    """
    Calculate the wavelength solution.
	The equation is from NSDRP_Software_Design p.35.

	Parameters
	----------
	wfit0-5: float
			 The fitting parameters in NSDRP_Software_Design p.35
	c3     : float
	         The coefficient of the third power of the polynomial
	c4     : float
			 The coefficient of the fourth power of the polynomial
	pixel  : int/array
			 The pixel number as the input of wavelength solution
    order  : int
             The order number as the input of wavelength

    Returns
    -------
    The wavelength solution: array-like


    """
    order = kwargs.get('order', 33)
    wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
    (wfit3 + wfit4*pixel + wfit5*pixel**2)/order \
    + c3*pixel**3 + c4*pixel**4
    return wave_sol

def corrRayleighJeans(wave):
	"""
	Calculate the Rayleigh-Jeans correction at a given wavelength. (SI unit)
	
	This is not applicable to NIRSPEC data because the telluric source is not
	a blackbody.
	"""
	T = 9500 #Kelvin
	c = 299792458 #meters
	k = 1.38*10**(-23) #Boltzmann constant
	return 2*c*k*T/wave**4

def getTelluric(wavelow, wavehigh, **kwargs):
	"""
	Get a telluric spectrum.

	Parameters
	----------
	wavelow:  int
			  lower bound of the wavelength range

	wavehigh: int
	          upper bound of the wavelength range

	airmass:  str
			  airmass of the telluric model
	alpha:    float
			  the power alpha parameter of the telluric model

	Returns
	-------
	telluric: model object
			  a telluric model with wavelength and flux


	Examples
	--------
	>>> import nirspec_pip as nsp
	>>> telluric = nsp.getTelluric(wavelow=22900,wavehigh=23250)

	"""
	
	airmass = kwargs.get('airmass', '1.5')
	alpha = kwargs.get('alpha', 1)
	am_key = {'1.0':'10', '1.5':'15'}
	tfile = 'pwv_R300k_airmass{}/LBL_A{}_s0_w005_R0300000_T.fits'.format(airmass, am_key[airmass])
	tellurics = fits.open(BASE + '/nirspec_pip/libraries/' + tfile)

	telluric = nsp.Model()
	telluric.wave = np.array(tellurics[1].data['lam'] * 10000)
	telluric.flux = np.array(tellurics[1].data['trans'])**(alpha)
	telluric_range = np.where


	# select the wavelength range
	criteria = (telluric.wave > wavelow) & (telluric.wave < wavehigh)
	telluric.wave = telluric.wave[criteria]
	telluric.flux = telluric.flux[criteria]
	return telluric

def _continuumFit(wave, m, b):
    return m*wave+b

def continuumTelluric(data, model=None):
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
		model = data
	popt, pcov = curve_fit(_continuumFit, data.wave, data.flux)
	const = np.mean(data.flux/_continuumFit(data.wave, *popt))-np.mean(model.flux)
	data.flux = data.flux/_continuumFit(data.wave, *popt) - const
	data.noise = data.noise/_continuumFit(data.wave, *popt)

	return data

def shiftTelluric(data, shift):
	"""
	Shift the data given a pixel shift.
	"""
	data2 = copy.deepcopy(data)
	# calculate the wavelength solution with a shifted pixel value
	try:
		wfit0 = data2.header['WFIT0NEW']
		wfit1 = data2.header['WFIT1NEW']
		wfit2 = data2.header['WFIT2NEW']
		wfit3 = data2.header['WFIT3NEW']
		wfit4 = data2.header['WFIT4NEW']
		wfit5 = data2.header['WFIT5NEW']
		c3    = data2.header['C3']
		c4    = data2.header['C4']

	except KeyError:
		# This occurs for the first iteration
		wfit0 = data2.header['WFIT0']
		wfit1 = data2.header['WFIT1']
		wfit2 = data2.header['WFIT2']
		wfit3 = data2.header['WFIT3']
		wfit4 = data2.header['WFIT4']
		wfit5 = data2.header['WFIT5']
		c3 = 0
		c4 = 0

	pixel = np.delete(np.arange(1024),data2.mask)+1+shift
	data2.wave = nsp.waveSolution(pixel,wfit0,wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4)
	return data2

def xcorrTelluric(data, model, shift, start_pixel,width):
	"""
	(Deprecated)
	Calculate the cross-correlation of the telluric model and data.
	Using integer pixel shift as steps.
	"""
	# the width of the range of pixels to calculate the xcorrelation.
	data = nsp.shiftTelluric(data, shift)
	if shift >= 0:
		d = data.flux[start_pixel:start_pixel+width]
		m = model.flux[start_pixel+shift:start_pixel+width+shift]
	else:
		if start_pixel+shift > 0:
			d = data.flux[start_pixel:start_pixel+width]
			m = model.flux[start_pixel+shift:start_pixel+width+shift]
		else:
			d = data.flux[start_pixel-shift:start_pixel+width-shift]
			m = model.flux[start_pixel:start_pixel+width]
	xcorr = np.inner(d, m)/(np.average(d)*np.average(m))
	central_pixel = width/2 + shift
	#return central_pixel, xcorr
	return xcorr


def xcorrTelluric2(data, model, shift, start_pixel,width):
	"""
	Calculate the cross-correlation of the telluric model and data.
	Using wavelength shift as steps.
	
	Parameters
	----------
	data: telluric data
	model: telluric model
	shift: shift of the telluric MODEL in wavelengths(Angstrom)
	start_pixel: starting pixel number to compute the xcorr
	width: window width to compute the xcorr
	"""
	# shift the wavelength of the telluric model
	# minus sign means we want to know the shift of the data
	model2 = copy.deepcopy(model)
	model2.wave = model.wave - shift

	# select a range of wavelength to compute x-correlation value
	model_low = data.wave[start_pixel]-20
	model_high = data.wave[start_pixel+width]+20
	condition = np.where(model2.wave<model_high)
	model2.wave = model2.wave[condition]
	model2.flux = model2.flux[condition]
	condition = np.where(model2.wave>model_low)
	model2.wave = model2.wave[condition]
	model2.flux = model2.flux[condition]
	# resampling the telluric model
	model2.flux = np.array(spt.integralResample(xh=model2.wave, yh=model2.flux, xl=data.wave[start_pixel:start_pixel+width]))
	model2.wave = data.wave[start_pixel:start_pixel+width]

	# LSF of the intrument
	#vbroad = (299792458/1000)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	#model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, flux=model2.flux, vbroad=vbroad, rotate=False, gaussian=True)

	d = data.flux[start_pixel:start_pixel+width]
	#m = model2.flux[start_pixel:start_pixel+width]
	m = model2.flux
	xcorr = np.inner(d, m)/(np.average(d)*np.average(m))
	return xcorr

def pixelWaveShift2(data, model, start_pixel,window_width=40,delta_wave_range=10,model2=None,test=False,testname=None,gaussian=True,counter=None,**kwargs):
	"""
	Find the max cross-correlation and compute the pixel to wavelength shift.
	
	Parameters
	----------
	model: MUST BEFORE resampling and LSF broadening
	model2: model AFTER resampling and LSF broadening (to increase computation speed)


	Returns
	-------
	best wavelength shift: float
					  the wavelength correction after xcorr

	"""
	# the step for wavelength shift, default=0.1 Angstrom
	step = kwargs.get('step',0.1)
	xcorr_list = []
	#model2 = copy.deepcopy(model)
	if model2 is None:
		model2 = model
	# select the range of the pixel shift to compute the max xcorr
	for i in np.arange(-delta_wave_range,delta_wave_range,step):
		# propagate the best pixel shift
		if len(data.bestshift) > 1:
			j = data.bestshift[counter] + i
		else:
			j = i
		xcorr = nsp.xcorrTelluric2(data,model,j,start_pixel,window_width)
		#print("delta wavelength shift:{}, xcorr value:{}".format(i,xcorr))
		xcorr_list.append(xcorr)
	#print("xcorr list:",xcorr_list)
	best_shift = np.arange(-delta_wave_range,delta_wave_range,step)[np.argmax(xcorr_list)]
	central_pixel = start_pixel + window_width/2
	#new_wave = nsp.shiftTelluric(data, best_shift).wave[central_pixel]
	#diff_wave = new_wave - data.wave[central_pixel]

	# model resample and LSF broadening
	#model2.flux = np.array(spt.integralResample(xh=model2.wave, yh=model2.flux, xl=data.wave))
	#model2.wave = data.wave

	# LSF of the intrument
	#vbroad = (299792458/1000)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	#model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, flux=model2.flux, vbroad=vbroad, rotate=False, gaussian=True)


	if test is True:
		# parameters setup for plotting
		linewidth=1.0

		pixel = np.delete(np.arange(1024),data.mask)+1

		fig = plt.figure(figsize=(12,8))
		gs1 = gridspec.GridSpec(4, 4)
		ax1 = plt.subplot(gs1[0:2, :])
		ax2 = plt.subplot(gs1[2:, 0:2])
		ax3 = plt.subplot(gs1[2:, 2:])
		
		ax1.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=linewidth)
		ax1.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',alpha=0.5,linewidth=linewidth)
		ax1.set_xlabel('Wavelength(Angstrom)')
		ax1.set_ylabel('Transmission')
		ax1.set_ylim(0,1.1)
		ax1.set_xlim(data.wave[0],data.wave[-1])
		ax1.set_title('Telluric Spectra Region for Cross-Correlation')
		ax1.axvline(x=data.wave[start_pixel],linestyle='--',color='blue',linewidth=linewidth)
		ax1.axvline(x=data.wave[start_pixel+window_width],linestyle='--',color='blue',linewidth=linewidth)
		ax1.get_xaxis().get_major_formatter().set_scientific(False)
		ax1.legend()
		#ax1t = ax1.twiny()
		#ax1t.plot(pixel,data.flux,color='w',alpha=0)

		ax2.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=linewidth)
		ax2.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',alpha=0.5,linewidth=linewidth)
		ax2.set_ylim(0,1.1)
		ax2.set_xlim(data.wave[start_pixel]-0.1,data.wave[start_pixel+window_width]+0.1)
		ax2.axvline(x=data.wave[start_pixel],linestyle='--',color='blue',linewidth=linewidth)
		ax2.axvline(x=data.wave[start_pixel+window_width],linestyle='--',color='blue',linewidth=linewidth)
		ax2.get_xaxis().get_major_formatter().set_scientific(False)
		ax2.legend()
		#ax2t = ax2.twiny()
		#ax2t.set_xlabel('pixel')
		
		#labels = np.arange(pixel[start_pixel]-1,pixel[start_pixel+window_width]-1,1).tolist()
		#for i in np.arange(pixel[start_pixel],pixel[start_pixel+window_width+1]):
		#	labels.append(pixel[i])
		
		#ax2t.plot(pixel[start_pixel:start_pixel+window_width],data.flux[start_pixel:start_pixel+window_width],color='w',alpha=0)
		#ax2.set_xlabel('Wavelength(Angstrom)')
		#ax2.set_ylabel('Transmission')
		#ax2.legend(loc=9, bbox_to_anchor=(0.5, -0.3))

		#ax2tmajor_ticks = np.arange(data.wave[start_pixel],data.wave[start_pixel+window_width],5).tolist()
		#ax2t.set_xticklabels(labels)
		#ax2t.set_xticklabels([start_pixel,start_pixel+window_width])
		#ax2tminor_ticks = np.arange(start_pixel,(start_pixel+window_width+1),1)
		#ax2t.set_xticks(ax2tminor_ticks, minor=True)
		
		
		
		if testname is None:
			testname = 'test1'
		if gaussian is True:
			# pass if the max shift is equal to the delta wavelength shift
			# in this case the gaussian fit is meaningless
			if np.absolute(np.argmax(xcorr_list)) is delta_wave_range/step:
				pass
			else:
				x = np.arange(-delta_wave_range,delta_wave_range,step)
				y = xcorr_list
				# interpolate the xcorr and find the local minimum near the 
				# best shift
				xcorr_int_y = UnivariateSpline(x, xcorr_list, k=4, s=0)
				xcorr_int_x = np.arange(x[0],x[-1],1000)

				# select the range of elements for the gaussian fitting
				# percent for the guassian fitting
				xcorr_fit_percent = 0.8
				condition = y>np.min(y)+(np.max(y)-np.min(y))*xcorr_fit_percent
				x_select = np.select([condition],[x])[condition]
				y_select = np.select([condition],[y])[condition]
				print("start pixel: {}".format(start_pixel))
				#print("x_select:",x_select)
				#print("y_select:",y_select)
				# select the values only around central central maximum
				roots = xcorr_int_y.derivative().roots()
				#if len(roots) is 0:
				#	pass
				#elif len(roots) is 1:
				#	if roots[0] < best_shift:
				#		condition1 = x_select>roots[0]
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#	elif roots[0] > best_shift:
				#		condition1 = x_select<roots[0]
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#else:
				#	root_index = np.searchsorted(roots, best_shift, side='left')
				#	if root_index is 0:
				#		if roots[0] > best_shift:
				#			condition1 = x_select<roots[0]
				#			x_select = np.select([condition1],[x_select])[condition1]
				#			y_select = np.select([condition1],[y_select])[condition1]
				#		elif roots[0] < best_shift:
				#			condition1 = x_select>roots[0]
				#			x_select = np.select([condition1],[x_select])[condition1]
				#			y_select = np.select([condition1],[y_select])[condition1]
				#	else:
				#		root_right = roots[root_index]
				#		root_left  = roots[root_index-1]
				#		condition1 = x_select<root_right
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#		condition2 = x_select>root_left
				#		x_select = np.select([condition2],[x_select])[condition2]
				#		y_select = np.select([condition2],[y_select])[condition2]
				#	print("len(y_select):",len(y_select))

				diff = np.diff(x_select)
				b = np.where(diff>1)[0]
				if len(b) is 0:
					pass
				elif len(b) is 1:
					k = b[0]
					l = x_select[k]
					if l < best_shift:
						condition1 = x_select>l
						x_select = np.select([condition1],[x_select])[condition1]
						y_select = np.select([condition1],[y_select])[condition1]
					elif x_select[k] > best_shift:
						condition1 = x_select<l
						x_select = np.select([condition1],[x_select])[condition1]
						y_select = np.select([condition1],[y_select])[condition1]
				else:
					for k in b:
						l_list = []
						l_list.append(x_select[k])
					for l in l_list:	
						if l < best_shift:
							condition1 = x_select>l
							x_select = np.select([condition1],[x_select])[condition1]
							y_select = np.select([condition1],[y_select])[condition1]
						elif l > best_shift:
							condition1 = x_select<l
							x_select = np.select([condition1],[x_select])[condition1]
							y_select = np.select([condition1],[y_select])[condition1]
				#print("x_select after:",x_select)
				#print("y_select after:",y_select)

				n = len(xcorr_list)                 #the number of data
				mean0 = best_shift             #note this correction
				sigma0 = sum((x-mean0)**2)/n        #note this correction

				# initial parameters of selected xcorr for the gaussian fit
				n2 = len(y_select)
				mean2 = best_shift
				sigma2 = sum((x_select-mean2)**2)/n2

				def gaus(x,a,x0,sigma):
					return a*np.e**(-(x-x0)**2/(2*sigma**2))

				try:
					popt,pcov = curve_fit(gaus,x,y,p0=[np.max(xcorr_list),mean0,sigma0])
					if np.absolute(popt[1]) >= delta_wave_range:
						popt[1] = best_shift
					#ax3.plot(x, gaus(x,popt[0],popt[1],popt[2]),'c-',label='gaussian fit')
					#ax3.plot([popt[1],popt[1]],[float(np.min(xcorr_list)),float(np.max(xcorr_list))],'c--',label="gaussian fitted pixel:{}".format(popt[1]))
					ax3.plot(x_select,y_select,color='fuchsia',label="{} persent range".format(xcorr_fit_percent*100))
					#popt2,pcov2 = curve_fit(gaus,x_select,y_select,p0=[np.max(y_select),mean2,sigma2])
					popt2,pcov2 = curve_fit(gaus,x_select,y_select)
					if np.absolute(popt2[1]) >= delta_wave_range:
						popt2[1] = best_shift
					ax3.plot(x_select,gaus(x_select,popt2[0],popt2[1],popt2[2]),color='olive',label='gaussian fit for selected parameters, shift:{}'.format(popt2[1]),alpha=0.5)
					ax3.axvline(x=popt2[1],linewidth=0.5,linestyle='--',color='olive')
					for root in xcorr_int_y.derivative().roots():
						ax3.axvline(x=root, linewidth=0.3, color='purple')
					# replace the fitted gaussian value 
					replace_shift_criteria = 0.5 #the threshold for replacing the pixel shift with gaussian fit
					if np.absolute(best_shift-popt2[1]) < replace_shift_criteria:
						best_shift = popt2[1]
					else:
						try:
							condition3 = x_select < np.argmax(x_select)+3
							x_select2 = np.select([condition3],[x_select])[condition3]
							y_select2 = np.select([condition3],[y_select])[condition3]
							condition4 = x_select > np.argmax(x_select2)-3
							x_select2 = np.select([condition4],[x_select2])[condition4]
							y_select2 = np.select([condition4],[y_select2])[condition4]
						except ValueError:
							try:
								condition3 = x_select < np.argmax(x_select)+2
								x_select2 = np.select([condition3],[x_select])[condition3]
								y_select2 = np.select([condition3],[y_select])[condition3]
								condition4 = x_select > np.argmax(x_select2)-2
								x_select2 = np.select([condition4],[x_select2])[condition4]
								y_select2 = np.select([condition4],[y_select2])[condition4]
							except ValueError:
								pass
						try:
							popt3,pcov3 = curve_fit(gaus,x_select2,y_select2)
							if np.absolute(best_shift-popt3[1]) < replace_shift_criteria:
								best_shift = popt3[1]
							ax3.plot(x_select2,gaus(x_select2,popt3[0],popt3[1],popt3[2]),color='salmon',label='gaussian fit for selected parameters, gaussian fitted pixel:{}'.format(popt3[1]),alpha=0.5)
							ax3.axvline(x=popt3[1],linewidth=0.5,linestyle='--',color='salmon')
						except RuntimeError:
							pass
						except TypeError:
							pass
				except RuntimeError:
					pass
				except TypeError:
					pass
		

		ax3.plot(np.arange(-delta_wave_range,delta_wave_range,step),xcorr_list,color='black',label='cross correlation',alpha=0.5)
		ax3.plot([best_shift,best_shift],[float(np.min(xcorr_list)),float(np.max(xcorr_list))],'k:',label="best wavelength shift:{}".format(best_shift))
		ax3major_ticks = np.arange(-(delta_wave_range),(delta_wave_range+1),2)
		ax3minor_ticks = np.arange(-(delta_wave_range),(delta_wave_range+1),0.1)
		ax3.set_xticks(ax3major_ticks)
		ax3.set_xticks(ax3minor_ticks, minor=True)
		ax3.set_title("Cross-Correlation Plot, pixels start at {} with width {}".format(start_pixel,window_width))
		ax3.set_xlabel('Wavelength shift(Angstrom)')
		ax3.set_ylabel('Cross correlation')
		ax3.set_xlim(-(delta_wave_range),(delta_wave_range))
		ax3.set_ylim(np.min(xcorr_list),np.max(xcorr_list))
		ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.5))
		plt.tight_layout(h_pad=3.3)
		plt.savefig('{}_{}.png'.format(testname,start_pixel), bbox_inches='tight',dpi=128)
		plt.close()

	return best_shift

def wavelengthSolutionFit2(data,model,**kwargs):
	"""
	Using the least square fit to calculate the new wavelength solution.
	
	Parameters
	----------
	data:  spectrum object
	       The input telluric data to obtain the initial wavelength solution
	       parameters

	model: model object
		   The telluric model to calculate the delta wavelength

	Optional Parameters
	-------------------
	spec_range  : int
				  range of the spectrum for fitting wavelength solution 
				  in the unit of pixels

	window_width: int
				  width of each xcorr window

	delta_wave_range: float
				  xcorr window range BY SHIFTING THE WAVELENGTH to calculate the max xcorr

	step_size   : int
				  step size of moving to the next xcorr window

	niter		: int
				  number of interations for fitting the wavelength solution

	Returns
	-------
	wavelength sol: numpy array
				    a new wavelength solution


	"""
	# set up the initial parameters
	spec_range = kwargs.get('spec_range',900)
	width = kwargs.get('window_width',40)
	delta_wave_range = kwargs.get('delta_wave_range',10)
	step_size = kwargs.get('step_size',5)
	step = kwargs.get('step',0.1)
	niter = kwargs.get('niter',1)
	test  = kwargs.get('test',False)

	width_range = np.arange(0,spec_range,step_size)
	width_range_center = width_range + width/2
	pixel = np.delete(np.arange(1024),data.mask)+1

	data2 = copy.deepcopy(data)
	model2 = copy.deepcopy(model)
	# model resample and LSF broadening
	model2.flux = np.array(spt.integralResample(xh=model2.wave, yh=model2.flux, xl=data.wave))
	model2.wave = data.wave

	# LSF of the intrument
	vbroad = (299792458/1000)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, flux=model2.flux, vbroad=vbroad, rotate=False, gaussian=True)

	# fitting the new wavelength solution
	if test is True:
		print("initial WFIT:",data2.header['WFIT0'],data2.header['WFIT1'],data2.header['WFIT2'],data2.header['WFIT3'],data2.header['WFIT4'],data2.header['WFIT5'])

	for i in range(niter):
	# getting the parameters of initial wavelength solution 
		k = i + 1
		try:
			wfit0 = data2.header['WFIT0NEW']
			wfit1 = data2.header['WFIT1NEW']
			wfit2 = data2.header['WFIT2NEW']
			wfit3 = data2.header['WFIT3NEW']
			wfit4 = data2.header['WFIT4NEW']
			wfit5 = data2.header['WFIT5NEW']
			c3    = data2.header['C3']
			c4    = data2.header['C4']
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} new wave parameters, time {} s".format(k,time.time()))


		except KeyError:
			wfit0 = data2.header['WFIT0']
			wfit1 = data2.header['WFIT1']
			wfit2 = data2.header['WFIT2']
			wfit3 = data2.header['WFIT3']
			wfit4 = data2.header['WFIT4']
			wfit5 = data2.header['WFIT5']
			c3 = 0
			c4 = 0
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} old wave parameters, time {} s".format(k,time.time()))

		# calcutate the delta wavelentgh
		best_shift_list  = []
		for counter, j in enumerate(width_range):
			testname = "loop{}".format(k)
			if i is 0:
				best_shift = nsp.pixelWaveShift2(data2,model,j,width,delta_wave_range,model2,test=test,testname=testname,counter=counter,step=step)
			elif i is 1:
				# reduce the delta_wave_range by 5
				best_shift = nsp.pixelWaveShift2(data2,model,j,width,5,model2,test=test,testname=testname,counter=counter,step=step)
			else:
				# reduce the delta_wave_range by 8
				best_shift = nsp.pixelWaveShift2(data2,model,j,width,2,model2,test=test,testname=testname,counter=counter,step=step)
			#else:
			#	best_shift = nsp.pixelWaveShift2(data2,model,j,width,0.2,model2,test=test,testname=testname,counter=counter,step=step)
			#else:
			#	best_shift = nsp.pixelWaveShift2(data2,model,j,width,0.15,model2,test=test,testname=testname,counter=counter,step=0.001)
			best_shift_list.append(best_shift)
		# fit a new wavelength solution
		popt, pcov = curve_fit(nsp.waveSolution, width_range_center, best_shift_list,p0)

		# outlier rejection
		best_shift_array = np.asarray(best_shift_list)
		original_fit = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])
		m = 3 # number of sigma for outlier rejection
		if k is 1:
			fit_sigma = 1.
			#fit_sigma = np.std(original_fit - best_shift_array)
		else:
			fit_sigma = data2.header['FITSTD']
		#else:
		#	fit_sigma = 0.1
		
		# exclude the edge pixels in the fitting
		#width_range_center2 = width_range_center[5:-5]
		#best_shift_array2 = best_shift_array[5:-5]
		#width_range_center2 = width_range_center2[np.where(abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		#best_shift_array2 = best_shift_array2[np.where(abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		width_range_center2 = width_range_center[np.where(abs(original_fit - best_shift_array) < m*fit_sigma)]
		best_shift_array2 = best_shift_array[np.where(abs(original_fit - best_shift_array) < m*fit_sigma)]

		# drop the last 10 pixels for the fitting
		popt2, pcov2 = curve_fit(nsp.waveSolution,width_range_center2,best_shift_array2,p0)

		# update the parameters
		wfit0 = wfit0+popt2[0]
		wfit1 = wfit1+popt2[1]
		wfit2 = wfit2+popt2[2]
		wfit3 = wfit3+popt2[3]
		wfit4 = wfit4+popt2[4]
		wfit5 = wfit5+popt2[5]
		c3    = c3+popt2[6]
		c4    = c4+popt2[7]
		p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
		
		# update the fits header keywords WFIT0-5, c3, c4
		data2.header['COMMENT']  = 'Below are the keywords added by NIRSPEC_PIP...'
		data2.header['WFIT0NEW'] = wfit0
		data2.header['WFIT1NEW'] = wfit1
		data2.header['WFIT2NEW'] = wfit2
		data2.header['WFIT3NEW'] = wfit3
		data2.header['WFIT4NEW'] = wfit4
		data2.header['WFIT5NEW'] = wfit5
		data2.header['c3']       = c3
		data2.header['c4']       = c4
		data2.bestshift          = data2.bestshift + best_shift_list
		data2.header['FITSTD']   = np.std(nsp.waveSolution(width_range_center2,popt2[0]\
			,popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7]) - best_shift_array2)

		# plot for analysis
		if test is True:
			residual1 = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])-best_shift_array
			residual2 = nsp.waveSolution(width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7])-best_shift_array2
			grid = plt.GridSpec(4, 4, wspace=0.4, hspace=0.3)
			#print("The new best-shift list:",best_shift_list)
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')
			plt.subplot(grid[0:3,0:])
			plt.plot(width_range_center,best_shift_array,'k.',label="delta wavelength")
			plt.plot(width_range_center2,best_shift_array2,'b.',label="delta wavelength with ourlier rejection")
			plt.plot(width_range_center,nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]),'g.',alpha=0.5)
			plt.plot(width_range_center2,nsp.waveSolution(width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7]),'r.',alpha=0.5)
			plt.title(r"Pixel-" "$\displaystyle\Delta \lambda$" "({} Iteration)".format(k))
			plt.ylabel(r"$\displaystyle\Delta \lambda$")
			
			# plot the residual
			plt.subplot(grid[3:,0:])
			plt.plot(width_range_center,residual1,'g.',label="fitted wavelength function, STD={} Angstrom".format(np.std(residual1)),alpha=0.5)
			plt.plot(width_range_center2, residual2,'r.',label="fitted wavelength function with outlier rejection, STD={} Angstrom".format(np.std(residual2)),alpha=0.5)
			plt.ylim(-1,1)
			plt.ylabel(r"$residual \displaystyle\Delta \lambda$")
			plt.xlabel(r'Pixel')
			plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5))
			plt.savefig("pixel_to_delta_wavelength_loop_{}.png".format(k),bbox_inches='tight')
			plt.close()
			#print("fitted popt loop {}:".format(k),popt)
		
	new_wave_sol = nsp.waveSolution(pixel,wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4)
	data2.wave = new_wave_sol

	##test
	#linewidth=1.0
	#residual_telluric_data = nsp.residual(data,model2)
	#telluric_new = copy.deepcopy(data)
	#telluric_new.wave = new_wave_sol
	#telluric_new.flux = data.flux
	#residual_telluric_wavesol = nsp.residual(telluric_new,model2)

	#fig = plt.figure(figsize=(16,6))
	#ax1 = fig.add_subplot(111)
	#ax1.plot(data.wave, data.flux, color='black',linestyle='-', label="data, STD={}".format(\
	#np.nanstd(residual_telluric_data.flux)),alpha=0.5,linewidth=linewidth)
	#ax1.plot(model2.wave, model2.flux, color='red',linestyle='-',label='model',alpha=0.5,linewidth=linewidth)
	#ax1.plot(new_wave_sol,data.flux,color='blue',linestyle='-',label="new wavelength solution, STD={}".format(\
	#	np.nanstd(residual_telluric_wavesol.flux)),alpha=0.5,linewidth=linewidth)
	#ax1.fill_between(residual_telluric_data.wave,-residual_telluric_data.flux,residual_telluric_data.flux\
	#	,color='black',alpha=0.5)
	#ax1.plot(residual_telluric_wavesol.wave,residual_telluric_wavesol.flux,color='blue',linestyle='-',alpha=0.5,linewidth=linewidth)
	#ax1.axhline(y=0,color='grey',linestyle=':',alpha=0.5)
	#ax1.set_title('Telluric model and wavelength solution comparison',y=1.15)
	#ax1.set_xlabel('wavelength(Angstrom)')
	#ax1.set_ylabel('normalized flux')
	#ax2 = ax1.twiny()
	#ax2.plot(pixel,data.flux,color='w',alpha=0)
	#ax2.set_xlabel('pixel')
	#ax1.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
	#fig.savefig('telluric_new_wavelength_solution_comparison.png',dpi=2048, bbox_inches='tight')
	#plt.show()
	#plt.close()

	return new_wave_sol, p0

def pixelWaveShift(data, model, start_pixel,window_width=40,delta_pixel_range=15,test=False,testname=None,gaussian=True,counter=None):
	"""
	(Deprecated)
	Find the max cross-correlation and compute the pixel to wavelength shift.
	
	Returns
	-------
	delta wavelength: float
					  the wavelength correction after xcorr

	"""
	xcorr_list = []
	# select the range of the pixel shift to compute the max xcorr
	for i in range(-delta_pixel_range,delta_pixel_range):
		# propagate the best pixel shift
		if len(data.bestshift) > 1:
			j = data.bestshift[counter] + i
		else:
			j = i
		try:
			xcorr = nsp.xcorrTelluric(data,model,j,start_pixel,window_width)
			print("delta pixel:{}, xcorr value:{}".format(i,xcorr))
			xcorr_list.append(xcorr)
		except ValueError:
			pass
	print("xcorr list:",xcorr_list)
	best_shift = np.arange(-delta_pixel_range,delta_pixel_range)[np.argmax(xcorr_list)]
	central_pixel = start_pixel + window_width/2
	new_wave = nsp.shiftTelluric(data, best_shift).wave[central_pixel]
	diff_wave = new_wave - data.wave[central_pixel]

	if test is True:
		# parameters setup for plotting
		linewidth=1.0


		pixel = np.delete(np.arange(1024),data.mask)+1

		fig = plt.figure(figsize=(12,8))
		gs1 = gridspec.GridSpec(4, 4)
		ax1 = plt.subplot(gs1[0:2, :])
		ax2 = plt.subplot(gs1[2:, 0:2])
		ax3 = plt.subplot(gs1[2:, 2:])
		
		ax1.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=linewidth)
		ax1.plot(model.wave, model.flux, 'r-' ,label='telluric model',alpha=0.5,linewidth=linewidth)
		ax1.set_xlabel('Wavelength(Angstrom)')
		ax1.set_ylabel('Transmission')
		#ax1.set_title('Telluric Spectra Region for Cross-Correlation')
		ax1.axvline(x=data.wave[start_pixel],linestyle='--',color='blue',linewidth=linewidth)
		ax1.axvline(x=data.wave[start_pixel+window_width],linestyle='--',color='blue',linewidth=linewidth)
		ax1t = ax1.twiny()
		ax1t.plot(pixel,data.flux,color='w',alpha=0)

		ax2.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=linewidth)
		ax2.plot(model.wave, model.flux, 'r-' ,label='telluric model',alpha=0.5,linewidth=linewidth)
		ax2.set_xlim(data.wave[start_pixel]-0.1,data.wave[start_pixel+window_width]+0.1)
		ax2.axvline(x=data.wave[start_pixel],linestyle='--',color='blue',linewidth=linewidth)
		ax2.axvline(x=data.wave[start_pixel+window_width],linestyle='--',color='blue',linewidth=linewidth)
		ax2t = ax2.twiny()
		ax2t.set_xlabel('pixel')
		ax2t.plot(pixel[start_pixel:start_pixel+window_width],data.flux[start_pixel:start_pixel+window_width],color='w',alpha=0)
		#ax2tmajor_ticks = [data.wave[start_pixel],data.wave[start_pixel+window_width]]
		#ax2t.set_xticks(ax2tmajor_ticks)
		#ax2t.set_xticklabels([start_pixel,start_pixel+window_width])
		#ax2tminor_ticks = np.arange(start_pixel,(start_pixel+window_width+1),1)
		#ax2t.set_xticks(ax2tminor_ticks, minor=True)
		ax2.set_xlabel('Wavelength(Angstrom)')
		ax2.set_ylabel('Transmission')
		ax2.legend(loc=9, bbox_to_anchor=(0.5, -0.3))
		
		
		if testname is None:
			testname = 'test1'
		if gaussian is True:
			# pass if the max shift is equal to the delta wavelength shift
			# in this case the gaussian fit is meaningless
			if np.absolute(np.argmax(xcorr_list)) is delta_pixel_range:
				pass
			else:
				x = np.arange(-delta_pixel_range,delta_pixel_range)
				y = xcorr_list
				# interpolate the xcorr and find the local minimum near the 
				# best shift
				xcorr_int_y = UnivariateSpline(x, xcorr_list, k=4, s=0)
				xcorr_int_x = np.arange(x[0],x[-1],1000)

				# select the range of elements for the gaussian fitting
				# percent for the guassian fitting
				xcorr_fit_percent = 0.5
				condition = y>(np.max(y)+np.min(y))*xcorr_fit_percent
				x_select = np.select([condition],[x])[condition]
				y_select = np.select([condition],[y])[condition]
				print("start pixel:",start_pixel)
				print("x_select:",x_select)
				print("y_select:",y_select)
				# select the values only around central central maximum
				roots = xcorr_int_y.derivative().roots()
				#if len(roots) is 0:
				#	pass
				#elif len(roots) is 1:
				#	if roots[0] < best_shift:
				#		condition1 = x_select>roots[0]
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#	elif roots[0] > best_shift:
				#		condition1 = x_select<roots[0]
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#else:
				#	root_index = np.searchsorted(roots, best_shift, side='left')
				#	if root_index is 0:
				#		if roots[0] > best_shift:
				#			condition1 = x_select<roots[0]
				#			x_select = np.select([condition1],[x_select])[condition1]
				#			y_select = np.select([condition1],[y_select])[condition1]
				#		elif roots[0] < best_shift:
				#			condition1 = x_select>roots[0]
				#			x_select = np.select([condition1],[x_select])[condition1]
				#			y_select = np.select([condition1],[y_select])[condition1]
				#	else:
				#		root_right = roots[root_index]
				#		root_left  = roots[root_index-1]
				#		condition1 = x_select<root_right
				#		x_select = np.select([condition1],[x_select])[condition1]
				#		y_select = np.select([condition1],[y_select])[condition1]
				#		condition2 = x_select>root_left
				#		x_select = np.select([condition2],[x_select])[condition2]
				#		y_select = np.select([condition2],[y_select])[condition2]
				#	print("len(y_select):",len(y_select))

				diff = np.diff(x_select)
				b = np.where(diff>1)[0]
				if len(b) is 0:
					pass
				elif len(b) is 1:
					k = b[0]
					l = x_select[k]
					if l < best_shift:
						condition1 = x_select>l
						x_select = np.select([condition1],[x_select])[condition1]
						y_select = np.select([condition1],[y_select])[condition1]
					elif x_select[k] > best_shift:
						condition1 = x_select<l
						x_select = np.select([condition1],[x_select])[condition1]
						y_select = np.select([condition1],[y_select])[condition1]
				else:
					for k in b:
						l_list = []
						l_list.append(x_select[k])
					for l in l_list:	
						if l < best_shift:
							condition1 = x_select>l
							x_select = np.select([condition1],[x_select])[condition1]
							y_select = np.select([condition1],[y_select])[condition1]
						elif l > best_shift:
							condition1 = x_select<l
							x_select = np.select([condition1],[x_select])[condition1]
							y_select = np.select([condition1],[y_select])[condition1]
				print("x_select after:",x_select)
				print("y_select after:",y_select)

				n = len(xcorr_list)                 #the number of data
				mean0 = best_shift             #note this correction
				sigma0 = sum((x-mean0)**2)/n        #note this correction

				# initial parameters of selected xcorr for the gaussian fit
				n2 = len(y_select)
				mean2 = best_shift
				sigma2 = sum((x_select-mean2)**2)/n2

				def gaus(x,a,x0,sigma):
					return a*np.e**(-(x-x0)**2/(2*sigma**2))

				try:
					popt,pcov = curve_fit(gaus,x,y,p0=[np.max(xcorr_list),mean0,sigma0])
					if np.absolute(popt[1]) >= delta_pixel_range:
						popt[1] = best_shift
					#ax3.plot(x, gaus(x,popt[0],popt[1],popt[2]),'c-',label='gaussian fit')
					#ax3.plot([popt[1],popt[1]],[float(np.min(xcorr_list)),float(np.max(xcorr_list))],'c--',label="gaussian fitted pixel:{}".format(popt[1]))
					ax3.plot(x_select,y_select,color='fuchsia',label='50 persent range')
					#popt2,pcov2 = curve_fit(gaus,x_select,y_select,p0=[np.max(y_select),mean2,sigma2])
					popt2,pcov2 = curve_fit(gaus,x_select,y_select)
					if np.absolute(popt2[1]) >= delta_pixel_range:
						popt2[1] = best_shift
					ax3.plot(x_select,gaus(x_select,popt2[0],popt2[1],popt2[2]),color='olive',label='gaussian fit for selected parameters, gaussian fitted pixel:{}'.format(popt2[1]),alpha=0.5)
					ax3.axvline(x=popt2[1],linewidth=0.5,linestyle='--',color='olive')
					for root in xcorr_int_y.derivative().roots():
						ax3.axvline(x=root, linewidth=0.3, color='purple')
					# replace the fitted gaussian value 
					replace_shift_criteria = 0.5 #the threshold for replacing the pixel shift with gaussian fit
					if np.absolute(best_shift-popt2[1]) < replace_shift_criteria:
						best_shift = popt2[1]
					else:
						try:
							condition3 = x_select < np.argmax(x_select)+3
							x_select2 = np.select([condition3],[x_select])[condition3]
							y_select2 = np.select([condition3],[y_select])[condition3]
							condition4 = x_select > np.argmax(x_select2)-3
							x_select2 = np.select([condition4],[x_select2])[condition4]
							y_select2 = np.select([condition4],[y_select2])[condition4]
						except ValueError:
							try:
								condition3 = x_select < np.argmax(x_select)+2
								x_select2 = np.select([condition3],[x_select])[condition3]
								y_select2 = np.select([condition3],[y_select])[condition3]
								condition4 = x_select > np.argmax(x_select2)-2
								x_select2 = np.select([condition4],[x_select2])[condition4]
								y_select2 = np.select([condition4],[y_select2])[condition4]
							except ValueError:
								pass
						try:
							popt3,pcov3 = curve_fit(gaus,x_select2,y_select2)
							if np.absolute(best_shift-popt3[1]) < replace_shift_criteria:
								best_shift = popt3[1]
							ax3.plot(x_select2,gaus(x_select2,popt3[0],popt3[1],popt3[2]),color='salmon',label='gaussian fit for selected parameters, gaussian fitted pixel:{}'.format(popt3[1]),alpha=0.5)
							ax3.axvline(x=popt3[1],linewidth=0.5,linestyle='--',color='salmon')
						except RuntimeError:
							pass
						except TypeError:
							pass
				except RuntimeError:
					pass
				except TypeError:
					pass
		

		ax3.plot(np.arange(-delta_pixel_range,delta_pixel_range),xcorr_list,color='black',label='cross correlation',alpha=0.5)
		ax3.plot([best_shift,best_shift],[float(np.min(xcorr_list)),float(np.max(xcorr_list))],'k--',label="best pixel shift:{}".format(best_shift))
		ax3major_ticks = np.arange(-(delta_pixel_range),(delta_pixel_range+1),5)
		ax3minor_ticks = np.arange(-(delta_pixel_range),(delta_pixel_range+1),1)
		ax3.set_xticks(ax3major_ticks)
		ax3.set_xticks(ax3minor_ticks, minor=True)
		ax3.set_title("XCORR PLOT, pixels start at {} with width {}".format(start_pixel,window_width))
		ax3.set_xlabel('pixel shift')
		ax3.set_ylabel('cross correlation')
		ax3.set_xlim(-(delta_pixel_range),(delta_pixel_range))
		ax3.set_ylim(np.min(xcorr_list),np.max(xcorr_list))
		ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.3))
		plt.tight_layout(h_pad=3.3)
		plt.savefig('{}_{}.png'.format(testname,start_pixel), bbox_inches='tight',dpi=128)
		plt.close()

	return diff_wave, best_shift


def wavelengthSolutionFit(data,model,**kwargs):
	"""
	(Deprecated)
	Using the least square fit to calculate the new wavelength solution.
	
	Parameters
	----------
	data:  spectrum object
	       The input telluric data to obtain the initial wavelength solution
	       parameters

	model: model object
		   The telluric model to calculate the delta wavelength

	Optional Parameters
	-------------------
	spec_range  : int
				  range of the spectrum for fitting wavelength solution 
				  in the unit of pixels

	window_width: int
				  width of each xcorr window

	delta_pixel_range: int
				  xcorr window range to calculate the max xcorr

	step_size   : int
				  step size of moving to the next xcorr window

	niter		: int
				  number of interations for fitting the wavelength solution

	Returns
	-------
	wavelength sol: numpy array
				    a new wavelength solution


	"""
	# set up the initial parameters
	spec_range = kwargs.get('spec_range',900)
	width = kwargs.get('window_width',40)
	delta_pixel_range = kwargs.get('delta_pixel_range',15)
	step_size = kwargs.get('step_size',20)
	niter = kwargs.get('niter',1)
	test  = kwargs.get('test',False)

	width_range = np.arange(0,spec_range,step_size)
	width_range_center = width_range + width/2
	pixel = np.delete(np.arange(1024),data.mask)+1

	# store the delta wavelength in each window
	#delta_wave_array = []
	#for i in width_range:
	#	delta_wave_array.append(nsp.pixelWaveShift(data,telluric,i,width,delta_pixel_range))

	data2 = copy.deepcopy(data)

	# fitting the new wavelength solution
	print("initial WFIT:",data2.header['WFIT0'],data2.header['WFIT1'],data2.header['WFIT2'],data2.header['WFIT3'],data2.header['WFIT4'],data2.header['WFIT5'])

	for i in range(niter):
	# getting the parameters of initial wavelength solution 
		k = i + 1
		try:
			wfit0 = data2.header['WFIT0NEW']
			wfit1 = data2.header['WFIT1NEW']
			wfit2 = data2.header['WFIT2NEW']
			wfit3 = data2.header['WFIT3NEW']
			wfit4 = data2.header['WFIT4NEW']
			wfit5 = data2.header['WFIT5NEW']
			c3    = data2.header['C3']
			c4    = data2.header['C4']
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} new wave parameters".format(k))


		except KeyError:
			wfit0 = data2.header['WFIT0']
			wfit1 = data2.header['WFIT1']
			wfit2 = data2.header['WFIT2']
			wfit3 = data2.header['WFIT3']
			wfit4 = data2.header['WFIT4']
			wfit5 = data2.header['WFIT5']
			c3 = 0
			c4 = 0
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} old wave parameters".format(k))


		#if i is 0:
		#	try:
		#		del data2.header['WFIT0NEW']
		#		del data2.header['WFIT1NEW']
		#		del data2.header['WFIT2NEW']
		#		del data2.header['WFIT3NEW']
		#		del data2.header['WFIT4NEW']
		#		del data2.header['WFIT5NEW']
		#		del data2.header['c3']
		#		del data2.header['c4']
		#	except KeyError:
		#		pass
		#	wfit0 = data2.header['WFIT0']
		#	wfit1 = data2.header['WFIT1']
		#	wfit2 = data2.header['WFIT2']
		#	wfit3 = data2.header['WFIT3']
		#	wfit4 = data2.header['WFIT4']
		#	wfit5 = data2.header['WFIT5']
		#	c3 = 0
		#	c4 = 0
		#	p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])

		#else:
		#	pass

		# calcutate the delta wavelentgh
		delta_wave_array = []
		best_shift_list  = []
		for counter, j in enumerate(width_range):
			testname = "loop{}".format(k)
			if i is 0:
				delta_wave, best_shift = nsp.pixelWaveShift(data2,model,j,width,delta_pixel_range,test=test,testname=testname,counter=counter)
			elif i is 1:
				# reduce the delta_pixel_range by 5
				delta_wave, best_shift = nsp.pixelWaveShift(data2,model,j,width,delta_pixel_range-5,test=test,testname=testname,counter=counter)
			else:
				# reduce the delta_pixel_range by 10
				delta_wave, best_shift = nsp.pixelWaveShift(data2,model,j,width,delta_pixel_range-10,test=test,testname=testname,counter=counter)
			best_shift_list.append(best_shift)
			delta_wave_array.append(delta_wave)
		# fit a new wavelength solution
		popt, pcov = curve_fit(nsp.waveSolution, width_range_center, delta_wave_array,p0)

		# outlier rejection
		delta_wave_array = np.asarray(delta_wave_array)
		original_fit = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7])
		m = 3 # number of sigma for outlier rejection
		if k is 1:
			fit_sigma = np.std(original_fit - delta_wave_array)
		else:
			fit_sigma = data2.header['FITSTD']

		width_range_center2 = width_range_center[np.where(abs(original_fit - delta_wave_array) < m*fit_sigma)]
		delta_wave_array2 = delta_wave_array[np.where(abs(original_fit - delta_wave_array) < m*fit_sigma)]

		# drop the last 10 pixels for the fitting
		popt2, pcov2 = curve_fit(nsp.waveSolution,width_range_center2[:-10],delta_wave_array2[:-10],p0)

		# update the parameters
		wfit0 = wfit0+popt2[0]
		wfit1 = wfit1+popt2[1]
		wfit2 = wfit2+popt2[2]
		wfit3 = wfit3+popt2[3]
		wfit4 = wfit4+popt2[4]
		wfit5 = wfit5+popt2[5]
		c3    = c3+popt2[6]
		c4    = c4+popt2[7]
		p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
		
		# update the fits header keywords WFIT0-5, c3, c4
		data2.header['COMMENT']  = 'Below are the keywords added by NIRSPEC_PIP...'
		data2.header['WFIT0NEW'] = wfit0
		data2.header['WFIT1NEW'] = wfit1
		data2.header['WFIT2NEW'] = wfit2
		data2.header['WFIT3NEW'] = wfit3
		data2.header['WFIT4NEW'] = wfit4
		data2.header['WFIT5NEW'] = wfit5
		data2.header['c3']       = c3
		data2.header['c4']       = c4
		data2.bestshift          = data2.bestshift + best_shift_list
		data2.header['FITSTD']   = np.std(nsp.waveSolution(width_range_center2,popt2[0]\
			,popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7]) - delta_wave_array2)

		# plot for analysis
		if test is True:
			print("The new best-shift list:",best_shift_list)
			plt.plot(width_range_center,delta_wave_array,'k.',label="delta wavelength")
			plt.plot(width_range_center2,delta_wave_array2,'b.',label="delta wavelength with ourlier rejection")
			#plt.plot(pixel, nsp.waveSolution(pixel,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]),'g-',alpha=0.5)
			#plt.plot(pixel, nsp.waveSolution(pixel,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7]),'r-',alpha=0.5)
			plt.plot(width_range_center,nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]),'g.',\
				label="fitted wavelength function, STD={}".format(np.std(nsp.waveSolution(width_range_center\
					,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]) - delta_wave_array)),alpha=0.5)
			plt.plot(width_range_center2,nsp.waveSolution(width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7])[:-10],'r.',\
				label="fitted wavelength function with outlier rejection, STD={}".format(np.std(nsp.waveSolution(width_range_center2\
					,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7])[:-10] - delta_wave_array2[:-10])),alpha=0.5)
			plt.title("Pixel to Delta(Wavelength) {} Iteration(s)".format(k))
			plt.xlabel('pixel')
			plt.ylabel('delta(wavelength)')
			plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
			#plt.savefig("/Users/dinohsu/Google Drive/ucsd/research/nirspec/xcorr/pixel_to_delta_wavelength_loop_{}.png".format(k))
			plt.savefig("pixel_to_delta_wavelength_loop_{}.png".format(k),bbox_inches='tight')
			plt.close()
			print("fitted popt loop {}:".format(k),popt)
		
		new_wave_sol = nsp.waveSolution(pixel,wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4)
		data2.wave = new_wave_sol
			
	return new_wave_sol, p0














