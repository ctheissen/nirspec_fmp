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
    Calculate the wavelength solution givene parameters of a 
    fourth order polynomial.
	The equation is modified from NSDRP_Software_Design p.35.

	Parameters
	----------
	wfit0-5: float
			 The fitting parameters in NSDRP_Software_Design p.35
	c3     : float
	         The coefficient of the cubic term of the polynomial
	c4     : float
			 The coefficient of the fourth-power term of the polynomial
	pixel  : int/array
			 The pixel number as the input of wavelength solution
    order  : int
             The order number as the input of wavelength

    Returns
    -------
    The wavelength solution: array-like


    """
    order = kwargs.get('order', None)

    wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
    (wfit3 + wfit4*pixel + wfit5*pixel**2)/order \
    + c3*pixel**3 + c4*pixel**4

    return wave_sol

def getTelluric(wavelow, wavehigh, **kwargs):
	"""
	Get a telluric spectrum.

	Parameters
	----------
	wavelow:  int
			  lower bound of the wavelength range

	wavehigh: int
	          upper bound of the wavelength range

	Optional Parameters
	-------------------
	airmass:  str
			  airmass of the telluric model, either 1.0 or 1.5
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
	tfile = 'pwv_R300k_airmass{}/LBL_A{}_s0_w005_R0300000_T.fits'.format(airmass, 
		am_key[airmass])
	tellurics = fits.open(BASE + '/nirspec_pip/libraries/' + tfile)

	telluric = nsp.Model()
	telluric.wave = np.array(tellurics[1].data['lam'] * 10000)
	telluric.flux = np.array(tellurics[1].data['trans'])**(alpha)

	# select the wavelength range
	criteria = (telluric.wave > wavelow) & (telluric.wave < wavehigh)
	telluric.wave = telluric.wave[criteria]
	telluric.flux = telluric.flux[criteria]
	return telluric

def _continuumFit(wave, a, b, c):
    return a*wave**2 + b*wave + c

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
		wavelow = data.wave[0]
		wavehigh = data.wave[-1]
		model = nsp.getTelluric(wavelow,wavehigh)

	# this second order polynomial continnum correction 
	# works for the order 33 and 34
	popt, pcov = curve_fit(_continuumFit, data.wave, data.flux)
	const = np.mean(data.flux/_continuumFit(data.wave, *popt))-np.mean(model.flux)
	data.flux = data.flux/_continuumFit(data.wave, *popt) - const
	data.noise = data.noise/_continuumFit(data.wave, *popt)

	return data


def xcorrTelluric(data, model, shift, start_pixel, width):
	"""
	Calculate the cross-correlation of the telluric model and data.
	
	The shift (in Angstrom) acts on the model, and this function calculates given
	a starting pixel number and a window width (in pixels).

	This function skips the step of LSF convolution to speed up the calculation.
	
	Parameters
	----------
	data 	   	: spectrum object 
		         telluric data
	model      	: model object 
		   		 telluric model
	shift      	: float 
		   		 wavelength shift of the telluric MODEL in (ANGSTROM)
	start_pixel: int
				 starting pixel number to compute the xcorr
	width       : int
				 window width to compute the xcorr

	Returns
	-------
	xcorr 		: float
				  cross-corelation value
	"""
	# shift the wavelength on the telluric model
	# minus sign means we want to know the shift of the data
	model2 = copy.deepcopy(model)
	model2.wave = model.wave - shift

	# select a range of wavelength to compute x-correlation value
	# the model has the range within that window
	model_low = data.wave[start_pixel]-20
	model_high = data.wave[start_pixel+width]+20
	condition = np.where(model2.wave<model_high)
	model2.wave = model2.wave[condition]
	model2.flux = model2.flux[condition]
	condition = np.where(model2.wave>model_low)
	model2.wave = model2.wave[condition]
	model2.flux = model2.flux[condition]
	
	# LSF of the intrument (skipped)
	vbroad = (299792458/1000)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, 
		flux=model2.flux, vbroad=vbroad, rotate=False, gaussian=True)

	# resampling the telluric model
	model2.flux = np.array(spt.integralResample(xh=model2.wave, 
		yh=model2.flux, xl=data.wave[start_pixel:start_pixel+width]))
	model2.wave = data.wave[start_pixel:start_pixel+width]

	d = data.flux[start_pixel:start_pixel+width]
	##the model is selected in the pixel range in the beginning
	#m = model2.flux[start_pixel:start_pixel+width]
	m = model2.flux
	xcorr = np.inner(d, m)/(np.average(d)*np.average(m))
	
	return xcorr

def pixelWaveShift(data, model, start_pixel, window_width=40, delta_wave_range=10,
	model2=None,test=False,testname=None,counter=None,**kwargs):
	"""
	Find the max cross-correlation and compute the pixel to wavelength shift.
	
	Parameters
	----------
	data
	model: MUST BEFORE resampling and LSF broadening
	model2: model AFTER resampling and LSF broadening (to increase computation speed)


	Returns
	-------
	best wavelength shift: 	float
					  		the wavelength correction after xcorr

	"""
	# the step for wavelength shift, default=0.1 Angstrom
	step = kwargs.get('step',0.1)
	pixel = np.delete(np.arange(1024),data.mask)+1

	xcorr_list = [] # save the xcorr values

	if model2 is None:
		model2 = model
	# select the range of the pixel shift to compute the max xcorr
	for i in np.arange(-delta_wave_range,delta_wave_range,step):
		# propagate the best pixel shift
		if len(data.bestshift) > 1:
			j = data.bestshift[counter] + i
		else:
			j = i
		xcorr = nsp.xcorrTelluric(data,model,j,start_pixel,window_width)
		#print("delta wavelength shift:{}, xcorr value:{}".format(i,xcorr))
		xcorr_list.append(xcorr)
	#print("xcorr list:",xcorr_list)
	best_shift = np.arange(-delta_wave_range,delta_wave_range,
		step)[np.argmax(xcorr_list)]
	central_pixel = start_pixel + window_width/2

	# parameters setup for plotting
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	linewidth=1.0

	pixel = np.delete(np.arange(1024),data.mask)+1

	fig = plt.figure(figsize=(12,8))
	gs1 = gridspec.GridSpec(4, 4)
	ax1 = plt.subplot(gs1[0:2, :])
	ax2 = plt.subplot(gs1[2:, 0:2])
	ax3 = plt.subplot(gs1[2:, 2:])
		
	ax1.plot(data.wave, data.flux, color='black',linestyle='-', 
		label='telluric data',alpha=0.5,linewidth=linewidth)
	ax1.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',
		alpha=0.5,linewidth=linewidth)
	ax1.set_xlabel(r"Wavelength(${\AA}$)")
	ax1.set_ylabel('Transmission')
	ax1.set_ylim(0,1.1)
	ax1.set_xlim(data.wave[0],data.wave[-1])
	#ax1.set_title('Telluric Spectra Region for Cross-Correlation')
	ax1.axvline(x=data.wave[start_pixel],linestyle='--',color='blue',
		linewidth=linewidth)
	ax1.axvline(x=data.wave[start_pixel+window_width],linestyle='--',
		color='blue',linewidth=linewidth)
	ax1.get_xaxis().get_major_formatter().set_scientific(False)
	ax1.legend()
	ax1t = ax1.twiny()
	ax1t.set_xlim(data.wave[0],data.wave[-1])
	ax1t.set_xticks([data.wave[np.where(pixel==200)],
		data.wave[np.where(pixel==400)],
		data.wave[np.where(pixel==600)],
		data.wave[np.where(pixel==800)],
		data.wave[np.where(pixel==1000)]])
	ax1t.set_xticklabels(['200','400','600','800','1000'])
	ax1t.set_xlabel("Pixel")

	ax2.plot(data.wave, data.flux, color='black',linestyle='-', 
		label='telluric data',alpha=0.5,linewidth=linewidth)
	ax2.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',
		alpha=0.5,linewidth=linewidth)
	ax2.set_xlabel(r"Wavelength(${\AA}$)")
	ax2.set_ylabel('Transmission')
	ax2.set_ylim(0,1.1)
	ax2.set_xlim(data.wave[start_pixel]-0.1,
		data.wave[start_pixel+window_width]+0.1)
	ax2.axvline(x=data.wave[start_pixel],linestyle='--',
		color='blue',linewidth=linewidth)
	ax2.axvline(x=data.wave[start_pixel+window_width],linestyle='--',
		color='blue',linewidth=linewidth)
	ax2.get_xaxis().get_major_formatter().set_scientific(False)
	ax2.legend()
	ax2t = ax2.twiny()
	ax2t.set_xlim(data.wave[start_pixel]-0.1,
		data.wave[start_pixel+window_width]+0.1)
	ax2t.set_xticks([data.wave[start_pixel],
		data.wave[start_pixel+window_width]])
	ax2t.set_xticklabels([str(start_pixel),
		str(start_pixel+window_width)])
	ax2t.set_xlabel("Pixel")

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
		if test is True:
			print("start pixel: {}".format(start_pixel))
		#print("x_select:",x_select)
		#print("y_select:",y_select)
		# select the values only around central central maximum
		#roots = xcorr_int_y.derivative().roots()
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
			ax3.plot(x_select,y_select,color='fuchsia',
				label="{} persent range".format(xcorr_fit_percent*100))
			#popt2,pcov2 = curve_fit(gaus,x_select,y_select,p0=[np.max(y_select),mean2,sigma2])
			popt2,pcov2 = curve_fit(gaus,x_select,y_select)
			if np.absolute(popt2[1]) >= delta_wave_range:
				popt2[1] = best_shift
			ax3.plot(x_select,gaus(x_select,popt2[0],popt2[1],popt2[2]),color='olive',
				label='gaussian fit for selected parameters, shift:{}'.format(popt2[1]),
				alpha=0.5)
			ax3.axvline(x=popt2[1],linewidth=0.5,linestyle='--',color='olive')
			#for root in xcorr_int_y.derivative().roots():
			#	ax3.axvline(x=root, linewidth=0.3, color='purple')
			# replace the fitted gaussian value

			#the threshold for replacing the pixel shift with gaussian fit
			replace_shift_criteria = 0.5 
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
					ax3.plot(x_select2,gaus(x_select2,popt3[0],popt3[1],popt3[2]),color='salmon',label=r"gaussian fit for selected parameters, gaussian fitted pixel:{}($\displaystyle \AA $)".format(round(popt3[1],5)),alpha=0.5)
					ax3.axvline(x=popt3[1],linewidth=0.5,linestyle='--',color='salmon')
				
				except RuntimeError:
					pass
				
				except TypeError:
					pass
		
		except RuntimeError:
			pass
		
		except TypeError:
			pass
		

	ax3.plot(np.arange(-delta_wave_range,delta_wave_range,step),xcorr_list,
		color='black',label='cross correlation',alpha=0.5)
	ax3.plot([best_shift,best_shift],[float(np.min(xcorr_list)),
		float(np.max(xcorr_list))],'k:',
		label=r"best wavelength shift:{}($\displaystyle \AA $)".format(\
			round(best_shift,5)))
	ax3major_ticks = np.arange(-(delta_wave_range),(delta_wave_range+1),2)
	ax3minor_ticks = np.arange(-(delta_wave_range),(delta_wave_range+1),0.1)
	ax3.set_xticks(ax3major_ticks)
	ax3.set_xticks(ax3minor_ticks, minor=True)
	ax3.set_title("Cross-Correlation Plot, pixels start at {} with width {}"\
		.format(start_pixel, window_width))
	ax3.set_xlabel(r"Wavelength shift(${\AA}$)")
	ax3.set_ylabel('Cross correlation')
	ax3.set_xlim(-(delta_wave_range),(delta_wave_range))
	ax3.set_ylim(np.min(xcorr_list),np.max(xcorr_list))
	ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
	plt.tight_layout(h_pad=3.3)
	if test is True:
		if testname is None:
			testname = 'test'
		plt.savefig('{}_{}.png'.format(testname,start_pixel), bbox_inches='tight')
	plt.close()

	return best_shift

def wavelengthSolutionFit(data, model, order, **kwargs):
	"""
	Using the least-square method in fitting a 4th order polynomial 
	to obtain a new wavelength solution.
	
	Parameters
	----------
	data 		:  	spectrum object
	       			The input telluric data 

	model 		: 	model object
		   			The telluric model to calculate the delta wavelength

	order		: 	int
				  	spectral order defined by NIRSPEC

	Optional Parameters
	-------------------
	window_width:	int
				 	Width of each xcorr window
				  	Default is 40

	window_step : 	int
				  	Step size between adjacent xcorr windows
				  	Default is 5

	xcorr_range : 	float
				  	Max shift value for each xcorr window in 
				  	WAVELENGTH (Angstrom)
				  	Default is 15 Angstroms


	xcorr_step 	: 	float
				  	Shift step size for each xcorr window
				  	The xcorr calculation ranges from -xcorr_range to
				  	+xcorr_range in steps of this value
				  	Default is 0.05 Angstroms

	niter		: 	int
				  	Number of interations for fitting the wavelength solution
				  	Default is 15

	outlier_rej : 	float
					The number of sigmas in fitting the wavelength solution
					Default is 2

	test 		: 	boolean
					Output the diagnostic plots for each xcorr value
					This option will increase the total calculation time
					Default is False

	save 		: 	boolean
					Save the resulting wavelength solution as a new fits file
					The new file name would be "_new_wave_sol_order" after
					the original name
					The new wavelength solution parameters will be included in
					the corresponding fits header
					Default is False

	save_to_path: 	str
					This option is to specify the path to save the 
					new wavelength solution fits file. 
					The new wavelength solution telluric data will be 
					saved as its original name + 
					'_new_wave_sol_{order}_all.fits'
					Default is None

	data_path 	:	str
					The option specifies the path to the telluric data.
					If it is not specified, this will be the same path as
					of the save_to_path option.
					Default is None


	Returns
	-------
	new_wave_sol: 	numpy array
				    a new wavelength solution

	p0 			:	numpy array
					Fitted wavelength solution paramters

	width_range_center 	: numpy array
					the pixel center array for wavelength fitting

	residual 	: 	numpy array
					the residual of each fitting wavelength 
					correction and the wavelength shift
	
	#std 		:	float
	#				standard deviation for the final fit in Angstrom
	
	#stdV 		: 	float
	#				standard deviation for the final fit in km/s

	"""
	# set up the initial parameters
	#spec_range = kwargs.get('spec_range',900)
	#order = kwargs.get('order', None)
	width = kwargs.get('window_width', 40)
	step_size = kwargs.get('window_step', 5)
	delta_wave_range = kwargs.get('xcorr_range', 10)
	step = kwargs.get('xcorr_step', 0.05)
	niter = kwargs.get('niter', 15)
	outlier_rej = kwargs.get('outlier_rej', 3)
	test  = kwargs.get('test', False) # output the xcorr plots
	save  = kwargs.get('save', False) # save the new wavelength solution
	save_to_path  = kwargs.get('save_to_path', None)
	data_path  = kwargs.get('data_path', None)

	# calculation the necessary parameters
	pixel = np.delete(np.arange(1024),data.mask)+1
	spec_range = len(pixel) - width # window range coverage for xcorr
	width_range = np.arange(0, spec_range, step_size)
	width_range_center = width_range + width/2

	data2 = copy.deepcopy(data)
	model2 = copy.deepcopy(model)
	# model resample and LSF broadening
	model2.flux = np.array(spt.integralResample(xh=model2.wave, 
		yh=model2.flux, xl=data.wave))
	model2.wave = data.wave

	# LSF of the intrument
	#vbroad = (299792458/1000)*np.mean(np.diff(data.wave))/np.mean(data.wave)
	vbroad = nsp.getLSF(data2)
	if test is True:
		print("LSF: ", vbroad)

	model2.flux = apmdl.rotation_broaden.broaden(wave=model2.wave, 
		flux=model2.flux, vbroad=vbroad, rotate=False, gaussian=True)

	# fitting the new wavelength solution
	#if test is True:
	#	print("initial WFIT:",data2.header['WFIT0'],data2.header['WFIT1'],
	#		data2.header['WFIT2'],data2.header['WFIT3'],data2.header['WFIT4'],
	#		data2.header['WFIT5'])

	time0 = time.time()

	for i in range(niter):
	# getting the parameters of initial wavelength solution 
		k = i + 1
		time1 = time.time()
		if i is 0:
			wfit0 = data2.header['WFIT0']
			wfit1 = data2.header['WFIT1']
			wfit2 = data2.header['WFIT2']
			wfit3 = data2.header['WFIT3']
			wfit4 = data2.header['WFIT4']
			wfit5 = data2.header['WFIT5']
			c3 = 0
			c4 = 0
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("original p0: ",p0)

		else:
			wfit0 = data2.header['WFIT0NEW']
			wfit1 = data2.header['WFIT1NEW']
			wfit2 = data2.header['WFIT2NEW']
			wfit3 = data2.header['WFIT3NEW']
			wfit4 = data2.header['WFIT4NEW']
			wfit5 = data2.header['WFIT5NEW']
			c3    = data2.header['C3']
			c4    = data2.header['C4']
			popt0_ori = data2.header['POPT0']
			popt1_ori = data2.header['POPT1']
			popt2_ori = data2.header['POPT2']
			popt3_ori = data2.header['POPT3']
			popt4_ori = data2.header['POPT4']
			popt5_ori = data2.header['POPT5']
			popt6_ori = data2.header['POPT6']
			popt7_ori = data2.header['POPT7']
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])

		# calcutate the delta wavelentgh
		best_shift_list  = []
		for counter, j in enumerate(width_range):
			testname = "loop{}".format(k)
			if i is 0:
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,
					delta_wave_range,model2,test=test,testname=testname,
					counter=counter,step=step)
				time3 = time.time()
				if test is True:
					print("xcorr time: {} s".format(round(time3-time2,4)))
			elif i is 1:
				# reduce the delta_wave_range as 5
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,5,
					model2,test=test,testname=testname,counter=counter,step=step)
				time3 = time.time()
				if test is True:
					print("xcorr time: {} s".format(round(time3-time2,4)))
			else:
				# reduce the delta_wave_range as 2
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,2,model2,
					test=test,testname=testname,counter=counter,step=step)
				time3 = time.time()
				if test is True:
					print("xcorr time: {} s".format(round(time3-time2,4)))
			#else:
			#	best_shift = nsp.pixelWaveShift(data2,model,j,width,0.2,model2,test=test,testname=testname,counter=counter,step=step)
			#else:
			#	best_shift = nsp.pixelWaveShift(data2,model,j,width,0.15,model2,test=test,testname=testname,counter=counter,step=0.001)

			best_shift_list.append(best_shift)
		print("best_shift_list:",best_shift_list)

		time4 = time.time()
		if test is True:
			print("Total X correlation time for loop {}: {} s".format(k, 
				round(time4-time1,4)))
		# fit a new wavelength solution
		def waveSolutionFn1(orderNum):
			def fitFn(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4):
				order = float(orderNum)
				wave_sol = wfit0 + wfit1*pixel + wfit2*pixel**2 + \
				(wfit3 + wfit4*pixel + wfit5*pixel**2)/order \
				+ c3*pixel**3 + c4*pixel**4

				return wave_sol
			return fitFn

		popt, pcov = curve_fit(waveSolutionFn1(order), width_range_center, 
			np.asarray(best_shift_list),p0)

		#popt, pcov = curve_fit(lambda pixel, wfit0, wfit1, wfit2, wfit3, wfit4, 
		#	wfit5, c3, c4: nsp.waveSolution(pixel, wfit0, wfit1, wfit2, wfit3, 
		#		wfit4, wfit5, c3, c4,order=order), width_range_center, 
		#	np.asarray(best_shift_list),p0)
		print("1st width_range_center: ",width_range_center)
		print("1st best_shift_list: ",best_shift_list)
		print("1st popt",popt)

		# outlier rejection
		best_shift_array = np.asarray(best_shift_list)
		
		m = outlier_rej # number of sigma for outlier rejection
		if k is 1:
			fit_sigma = 0.8
			#fit_sigma = np.std(original_fit - best_shift_array)
			original_fit = nsp.waveSolution(width_range_center,popt[0],
				popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],
				order=order)
		elif k < 6:
			fit_sigma = data2.header['FITSTD']
			original_fit = nsp.waveSolution(width_range_center,popt[0],
				popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],
				order=order)
		else:
			fit_sigma = data2.header['FITSTD']
			original_fit = nsp.waveSolution(width_range_center,popt0_ori,
				popt1_ori,popt2_ori,popt3_ori,popt4_ori,popt5_ori,
				popt6_ori,popt7_ori,order=order)

		# exclude the edge pixels in the fitting
		#width_range_center2 = width_range_center[5:-5]
		#best_shift_array2 = best_shift_array[5:-5]
		#width_range_center2 = width_range_center2[np.where(\
		#	abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		#best_shift_array2 = best_shift_array2[np.where(\
		#	abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		
		width_range_center2 = width_range_center[np.where\
		(abs(original_fit - best_shift_array) < m*fit_sigma)]
		
		best_shift_array2 = best_shift_array[np.where\
		(abs(original_fit - best_shift_array) < m*fit_sigma)]		

		if len(width_range_center2) < 8:
			print("Number of selected pixel < number of fits parameters (8)")
			break

		# fit the wavelength again after the outlier rejections
		popt2, pcov2 = curve_fit(waveSolutionFn1(order),
			width_range_center2,best_shift_array2,p0)
		#popt2, pcov2 = curve_fit(lambda pixel, wfit0, wfit1, wfit2, wfit3, 
		#	wfit4, wfit5, c3, c4: nsp.waveSolution(pixel, wfit0, wfit1, 
		#		wfit2, wfit3, wfit4, wfit5, c3, c4,order=order),
		#	width_range_center2,best_shift_array2,p0)
		print("2nd width_range_center: ",width_range_center2)
		print("2nd best_shift_list: ",best_shift_array2)
		print("2nd popt",popt2)

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
		print("fitted p0: ",p0)
		
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
		data2.header['FITSTD']   = np.std(nsp.waveSolution(width_range_center2,
			popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],
			popt2[7],order=order) - best_shift_array2)
		data2.header['POPT0']	= popt2[0]
		data2.header['POPT1']	= popt2[1]
		data2.header['POPT2']	= popt2[2]
		data2.header['POPT3']	= popt2[3]
		data2.header['POPT4']	= popt2[4]
		data2.header['POPT5']	= popt2[5]
		data2.header['POPT6']	= popt2[6]
		data2.header['POPT7']	= popt2[7]

		new_wave_sol = nsp.waveSolution(pixel,wfit0,wfit1,wfit2,wfit3,
			wfit4,wfit5,c3,c4,order=order)

		time5 = time.time()
		if test is True:
			print("Pixel wavelength fit time for loop {}: {} s".format(k,
				round(time5-time4,4)))
		# plot for analysis
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		fig = plt.figure(figsize=(8,8))
		gs1 = gridspec.GridSpec(5, 4)
		ax1 = plt.subplot(gs1[0:2,0:])
		ax2 = plt.subplot(gs1[2:4,0:])
		ax3 = plt.subplot(gs1[4:,0:], sharex=ax2)
		
		ax1.xaxis.tick_top()
		#ax1.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=0.8)
		ax1.plot(new_wave_sol,data.flux,color='black',linestyle='-',label='corrected telluric data',alpha=0.8,linewidth=0.8)
		ax1.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',alpha=0.5)
		ax1.set_xlabel(r"Wavelength($\displaystyle \AA $)")
		ax1.set_ylabel('Transmission')
		ax1.xaxis.set_label_position('top') 
		ax1.set_ylim(0,1.1)
		ax1.set_xlim(min(data.wave[0],new_wave_sol[0]),max(data.wave[-1],new_wave_sol[-1]))
		#ax1.set_title("Telluric Spectra Region for Cross-Correlation:{} Iteration".format(k))
		ax1.get_xaxis().get_major_formatter().set_scientific(False)
		ax1.legend()
		residual1 = nsp.waveSolution(width_range_center,popt[0],
			popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],
			order=order)-best_shift_array
		residual2 = nsp.waveSolution(width_range_center2,popt2[0],
			popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],
			popt2[7],order=order)-best_shift_array2
		std = round(np.std(residual2),5)		
		ax2.plot(width_range_center,best_shift_array,'k.',label="delta wavelength")
		ax2.plot(width_range_center2,best_shift_array2,'b.',
			label="delta wavelength with ourlier rejection")
		#ax2.plot(width_range_center,nsp.waveSolution(width_range_center,
		#	popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],
		#	order=order),'g.',label="fitted wavelength function".format(np.std(residual1)),alpha=0.5)
		ax2.plot(width_range_center2,nsp.waveSolution(width_range_center2,
			popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],
			popt2[7],order=order),'r.',
			label=r"fitted wavelength function with outlier rejection, STD={} $\displaystyle \AA $={} km/s".format(\
			std,std/np.average(new_wave_sol)*299792.458),alpha=0.5)
		ax2.set_ylabel(r"$\displaystyle\Delta \lambda (\displaystyle \AA)$")
		ax2.legend()
		# plot the residual
		plt.setp(ax2.get_xticklabels(), visible=False)
		yticks = ax3.yaxis.get_major_ticks()
		yticks[-1].label1.set_visible(False)
		#ax3.plot(width_range_center,residual1,'g.',alpha=0.5)
		ax3.plot(width_range_center2, residual2,'r.',alpha=0.5)
		ax3.set_ylim(-3*fit_sigma,3*fit_sigma)
		ax3.set_ylabel(r"$residual (\displaystyle \AA)$")
		ax3.set_xlabel(r'Pixel')
		#ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.5))
		plt.subplots_adjust(hspace=.0)
		plt.savefig("pixel_to_delta_wavelength_loop_{}.png".format(k),
			bbox_inches='tight')
		plt.close()

		time6 = time.time()
		if test is True:
			print("Plot time:",format(round(time6-time5,4)))
		
		if data2.header['FITSTD']>fit_sigma and k is not 1:
			print("Wavelength solution converges in {} loops, with STD {} Angstrom ({} km/s)".format(k,
				np.std(residual2),np.std(residual2)/np.average(new_wave_sol)*299792.458))
			print("Total calculation time: {} min".format(round((time6-time0)/60,4)))
			break

	data2.wave = new_wave_sol

	if save is True:
		if data_path is None:
			data_path = save_to_path + '_' + str(order) + '_all.fits'
		save_name = save_to_path + "_new_wave_sol_{}_all.fits".format(order)
		with fits.open(data_path) as hdulist:
			hdulist[0].header['COMMENT']  = 'Below are the keywords added by NIRSPEC_PIP...'
			hdulist[0].header['WFIT0NEW'] = wfit0
			hdulist[0].header['WFIT1NEW'] = wfit1
			hdulist[0].header['WFIT2NEW'] = wfit2
			hdulist[0].header['WFIT3NEW'] = wfit3
			hdulist[0].header['WFIT4NEW'] = wfit4
			hdulist[0].header['WFIT5NEW'] = wfit5
			hdulist[0].header['c3']       = c3
			hdulist[0].header['c4']       = c4
			hdulist[0].bestshift          = data2.bestshift + best_shift_list
			hdulist[0].header['FITSTD']   = np.std(nsp.waveSolution(\
				width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],
				popt2[4],popt2[5],popt2[6],popt2[7],
				order=order) - best_shift_array2)
			hdulist[0].header['POPT0']	= popt2[0]
			hdulist[0].header['POPT1']	= popt2[1]
			hdulist[0].header['POPT2']	= popt2[2]
			hdulist[0].header['POPT3']	= popt2[3]
			hdulist[0].header['POPT4']	= popt2[4]
			hdulist[0].header['POPT5']	= popt2[5]
			hdulist[0].header['POPT6']	= popt2[6]
			hdulist[0].header['POPT7']	= popt2[7]
			hdulist[0].header['STD']    = str(np.std\
				(residual2)/np.average(new_wave_sol)*299792.458) + 'km/s'
			hdulist[0].data             = nsp.waveSolution(np.arange(1024)+1,
				wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4,order=order)
			hdulist.writeto(save_name,overwrite=True)
		print("The new wavelength solution file is saved to {}".format(save_name))

	std = np.std(residual2)
	stdV = np.std(residual2)/np.average(new_wave_sol)*299792.458
	#return new_wave_sol, p0, np.std(residual2), np.std(residual2)/np.average(new_wave_sol)*299792.458
	return new_wave_sol, p0, width_range_center2, residual2, best_shift_list
	#return new_wave_sol, p0, std, stdV

### Below is a test
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

	order		: int
				  spectral order defined by NIRSPEC

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
	order = kwargs.get('order',33)
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
		time1 = time.time()
		if i is 0:
			wfit0 = data2.header['WFIT0']
			wfit1 = data2.header['WFIT1']
			wfit2 = data2.header['WFIT2']
			wfit3 = data2.header['WFIT3']
			wfit4 = data2.header['WFIT4']
			wfit5 = data2.header['WFIT5']
			c3 = 0
			c4 = 0
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} old wave parameters, time {} s".format(k,time1))

		else:
			wfit0 = data2.header['WFIT0NEW']
			wfit1 = data2.header['WFIT1NEW']
			wfit2 = data2.header['WFIT2NEW']
			wfit3 = data2.header['WFIT3NEW']
			wfit4 = data2.header['WFIT4NEW']
			wfit5 = data2.header['WFIT5NEW']
			c3    = data2.header['C3']
			c4    = data2.header['C4']
			popt0_ori = data2.header['POPT0']
			popt1_ori = data2.header['POPT1']
			popt2_ori = data2.header['POPT2']
			popt3_ori = data2.header['POPT3']
			popt4_ori = data2.header['POPT4']
			popt5_ori = data2.header['POPT5']
			popt6_ori = data2.header['POPT6']
			popt7_ori = data2.header['POPT7']
			p0 = np.array([wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4])
			print("Loop {} new wave parameters, time {} s".format(k,time1))

		# calcutate the delta wavelentgh
		best_shift_list  = []
		for counter, j in enumerate(width_range):
			testname = "loop{}".format(k)
			if i is 0:
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,delta_wave_range,model2,test=test,testname=testname,counter=counter,step=step)
				time3 = time.time()
				print("xcorr time: {} s".format(round(time3-time2,4)))
			elif i is 1:
				# reduce the delta_wave_range by 5
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,5,model2,test=test,testname=testname,counter=counter,step=step)
				time3 = time.time()
				print("xcorr time: {} s".format(round(time3-time2,4)))
			else:
				# reduce the delta_wave_range by 8
				time2 = time.time()
				best_shift = nsp.pixelWaveShift(data2,model,j,width,2,model2,test=test,testname=testname,counter=counter,step=step)
				time3 = time.time()
				print("xcorr time: {} s".format(round(time3-time2,4)))
			#else:
			#	best_shift = nsp.pixelWaveShift(data2,model,j,width,0.2,model2,test=test,testname=testname,counter=counter,step=step)
			#else:
			#	best_shift = nsp.pixelWaveShift(data2,model,j,width,0.15,model2,test=test,testname=testname,counter=counter,step=0.001)

			best_shift_list.append(best_shift)

		time4 = time.time()
		print("Total X correlation time: {} s".format(round(time4-time1,4)))
		# fit a new wavelength solution
		popt, pcov = curve_fit(lambda pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4: \
			nsp.waveSolution(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4,order=order),
			width_range_center, best_shift_list,p0)

		# outlier rejection
		best_shift_array = np.asarray(best_shift_list)
		
		m = 2 # number of sigma for outlier rejection
		if k is 1:
			fit_sigma = 1.
			#fit_sigma = np.std(original_fit - best_shift_array)
			original_fit = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],order=order)
		elif k < 6:
			fit_sigma = data2.header['FITSTD']
			original_fit = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],order=order)
		else:
			fit_sigma = data2.header['FITSTD']
			original_fit = nsp.waveSolution(width_range_center,popt0_ori,popt1_ori,popt2_ori,popt3_ori,popt4_ori,popt5_ori,popt6_ori,popt7_ori,order=order)

		# exclude the edge pixels in the fitting
		#width_range_center2 = width_range_center[5:-5]
		#best_shift_array2 = best_shift_array[5:-5]
		#width_range_center2 = width_range_center2[np.where(abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		#best_shift_array2 = best_shift_array2[np.where(abs(original_fit[5:-5] - best_shift_array[5:-5]) < m*fit_sigma)]
		width_range_center2 = width_range_center[np.where(abs(original_fit - best_shift_array) < m*fit_sigma)]
		best_shift_array2 = best_shift_array[np.where(abs(original_fit - best_shift_array) < m*fit_sigma)]		

		if len(width_range_center2) < 8:
			print("Number of selected pixel < number of fits parameters (8)")
			break

		# drop the last 10 pixels for the fitting
		popt2, pcov2 = curve_fit(lambda pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4: \
			nsp.waveSolution(pixel, wfit0, wfit1, wfit2, wfit3, wfit4, wfit5, c3, c4,order=order),width_range_center2,best_shift_array2,p0)

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
			,popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7],order=order) - best_shift_array2)
		data2.header['POPT0']	= popt2[0]
		data2.header['POPT1']	= popt2[1]
		data2.header['POPT2']	= popt2[2]
		data2.header['POPT3']	= popt2[3]
		data2.header['POPT4']	= popt2[4]
		data2.header['POPT5']	= popt2[5]
		data2.header['POPT6']	= popt2[6]
		data2.header['POPT7']	= popt2[7]

		new_wave_sol = nsp.waveSolution(pixel,wfit0,wfit1,wfit2,wfit3,wfit4,wfit5,c3,c4,order=order)

		time5 = time.time()
		print("Pixel wavelength fit time: {} s".format(round(time5-time4,4)))
		# plot for analysis
		if test is True:
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')
			fig = plt.figure(figsize=(8,8))
			gs1 = gridspec.GridSpec(5, 4)
			ax1 = plt.subplot(gs1[0:2,0:])
			ax2 = plt.subplot(gs1[2:4,0:])
			ax3 = plt.subplot(gs1[4:,0:], sharex=ax2)
		
			ax1.xaxis.tick_top()
			ax1.plot(data.wave, data.flux, color='black',linestyle='-', label='telluric data',alpha=0.5,linewidth=0.8)
			ax1.plot(new_wave_sol,data.flux,color='black',linestyle='-',label='corrected telluric data',alpha=0.8,linewidth=0.8)
			ax1.plot(model2.wave, model2.flux, 'r-' ,label='telluric model',alpha=0.5)
			ax1.set_xlabel(r"Wavelength($\displaystyle \AA $)")
			ax1.set_ylabel('Transmission')
			ax1.xaxis.set_label_position('top') 
			ax1.set_ylim(0,1.1)
			ax1.set_xlim(min(data.wave[0],new_wave_sol[0]),max(data.wave[-1],new_wave_sol[-1]))
			#ax1.set_title("Telluric Spectra Region for Cross-Correlation:{} Iteration".format(k))
			ax1.get_xaxis().get_major_formatter().set_scientific(False)
			ax1.legend()
			residual1 = nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],order=order)-best_shift_array
			residual2 = nsp.waveSolution(width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7],order=order)-best_shift_array2
			std = round(np.std(residual2),5)			
			ax2.plot(width_range_center,best_shift_array,'k.',label="delta wavelength")
			ax2.plot(width_range_center2,best_shift_array2,'b.',label="delta wavelength with ourlier rejection")
			ax2.plot(width_range_center,nsp.waveSolution(width_range_center,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],order=order),'g.',\
				label="fitted wavelength function".format(np.std(residual1)),alpha=0.5)
			ax2.plot(width_range_center2,nsp.waveSolution(width_range_center2,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6],popt2[7],order=order),'r.',\
				label=r"fitted wavelength function with outlier rejection, STD={} $\displaystyle \AA $={} km/s".format(\
				std,std/np.average(new_wave_sol)*299792.458),alpha=0.5)
			ax2.set_ylabel(r"$\displaystyle\Delta \lambda (\displaystyle \AA)$")
			ax2.legend()
			# plot the residual
			plt.setp(ax2.get_xticklabels(), visible=False)
			yticks = ax3.yaxis.get_major_ticks()
			yticks[-1].label1.set_visible(False)
			ax3.plot(width_range_center,residual1,'g.',alpha=0.5)
			ax3.plot(width_range_center2, residual2,'r.',alpha=0.5)
			ax3.set_ylim(-3*fit_sigma,3*fit_sigma)
			ax3.set_ylabel(r"$residual (\displaystyle \AA)$")
			ax3.set_xlabel(r'Pixel')
			#ax3.legend(loc=9, bbox_to_anchor=(0.5, -0.5))
			plt.subplots_adjust(hspace=.0)
			plt.savefig("pixel_to_delta_wavelength_loop_{}.png".format(k),bbox_inches='tight')
			plt.close()

			time6 = time.time()
			print("Plot time:",format(round(time6-time5,4)))
		
		if data2.header['FITSTD']>fit_sigma and k is not 1:
			print("Wavelength solution converges in {} loops, with STD {} Angstrom ({} km/s)".format(k,np.std(residual2),np.std(residual2)/np.average(new_wave_sol)*299792.458))
			break

	data2.wave = new_wave_sol

	#return new_wave_sol, p0, np.std(residual2), np.std(residual2)/np.average(new_wave_sol)*299792.458
	return new_wave_sol, p0, width_range_center2, residual2












