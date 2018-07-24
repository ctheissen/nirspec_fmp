import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import fits
import copy
import os
import shutil
import glob
from wavelets import WaveletAnalysis
import time

###################################################
# Defringe Flat method 
# adopted from ROJO & HARRINGTON 2006
###################################################

def defringeflat(flat_file, wbin=32, start_col=10, 
	end_col=980 ,diagnostic=True, save_to_path=None,filename=None):
	"""
	This function is to remove the fringe pattern using
	the method described in Rojo and Harrington (2006).

	Use a fifth order polynomial to remove the continuum.

	Parameters
	----------
	flat_file 		: 	fits
						original flat file

	Optional Parameters
	-------------------
	wbin 			:	int
						the bin width to calculate each 
						enhance row
						Default is 32

	start_col 		: 	int
						starting column number for the
						wavelet analysis
						Default is 10

	end_col 		: 	int
						ending column number for the
						wavelet analysis
						Default is 980

	diagnostic 		: 	boolean
						output the diagnostic plots
						Default is True

	Returns
	-------
	defringe file 	: 	fits
						defringed flat file

	"""
	# the path to save the diagnostic plots
	#save_to_path = 'defringeflat/allflat/'

	flat = flat_file

	# initial flat plot
	if diagnostic is True:
		fig = plt.figure(figsize=(8,8))
		fig.suptitle("original flat",fontsize=14)
		gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1]) 
		ax0 = plt.subplot(gs[0])
		ax0.imshow(flat[0].data,cmap='gray',vmax=12000)
		ax0.set_ylabel("Row number")
		ax1 = plt.subplot(gs[1],sharex=ax0)
		ax1.plot(flat[0].data[60,:],'k-',
			alpha=0.5, label='60th row profile')
		ax1.set_ylabel("Amp (DN)")
		ax1.set_xlabel("Column number")
		ax1.set_ylim(6000,10000)
		fig.subplots_adjust(hspace=0)
		plt.legend()
		plt.savefig(save_to_path + "defringeflat_{}_0_original_flat.png"\
			.format(filename))
		plt.close()

	defringeflat_img = flat_file

	for k in np.arange(0,1024,wbin):
		# extract the patch from the fits file
		flat_patch = flat[0].data[k:k+wbin,:]

		# mean average the selected region in the order
		flat_patch_mean = np.mean(flat_patch,axis=0)

		# continuum fit
		pcont = np.polyfit(np.arange(start_col,end_col),
			flat_patch_mean[start_col:end_col],5)
		cont_fit = np.polyval(pcont, np.arange(0,1024))

		# use wavelets package: WaveletAnalysis
		enhance_row = flat_patch_mean - cont_fit
		dt = 0.1
		wa = WaveletAnalysis(enhance_row[start_col:end_col],
		 dt=dt)
		# wavelet power spectrum
		power = wa.wavelet_power
		# scales
		cales = wa.scales
		# associated time vector
		t = wa.time
		# reconstruction of the original data
		rx = wa.reconstruction()

		# reconstruct the fringe image
		img_length = wbin * 1024
		reconstruct_image = np.zeros(img_length).reshape(wbin,1024)
		for i in range(wbin):
			for j in np.arange(start_col,end_col):
				reconstruct_image[i,j] = rx[j - start_col]
		
		defringeflat_img[0].data[k:k+wbin,:] -= reconstruct_image
		print("{} row starting {} is done".format(filename,k))

		# diagnostic plots
		if diagnostic is True:
			# middle cut plot
			fig = plt.figure(figsize=(12,4))
			fig.suptitle("middle cut at row{}".format(k+wbin/2),
        	 fontsize=14)
			ax1 = fig.add_subplot(2,1,1)
			ax1.imshow(flat_patch, cmap='gray',vmax=12000)
			ax1.set_ylim(0,100)
			ax1.set_ylabel("Row number")
			ax1.invert_yaxis()
			ax2 = fig.add_subplot(2,1,2, sharex=ax1)
			ax2.plot(flat_patch[wbin/2,:],'k-',alpha=0.5)
			ax2.set_ylim(6000,10000)
			ax2.set_ylabel("Amp (DN)")
			ax2.set_xlabel("Column number")
			fig.subplots_adjust(hspace=0)
			plt.savefig(save_to_path + \
				'defringeflat_{}_flat_start_row_{}_middle_profile.png'\
				.format(filename,k))
			plt.close()

        	# continuum fit plot
			fig = plt.figure()
			fig.suptitle("continuum fit row {}-{}".format(k,k+wbin),fontsize=14)
			gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
			ax0 = plt.subplot(gs[0])
			ax0.plot(flat_patch_mean,'k-',alpha=0.5,
				label='mean average patch')
			ax0.plot(cont_fit,'r-',alpha=0.5,label='continuum fit')
			ax0.set_ylim(6000,10000)
			ax0.set_ylabel("Amp (DN)")
			plt.legend()
			ax1 = plt.subplot(gs[1])
			ax1.plot(flat_patch_mean - cont_fit,'k-',
				alpha=0.5, label='residual')
			ax1.set_ylabel("Amp (DN)")
			ax1.set_xlabel("Column number")
			ax1.set_ylim(-500,500)
			fig.subplots_adjust(hspace=0)
			plt.legend()
			plt.savefig(save_to_path + \
				"defringeflat_{}_start_row_{}_continuum_fit.png".\
				format(filename,k))
			plt.close()

			# enhance row vs. reconstructed wavelet plot
			try:
				fig = plt.figure(figsize=(12,4))
				fig.suptitle("reconstruct fringe conparison row {}-{}".\
					format(k,k+wbin), fontsize=14)
				ax1 = fig.add_subplot(3,1,1)
				ax1.plot(enhance_row,'k-',alpha=0.5,
					label="enhance_row start row {}".format(k))
				ax1.set_ylim(-500,500)
				ax1.set_ylabel("Amp (DN)")
				plt.legend()
				ax2 = fig.add_subplot(3,1,2, sharex=ax1)
				ax2.plot(rx,'k-',alpha=0.5,
					label='reconstructed fringe pattern')
				ax2.set_ylim(-500,500)
				ax2.set_ylabel("Amp (DN)")
				plt.legend()
				ax3 = fig.add_subplot(3,1,3, sharex=ax1)
				ax3.plot(enhance_row[start_col:end_col] - rx,
					'k-',alpha=0.5, label='residual')
				ax3.set_ylim(-500,500)
				ax3.set_ylabel("Amp (DN)")
				ax3.set_xlabel("Column number")
				plt.legend()
				fig.subplots_adjust(hspace=0)
				plt.savefig(save_to_path + \
					"defringeflat_{}_start_row_{}_reconstruct_profile.png".\
					format(filename,k))
				plt.close()
			except RuntimeError:
				print("CANNOT GENERATE THE PLOT defringeflat\
					_{}_start_row_{}_reconstruct_profile.png"\
					.format(filename,k))
				pass

			# reconstruct image comparison plot
			fig = plt.figure(figsize=(12,4))
			fig.suptitle("reconstruct image row {}-{}".\
				format(k,k+wbin)
				, fontsize=14)
			ax1 = fig.add_subplot(3,1,1)
			ax1.imshow(flat_patch, cmap='gray',vmax=12000,
				label='flat O33')
			ax1.set_ylabel("Row number")
			ax1.set_ylim(0,30)
			plt.legend()
			ax2 = fig.add_subplot(3,1,2, sharex=ax1)
			ax2.imshow(reconstruct_image, cmap='gray',vmax=500,
				label='reconstructed fringe image')
			ax2.set_ylabel("Row number")
			ax2.set_ylim(0,30)
			plt.legend()
			ax3 = fig.add_subplot(3,1,3, sharex=ax1)
			ax3.imshow(flat_patch-reconstruct_image, 
				cmap='gray',vmax=12000,label='residaul')
			ax3.set_ylim(0,30)
			ax3.set_ylabel("Row number")
			ax3.set_xlabel("Column number")
			plt.legend()
			fig.subplots_adjust(hspace=0)
			plt.savefig(save_to_path + \
				"defringeflat_{}_start_row_{}_reconstruct_image.png".\
				format(filename,k))
			plt.close()

			# middle residual comparison plot
			fig = plt.figure(figsize=(12,4))
			fig.suptitle("middle row comparison row {}-{}".\
				format(k,k+wbin)
				, fontsize=14)
			ax1 = fig.add_subplot(3,1,1)
			ax1.plot(flat_patch[wbin/2,:],'k-',alpha=0.5,
				label='original flat row {}'.format(k+wbin/2))
			ax1.set_ylim(6000,10000)
			ax1.set_ylabel("Amp (DN)")
			plt.legend()
			ax2 = fig.add_subplot(3,1,2, sharex=ax1)
			ax2.plot(flat_patch[wbin/2,:]-\
				reconstruct_image[wbin/2,:],'k-',
				alpha=0.5, label='defringe flat row {}'.format(wbin/2))
			ax2.set_ylim(6000,10000)
			ax2.set_ylabel("Amp (DN)")
			plt.legend()
			ax3 = fig.add_subplot(3,1,3, sharex=ax1)
			ax3.plot(reconstruct_image[wbin/2,:],'k-',alpha=0.5,
				label='difference')
			ax3.set_ylim(-500,500)
			ax3.set_ylabel("Amp (DN)")
			ax3.set_xlabel("Column number")
			plt.legend()
			fig.subplots_adjust(hspace=0)
			plt.savefig(save_to_path + \
				"defringeflat_{}_start_row_{}_defringe_middle_profile.png".\
				format(filename,k))
			plt.close()

	# final diagnostic plot
	if diagnostic is True:
		fig = plt.figure(figsize=(8,8))
		fig.suptitle("defringe flat",fontsize=14)
		gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1]) 
		ax0 = plt.subplot(gs[0])
		ax0.imshow(defringeflat_img[0].data,cmap='gray',vmax=12000)
		ax0.set_ylabel("Row number")
		ax1 = plt.subplot(gs[1],sharex=ax0)
		ax1.plot(defringeflat_img[0].data[60,:],'k-',
			alpha=0.5, label='60th row profile')
		ax1.set_ylabel("Amp (DN)")
		ax1.set_xlabel("Column number")
		ax1.set_ylim(6000,10000)
		fig.subplots_adjust(hspace=0)
		plt.legend()
		plt.savefig(save_to_path + "defringeflat_{}_0_defringe_flat.png"\
			.format(filename))
		plt.close()

	hdu = fits.PrimaryHDU(data=defringeflat_img[0].data)
	hdu.header = flat_file[0].header
	return hdu

def defringeflatAll(data_folder_path, wbin=32, start_col=10, 
	end_col=980 ,diagnostic=True):
	"""
	Perform the defringe flat function and save the 
	efringed flat files under the data folder and 
	move the raw flat files under anotehr folder 
	called "defringefalt_diagnostics" with optional 
	diagnostic plots.

	Parameters
	----------
	data_folder_path: 	str
						data folder for processing defringe flat

	Optional Parameters
	-------------------
	wbin 			:	int
						the bin width to calculate each 
						enhance row
						Default is 32

	start_col 		: 	int
						starting column number for the
						wavelet analysis
						Default is 10

	end_col 		: 	int
						ending column number for the
						wavelet analysis
						Default is 980

	diagnostic 		: 	boolean
						output the diagnostic plots
						The option may cause much more 
						computation time and have some issues
						in plotting.
						Default is True

	Returns
	-------
	defringe file 	: 	fits
						defringed flat file

	Examples
	--------
	>>> import nirspec_pip as nsp
	>>> nsp.defringeflatAll(data_folder_path, diagnostic=False)

	"""
	originalpath = os.getcwd()

	save_to_path = data_folder_path + '/defringeflat_diagnostic/'
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)

	# store the fits file names
	files = glob.glob1(data_folder_path,'*.fits')

	for filename in files:
		file_path = data_folder_path + filename
		
		data = fits.open(file_path, ignore_missing_end=True)
		if ('flat field' in str(data[0].header['COMMENT'])) is True:
			defringeflat_file = defringeflat(data, 
				wbin=wbin, start_col=start_col, 
				end_col=end_col ,diagnostic=diagnostic, 
				save_to_path=save_to_path,filename=filename)
			save_name = data_folder_path + filename.split('.')[0] + \
			'_defringe.fits'
			defringeflat_file.writeto(save_name, clobber=True,
				output_verify='ignore')
			file_path2 = save_to_path + filename
			shutil.move(file_path, file_path2)

	return None

#### Below is the test script for defringeflat function
## read in the median-combined flat
#flat =  fits.open('data/Burgasser/J0720-0846/2014jan19/reduced/flats/jan19s001_flat_0.fits')
#
#time1 = time.time()
#
#defringeflat(flat, diagnostic=False)
#
#time2 = time.time()
#print("total time: {} s".format(time2-time1))
