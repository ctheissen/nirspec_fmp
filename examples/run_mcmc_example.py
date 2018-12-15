###############################################################################################
#																							  #
# Forward-Modeling routine using multiprocessing emcee package (Foreman-Mackey et al. (2013)) #
#																							  #
###############################################################################################

import nirspec_fmp as nsp
import pandas as pd
import numpy as np
from astropy.io import fits
import copy
import matplotlib.pyplot as plt
import sys
import os

#########################################
## Parameters Set Up
#########################################


data_dic = {
'20160216':{'feb16s0080':'feb16s0076','feb16s0081':'feb16s0078'
}
}

source                 = 'J1106+2754'
order                  = 33
ndim, nwalkers, step   = 8, 50, 600
burn                   = 500 # mcmc sampler chain burn
moves                  = 2.0
ndim2, nwalkers2, step2= 8, 50, 500
burn2                  = 400
applymask              = False
pixel_start, pixel_end = 10, -40
alpha_tell             = 1.0 # telluric alpha for LSF
plot_show              = False
custom_mask            = [427,428]#[183,190,195,649,650,652,655,656,820,821,824,825,827,828,829,831,832,834]
outlier_rejection      = 2.2 #5

## coadd
coadd                  = True
sci_data_name2         = 'feb16s0081'

modelset               = 'btsettl08'
BASE                   = '/Users/dinohsu/projects/nirspec_fmp/nirspec_fmp/forward_model/'
data_BASE              = '/Volumes/LaCie/nirspec/data_all/'
save_BASE              = '/Users/dinohsu/nirspec/analysis/'

## priors
priors                 =  { 'teff_min':1290,  'teff_max':1310,
			            	'logg_min':3.6,  'logg_max':3.8,
							'vsini_min':0.0,  'vsini_max':100.0,
							'rv_min':-50.0,    'rv_max':50.0,
							'alpha_min':0.9,  'alpha_max':1.1,
							'A_min':-0.01,     'A_max':0.01,
							'B_min':-0.01,     'B_max':0.01,
							'N_min':0.99,      'N_max':1.01 			}

teff_min_limit		   = 900
teff_max_limit 		   = 1300

## save to catalogue
save_catalougue        = False
catalougue_path        = '/Users/dinohsu/nirspec/catalogues/NIRSPEC_RV_measurements.xlsx'

#########################################

for date_obs in data_dic.keys():
	for sci_data_name in data_dic[date_obs].keys():
		tell_data_name = data_dic[date_obs][sci_data_name]

		print("MCMC for {} on {} order {}".format(sci_data_name, date_obs, order))

		data_path      = data_BASE + date_obs + '/reduced2/fits/all'

		tell_path      = save_BASE + source + '/' \
		+ date_obs +'/telluric_wave_cal/' + tell_data_name + '/O' + str(order)
		
		save_to_path   = save_BASE + source + '/' \
		+ date_obs + '/mcmc/' + sci_data_name + '/O' + str(order) + '/{}_{}_{}'.format(ndim, nwalkers, step)

		if coadd:
			save_to_path   = save_BASE + source + '/' \
			+ date_obs + '/mcmc/' + sci_data_name + '+' + sci_data_name2 + '/O' + str(order) + '/{}_{}_{}'.format(ndim, nwalkers, step)

		save_to_path0  = save_to_path + '/telluric_mcmc'
		save_to_path1  = save_to_path + '/init_mcmc'
		save_to_path2  = save_to_path + '/final_mcmc'

		data        = nsp.Spectrum(name=sci_data_name, order=order, path=data_path, applymask=applymask)
		barycorr    = nsp.barycorr(data.header).value
		## coadd the spectra
		if coadd:
			if not os.path.exists(save_to_path):
				os.makedirs(save_to_path)
			data1       = copy.deepcopy(data)
			data2       = nsp.Spectrum(name=sci_data_name2, order=order, path=data_path, applymask=applymask)
			data.coadd(data2, method='pixel')
			"""
			plt.figure(figsize=(16,6))
			plt.plot(np.arange(1024),data.flux,'k',
				label='coadd median S/N = {}'.format(np.median(data.flux/data.noise)),alpha=1)
			plt.plot(np.arange(1024),data1.flux,'C0',
				label='{} median S/N = {}'.format(sci_data_name,np.median(data1.flux/data1.noise)),alpha=0.5)
			plt.plot(np.arange(1024),data2.flux,'C1',
				label='{} median S/N = {}'.format(sci_data_name2,np.median(data2.flux/data2.noise)),alpha=0.5)
			plt.plot(np.arange(1024),data.noise,'k',alpha=0.5)
			plt.plot(np.arange(1024),data1.noise,'C0',alpha=0.5)
			plt.plot(np.arange(1024),data2.noise,'C1',alpha=0.5)
			plt.legend()
			plt.xlabel('pixel')
			plt.ylabel('cnts/s')
			plt.minorticks_on()
			plt.savefig(save_to_path+'/coadd_spectrum.png')
			#plt.show()
			plt.close()
			"""
		# tell data path
		tell_data_name2 = tell_data_name + '_calibrated'

		tell_sp     = nsp.Spectrum(name=tell_data_name2, order=data.order, path=tell_path, applymask=applymask)

		data.updateWaveSol(tell_sp)
		
		try:
			with fits.open(tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits') as hdulist:
				lsf        = hdulist[0].header['LSF']
		except:
			os.system("python "+BASE+'run_mcmc_telluric.py'+" "+str(order)+" "+date_obs+" "+tell_data_name+" "+tell_path+" "+save_to_path0)

			with fits.open(tell_sp.path + '/' + tell_sp.name + '_' + str(tell_sp.order) + '_all.fits') as hdulist:
				lsf        = hdulist[0].header['LSF']
				

		if not os.path.exists(save_to_path1):
			os.makedirs(save_to_path1)
		
		log_path = save_to_path1 + '/mcmc_parameters.txt'
		file_log = open(log_path,"w+")
		file_log.write("data_path {} \n".format(data.path))
		file_log.write("tell_path {} \n".format(tell_sp.path))
		file_log.write("data_name {} \n".format(data.name))
		file_log.write("tell_name {} \n".format(tell_sp.name))
		file_log.write("order {} \n".format(data.order))
		file_log.write("custom_mask {} \n".format(custom_mask))
		file_log.write("priors {} \n".format(priors))
		file_log.write("ndim {} \n".format(ndim))
		file_log.write("nwalkers {} \n".format(nwalkers))
		file_log.write("step {} \n".format(step))
		file_log.write("burn {} \n".format(burn))
		file_log.write("pixel_start {} \n".format(pixel_start))
		file_log.write("pixel_end {} \n".format(pixel_end))
		file_log.write("barycorr {} \n".format(barycorr))
		file_log.write("lsf {} \n".format(lsf))
		file_log.close()
		
		if applymask:
			if coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)\
					+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2)
			elif not coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask))
		elif not applymask:
			if coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)\
					+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2)
			elif not coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end))
		
		
		
		df = pd.read_csv(save_to_path1+'/mcmc_result.txt', sep=' ', header=None)
		barycorr = nsp.barycorr(data.header).value
		priors1 =  {'teff_min':max(float(df[1][0])-20,teff_min_limit),   'teff_max':min(float(df[1][0])+20,teff_max_limit),
					'logg_min':max(float(df[1][1])-0.001,3.5), 'logg_max':min(float(df[1][1])+0.001,5.5),
					'vsini_min':float(df[1][2])-0.1,           'vsini_max':float(df[1][2])+0.1,
					'rv_min':float(df[1][3])-0.1-barycorr,     'rv_max':float(df[1][3])+0.1-barycorr,
					'alpha_min':float(df[1][4])-0.01,          'alpha_max':float(df[1][4])+0.01,
					'A_min':float(df[1][5])-0.001,             'A_max':float(df[1][5])+0.001,
					'B_min':float(df[1][6])-0.001,             'B_max':float(df[1][6])+0.001,
					'N_min':float(df[1][7])-0.001,             'N_max':float(df[1][7])+0.001  		}

		## mask based on the MCMC parameters with the outlier rejection
		mcmc_dic = {'teff':float(df[1][0]),
					'logg':float(df[1][1]),
					'vsini':float(df[1][2]),
					'rv':float(df[1][3]),
					'alpha':float(df[1][4]),
					'A':float(df[1][5]),
					'B':float(df[1][6]),
					'N':float(df[1][7]),
					'lsf':lsf
					}

		data2 = copy.deepcopy(data)
		data2.wave = np.delete(data2.wave,np.array(custom_mask))
		data2.flux = np.delete(data2.flux,np.array(custom_mask))
		data2.wave = data2.wave[pixel_start: pixel_end]
		data2.flux = data2.flux[pixel_start: pixel_end]

		
		pixel = np.delete(np.arange(len(data2.oriWave)),np.array(custom_mask))[pixel_start: pixel_end]
		#pixel = np.delete(pixel,custom_mask)

		model0 = nsp.makeModel(mcmc_dic['teff'], mcmc_dic['logg'],0,
			mcmc_dic['vsini'], mcmc_dic['rv']-barycorr, mcmc_dic['alpha'], 
			mcmc_dic['B'], mcmc_dic['A'], lsf=mcmc_dic['lsf'], data=data, order=data.order)

		model = nsp.makeModel(mcmc_dic['teff'], mcmc_dic['logg'],0,
			mcmc_dic['vsini'], mcmc_dic['rv']-barycorr, mcmc_dic['alpha'], 
			mcmc_dic['B'], mcmc_dic['A'], lsf=mcmc_dic['lsf'], data=data2, order=data2.order)

		x = np.where(np.abs(data2.flux-model.flux) >= outlier_rejection*np.std(data2.flux-model.flux))
		y = np.where(np.abs(data2.flux-model.flux) <= outlier_rejection*np.std(data2.flux-model.flux))
		y = y[0]
		
		custom_mask2 = pixel[x]
		custom_mask2 = np.append(custom_mask2,np.array(custom_mask))
		custom_mask2.sort()
		custom_mask2 = custom_mask2.tolist()
		print('masking pixels: ',custom_mask2)
		"""
		plt.figure(figsize=(16,6))
		#plt.plot(np.arange(1024),data.flux,'k-',alpha=0.5)
		plt.plot(np.arange(1024),model0.flux,'r-',alpha=0.5)
		plt.plot(pixel[y], data2.flux[y],'b-',alpha=0.5)
		plt.ylabel('cnt/s')
		plt.xlabel('pixel')
		plt.minorticks_on()
		if not os.path.exists(save_to_path2):
			os.makedirs(save_to_path2)
		plt.savefig(save_to_path2+'/spectrum_mask.png')
		plt.show()
		plt.close()
		

		sys.exit()
		"""
		if not os.path.exists(save_to_path2):
			os.makedirs(save_to_path2)

		log_path = save_to_path2 + '/mcmc_parameters.txt'
		file_log = open(log_path,"w+")
		file_log.write("data_path {} \n".format(data.path))
		file_log.write("tell_path {} \n".format(tell_sp.path))
		file_log.write("data_name {} \n".format(data.name))
		file_log.write("tell_name {} \n".format(tell_sp.name))
		file_log.write("order {} \n".format(data.order))
		file_log.write("custom_mask {} \n".format(custom_mask2))
		file_log.write("priors {} \n".format(priors1))
		file_log.write("ndim {} \n".format(ndim2))
		file_log.write("nwalkers {} \n".format(nwalkers2))
		file_log.write("step {} \n".format(step2))
		file_log.write("burn {} \n".format(burn2))
		file_log.write("pixel_start {} \n".format(pixel_start))
		file_log.write("pixel_end {} \n".format(pixel_end))
		file_log.write("barycorr {} \n".format(barycorr))
		file_log.write("lsf {} \n".format(lsf))
		file_log.close()
		
		## Final MCMC
		if applymask:
			if coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)\
					+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2+" -final_mcmc")
			elif not coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)+" -final_mcmc")
		elif not applymask:
			if coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)\
					+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2+" -final_mcmc")
			elif not coadd:
				os.system("python "+BASE+'run_mcmc_science.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
					+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
					+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -final_mcmc")
		
		if save_catalougue:
		## save the result as the excel file
			df2   = pd.read_csv(save_to_path2+'/mcmc_result.txt', sep=' ', header=None)

			rvcat = pd.read_excel(catalougue_path)
			rvcat['fit_rv(km/s)'][rvcat['filename']      == sci_data_name] = round(float(df2[df2[0]=='rv_mcmc'][1]),2)
			rvcat['fit_rv_e(km/s)'][rvcat['filename']    == sci_data_name] = round(float(df2[df2[0]=='rv_mcmc_e'][1]),2)
			rvcat['fit_vsini(km/s)'][rvcat['filename']   == sci_data_name] = round(float(df2[df2[0]=='vsini_mcmc'][1]),2)
			rvcat['fit_vsini_e(km/s)'][rvcat['filename'] == sci_data_name] = round(float(df2[df2[0]=='vsini_mcmc_e'][1]),2)
			rvcat['fit_Teff(K)'][rvcat['filename']       == sci_data_name] = round(float(df2[df2[0]=='teff_mcmc'][1]),0)
			rvcat['fit_Teff_e(K)'][rvcat['filename']     == sci_data_name] = round(float(df2[df2[0]=='teff_mcmc_e'][1]),0)
			rvcat['fit_logg(dex)'][rvcat['filename']     == sci_data_name] = round(float(df2[df2[0]=='logg_mcmc'][1]),2)
			rvcat['fit_logg_e(dex)'][rvcat['filename']   == sci_data_name] = round(float(df2[df2[0]=='logg_mcmc'][1]),2)
			rvcat['tell_filename'][rvcat['filename']     == sci_data_name] = tell_data_name
			rvcat['tell_wave_cal_uncertainty(km/s)'][rvcat['filename'] == sci_data_name] = tell_sp.header['STD']


			rvcat.to_excel(catalougue_path,index=False)
		


