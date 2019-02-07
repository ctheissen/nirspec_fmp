###############################################################################################
#																							  #
# Forward-Modeling routine using multiprocessing emcee package (Foreman-Mackey et al. 2013)   #
#																							  #
###############################################################################################

import nirspec_fmp as nsp
import pandas as pd
import numpy as np
from astropy.io import fits
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import sys
import os
import os.path

#########################################
## Parameters Set Up
#########################################

## 20030720 N3 sci J2356-1553 112-115 tell 119-122
## 20051210 N3 sci J2356-1553 1-4 16 17 tell 8-15

data_dic = {
'J2356-1553':{
'20030720':{'20jus0112':'20jus0120','20jus0113':'20jus0120','20jus0114':'20jus0120','20jus0115':'20jus0120'},
'20051210':{'dec10s0001':'dec10s0008','dec10s0002':'dec10s0009','dec10s0003':'dec10s0009','dec10s0004':'dec10s0008','dec10s0016':'dec10s0008','dec10s0017':'dec10s0009'}}
}

#source                 = 'J2356-1553'
order                  = 57
ndim, nwalkers, step   = 7, 50, 400
burn                   = 300 # mcmc sampler chain burn
moves                  = 2.0
ndim2, nwalkers2, step2= 7, 50, 400
burn2                  = 300
applymask              = False
pixel_start, pixel_end = 10, -40
alpha_tell             = 1.0 # telluric alpha for LSF
plot_show              = False
custom_mask            = []#[ 674, 675, 676, 680, 681, 682, 707, 708, 709, 710, 713, 714, 715]
outlier_rejection      = 2.5

## coadd
coadd                  = False
sci_data_name2         = None

modelset               = 'btsettl08'
BASE                   = '/home/chh194/code/nirspec_fmp/nirspec_fmp/forward_model/'
data_BASE              = '/home/chh194/data/nirspec_all/'
save_BASE              = '/home/chh194/analysis/'

## priors
priors                 =  { #'teff_min':1090,  'teff_max':1110,
							'teff_min':800,		'teff_max':1200,
							'logg_min':4.0,		'logg_max':5.0,
							'vsini_min':0.0,	'vsini_max':70.0,
							'rv_min':-50.0,		'rv_max':50.0,
							'alpha_min':0.9,	'alpha_max':1.1,
							'A_min':-0.01,		'A_max':0.01,
							'N_min':0.99,		'N_max':1.01 			}

teff_min_limit		   = 800
teff_max_limit 		   = 1300

## save to catalogue
save_catalougue        = False
#catalougue_path        = '/Users/dinohsu/nirspec/catalogues/NIRSPEC_RV_measurements.xlsx'

#########################################

for source in data_dic.keys():
	for date_obs in data_dic[source].keys():
		for sci_data_name in data_dic[source][date_obs].keys():
			tell_data_name = data_dic[source][date_obs][sci_data_name]

			print("MCMC for {} on {} {} order {}".format(source, date_obs, sci_data_name, order))

			data_path      = data_BASE + date_obs + '/reduced/fits/all'

			tell_path      = save_BASE + source + '/' \
			+ date_obs +'/telluric_wave_cal/' + tell_data_name + '/O' + str(order)
		
			save_to_path   = save_BASE + source + '/' \
			+ date_obs + '/mcmc/' + sci_data_name + '/O' + str(order) + '/{}_{}_{}'.format(ndim, nwalkers, step)

			catalougue_path= save_BASE + source + '/{}_fit_table.xlsx'.format(source)

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
			
			# calculate mean snr
			med_snr  = np.median(data.flux/data.noise)

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
			file_log.write("med_snr {} \n".format(med_snr))
			file_log.close()
		
			if applymask:
				if coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)\
						+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2)
				elif not coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask))
			elif not applymask:
				if coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim)+" -nwalkers "+str(nwalkers)+" -step "+str(step)+" -burn "+str(burn)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)\
						+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2)
				elif not coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
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
						#'B_min':float(df[1][6])-0.001,             'B_max':float(df[1][6])+0.001,
						'N_min':float(df[1][6])-0.001,             'N_max':float(df[1][6])+0.001  		}

			## mask based on the MCMC parameters with the outlier rejection
			mcmc_dic = {'teff':float(df[1][0]),
						'logg':float(df[1][1]),
						'vsini':float(df[1][2]),
						'rv':float(df[1][3]),
						'alpha':float(df[1][4]),
						'A':float(df[1][5]),
						#'B':float(df[1][6]),
						'N':float(df[1][6]),
						'lsf':lsf
						}

			data2 = copy.deepcopy(data)
			data2.wave = data2.wave[pixel_start: pixel_end]
			data2.flux = data2.flux[pixel_start: pixel_end]

			model = nsp.makeModel(mcmc_dic['teff'], mcmc_dic['logg'],0,
				mcmc_dic['vsini'], mcmc_dic['rv']-barycorr, mcmc_dic['alpha'], 
				0, mcmc_dic['A'], lsf=mcmc_dic['lsf'], data=data, order=data.order)
			pixel = np.delete(np.arange(len(data2.oriWave)),data2.mask)[pixel_start: pixel_end]
			custom_mask2 = pixel[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) > outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))]
			"""
			plt.figure(figsize=(16,6))
			plt.plot(np.arange(1024),data.flux,'k-',alpha=0.5)
			plt.plot(np.arange(1024),model.flux,'r-',alpha=0.5)
			plt.plot(pixel[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) < outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))],
				data2.flux[np.where(np.abs(data2.flux-model.flux[pixel_start: pixel_end]) < outlier_rejection*np.std(data2.flux-model.flux[pixel_start: pixel_end]))],'b-',alpha=0.5)
			plt.ylabel('cnt/s')
			plt.xlabel('pixel')
			plt.minorticks_on()
			if not os.path.exists(save_to_path2):
				os.makedirs(save_to_path2)
			plt.savefig(save_to_path2+'/spectrum_mask.png')
			#plt.show()
			plt.close()
			"""

			custom_mask2 = np.append(custom_mask2,np.array(custom_mask))
			custom_mask2.sort()
			custom_mask2 = custom_mask2.tolist()
			print('masking pixels: ',custom_mask2)

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
			file_log.write("med_snr {} \n".format(med_snr))
			file_log.close()

			## Final MCMC
			if applymask:
				if coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)\
						+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2+" -final_mcmc")
				elif not coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -applymask "+str(applymask)+" -final_mcmc")
			elif not applymask:
				if coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)\
						+" -coadd "+str(coadd)+" -coadd_sp_name "+sci_data_name2+" -final_mcmc")
				elif not coadd:
					os.system("python "+BASE+'run_mcmc_science_no_wave_offset.py'+" "+str(order)+" "+date_obs+" "+sci_data_name+" "+tell_data_name+" "+data_path+" "+tell_path+" "+save_to_path+" "+str(lsf)\
						+" -outlier_rejection "+str(outlier_rejection)+" -ndim "+str(ndim2)+" -nwalkers "+str(nwalkers2)+" -step "+str(step2)+" -burn "+str(burn2)+" -moves "\
						+str(moves)+" -pixel_start "+str(pixel_start)+" -pixel_end "+str(pixel_end)+" -final_mcmc")
		
			if save_catalougue:
			## save the result as the excel file
				df2   = pd.read_csv(save_to_path2+'/mcmc_result.txt', sep=' ', header=None)
				try:
					rvcat = pd.read_excel(catalougue_path)
				except:
					rvcat = pd.DataFrame(columns=['source','date_obs','sci_filename','tell_filename','order',
						'rv','e_rv','vsini','e_vsini','teff','e_teff','logg','e_logg','snr','wave_cal_err'])

				rv           = round(float(df2[df2[0]=='rv_mcmc'][1]),1)
				e_rv         = round(float(df2[df2[0]=='rv_mcmc_e'][1]),1)
				vsini        = round(float(df2[df2[0]=='vsini_mcmc'][1]),1)
				e_vsini      = round(float(df2[df2[0]=='vsini_mcmc_e'][1]),1)
				teff         = round(float(df2[df2[0]=='teff_mcmc'][1]),0)
				e_teff       = round(float(df2[df2[0]=='teff_mcmc_e'][1]),0)
				logg         = round(float(df2[df2[0]=='logg_mcmc'][1]),1)
				e_logg       = round(float(df2[df2[0]=='logg_mcmc_e'][1]),1)
				snr          = round(med_snr,1)
				wave_cal_err = tell_sp.header['STD']

				try:
					rvcat['order'][rvcat['sci_filename']         == sci_data_name] = order
					rvcat['rv'][rvcat['sci_filename']            == sci_data_name] = rv
					rvcat['e_rv'][rvcat['sci_filename']          == sci_data_name] = e_rv
					rvcat['vsini'][rvcat['sci_filename']         == sci_data_name] = vsini
					rvcat['e_vsini'][rvcat['sci_filename']       == sci_data_name] = e_vsini
					rvcat['teff'][rvcat['sci_filename']          == sci_data_name] = teff
					rvcat['e_teff'][rvcat['sci_filename']        == sci_data_name] = e_teff
					rvcat['logg'][rvcat['sci_filename']          == sci_data_name] = logg
					rvcat['e_logg'][rvcat['sci_filename']        == sci_data_name] = e_logg
					rvcat['snr'][rvcat['sci_filename']           == sci_data_name] = snr
					rvcat['tell_filename'][rvcat['sci_filename'] == sci_data_name] = tell_data_name
					rvcat['wave_cal_err'][rvcat['sci_filename']  == sci_data_name] = wave_cal_err

				except:
					rvcat.append({'source':source,'date_obs':date_obs,'sci_filename':sci_data_name,
						'tell_filename':tell_data_name,'order':order,
						'rv':rv,'e_rv':e_rv,'vsini':vsini,'e_vsini':e_vsini,
						'teff':teff,'e_teff':e_teff,'logg':logg,'e_logg':e_logg,'snr':snr,
						'wave_cal_err':wave_cal_err}, ignore_index=True)

				rvcat.to_excel(catalougue_path,index=False)
