import nirspec_fmp as nsp

#########################################
## Parameter Setup
#########################################

data_dic           = {
'20160216':['feb16s0076','feb16s0077','feb16s0078','feb16s0079'],
}

source             = 'J1106+2754'
order_list         = [33,34]
test               = False
save               = True
applymask          = False

data_BASE          = '/Volumes/LaCie/nirspec/data_all/'
save_BASE          = '/Users/dinohsu/nirspec/analysis/'

for date_obs in data_dic.keys():
	for data_name in data_dic[date_obs]:
		data_path  = data_BASE + date_obs + '/reduced/fits/all'
	
		save_to_path = save_BASE + source +'/'\
		+ date_obs + '/telluric_wave_cal/' + data_name
		print(save_to_path)
		print("Telluric wavelength calibration on", data_name)
		nsp.run_wave_cal(data_name ,data_path ,order_list,
			save_to_path, test=test, save=save, applymask=applymask)