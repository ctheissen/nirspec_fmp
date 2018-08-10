####################################################
# The example to demonstrate the telluric wavelength
# calibration. The calibrated spectra will be save
# to the folder "tell_wave_cal" under the path
# 'data/reduced/fits/all'
####################################################
## Import the nirspec_pip package
import nirspec_pip as nsp
import os

## Set up the input paramters
order_list = [33]
data_path = os.getcwd() + '/data/reduced/fits/all'
data_name = 'jan19s0024' # Filename of the telluric standard
save_to_path = os.getcwd() + \
'/data/reduced/fits/all/tell_wave_cal/' + data_name
test = False # Generate the diagnostic plots
save = True # Save the wavelength calibrated spectra

## Run the telluric wavelength calibration
nsp.run_wave_cal(data_name ,data_path ,order_list ,
	save_to_path, test=test, save=save)