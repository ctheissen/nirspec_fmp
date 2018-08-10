##################################################
# The example to demonstrate the defringe function 
##################################################
## Import the nirspec_pip package
import nirspec_pip as nsp
import os

## Set up the data path
data_folder_path = os.getcwd() + '/data/'

## Run the defringe function
nsp.defringeflatAll(data_folder_path, diagnostic=False)

## You can iterate this process, and the previous 
## flats will be moved to a folder called 
##"defringeflat_diagnostic" under the given path
#nsp.defringeflatAll(data_folder_path, diagnostic=False)