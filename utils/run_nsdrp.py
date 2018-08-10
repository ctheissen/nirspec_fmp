# rountine for reducing the NIRSPEC data
# using NSDRP

import nirspec_pip as nsp
from astropy.io import fits
import os
import warnings
from subprocess import call
import subprocess
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Reduce the private data using NIRSPEC-Data-Reduction-Pipeline",\
	usage="run_nsdrp input_dir output_dir")

parser.add_argument("-f","--files",\
	dest="files",default=None,help="input_dir",nargs="+",required=True)

args = parser.parse_args()
datadir = args.files

if len(datadir) is 1:
	save_to_path = datadir[0] + '/reduced'
	datadir.append(save_to_path)

originalpath = os.getcwd()
path = originalpath + '/' + datadir[0] + '/'

# store the fits file names
mylist = os.listdir(path)

for filename in mylist:
    file_path = path + filename
    data, header = fits.getdata(file_path, header=True, ignore_missing_end=True)
    if ('IMAGETYP' in header) is False:
        if ('flat lamp off     ' in str(header['COMMENT'])) is True:
            header['IMAGETYP'] = 'dark'
        elif ('flat field' in str(header['COMMENT'])) is True:
        	header['IMAGETYP'] = 'flatlamp'
        elif ('NeArXeKr' in str(header['COMMENT'])) is True :
        	header['IMAGETYP'] = 'arclamp'
        else:
        	header['IMAGETYP'] = 'object'
    if ('DISPERS' in header) is False:
        header['DISPERS'] = 'high'

    fits.writeto(file_path, data, header, overwrite=True, output_verify='ignore')
print("Keywords are added to the data. Ready to process by NSDRP.")

# reduce the data using NSDRP
os.system("python /Users/dinohsu/projects/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py"\
	+ " " + datadir[0] + " " +datadir[1] + " " \
	+ "-oh_filename /Users/dinohsu/projects/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat\
	 -spatial_jump_override -debug")

## Verbose output in the terminal
#os.system("python /Users/dinohsu/projects/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py"\
#	+ " " + datadir[0] + " " + datadir[0] +"/"+datadir[1] + " " \
#	+ "-oh_filename /Users/dinohsu/projects/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat\
#	 -spatial_jump_override -verbose -debug")

print("The NIRSPEC data are reduced successfully by using NSDRP.")
