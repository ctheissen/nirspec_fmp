import os
import pandas as pd
from astropy.io import ascii, fits
import warnings
warnings.filterwarnings("ignore")

def makeTargetList(path, **kwargs):
	"""
	Return a csv file to store the KOA data information of sources.
	@Dino Hsu

	Parameters
	----------
	path      : str
	            The path of the data.

	outdir    : str
	            The path where the csv file will be stored.


	Returns
	-------
	targetlist: csv file format
			    including the header keywords
	            'OBJECT', 
	            'FILENAME', 
	            'SLITNAME', 
	            'DATE-OBS',
	            'PROGPI'

	Examples
	--------
	>>> makeTargetList(path='path/to/PI_name')

	"""

	outdir = kwargs.get('outdir',path)

	# crear empty lists to store keywords
	OBJECT_list   = []
	FILENAME_list = []
	SLITNAME_list = []
	DATEOBS_list  = []
	PROGPI_list   = []

	os.chdir(path)
	listDataDir = os.listdir('.')

	for folder in listDataDir:
		path = folder + '/raw/spec/'
		null = ''
		try:
			filenames = os.listdir(path)
			for filename in filenames:
				fullpath = path + filename
				try:
					with fits.open(fullpath, ignore_missing_end=True) as f:
						if f[0].header['IMAGETYP'] == 'object':
							if f[0].header['OBJECT'] is not null:
								OBJECT_list.append(f[0].header['OBJECT'])
								FILENAME_list.append(f[0].header['FILENAME'])
								SLITNAME_list.append(f[0].header['SLITNAME'])
								DATEOBS_list.append(f[0].header['DATE-OBS'])
								PROGPI_list.append(f[0].header['PROGPI'])
							else:
								pass
						else:
							pass
				except IOError:
					pass
				except IsADirectoryError:
					pass
				except NotADirectoryError:
					pass
				except FileNotFoundError:
					pass
		except NotADirectoryError:
			pass
		except FileNotFoundError:
			pass

	df = pd.DataFrame({"OBJECT" : OBJECT_list, "FILENAME" : FILENAME_list, "SLITNAME":SLITNAME_list, "DATE-OBS":DATEOBS_list, "PROGPI":PROGPI_list})
	save_to_path = outdir + 'targetlist.csv'
	df.to_csv(save_to_path, index=False)
	print("The target list is saved to {} .".format(save_to_path))