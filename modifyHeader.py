#!/usr/bin/env python
#
# Oct. 11 2017
# @Dino Hsu

from astropy.io import fits

# This code is to modify the headers to fit the NSDRP pipeline. 

#### The parameters need to be set up here
# Note this path is set before the running 3 digits
# ex: if the data filename is nov16s0001.fits, the path
#     to set up is 'nov16s'
path = '' 


# Note that the NSDRP does not have the arclamp correction
add_header(1, 11, imagetypes='dark')
add_header(12, 22, imagetypes='flatlamp')
add_header(24, 39, imagetypes='object')

######

def add_header(begin, end, imagetypes='object'):
	"""
	The function is aimed to add the keywords in the header such as dispers and imagetypes.
	begin: the begining number of the files, defined by different imagetypes of the data
	end: the ending number of the files, defined by different imagetypes of the data
	mode:
		1. DISPERS is always 'high', adding to the header all the time
		2. IMAGETYPES can be 'object', 'flatlamp', or 'dark' (we don't add the 'arc' here)
			, since NSDRP doesn't deal with the arc lamp data.

	"""
	end = end + 1 #converge to the correct index of files

	for i in range(begin, end):
		files = path + format(i, '03d') +'.fits'
		data, header = fits.getdata(files, header=True, ignore_missing_end=True)

		if ('IMAGETYP' in header) is False:
			header['IMAGETYP'] = imagetypes
		if ('DISPERS' in header) is False:
			header['DISPERS'] = 'high'

		#save the changes
		fits.writeto(files, data, header, clobber=True, output_verify='ignore')
		
		#check if the keywords were added to the header correctly
		if ('IMAGETYP' in header) is True and ('DISPERS' in header) is True:
			print('The imagetype {0} and the dispers are added to the {1}'.format(imagetypes, files))





