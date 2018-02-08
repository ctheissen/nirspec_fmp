#!/usr/bin/env python
#
# Feb. 7 2017
# @Dino Hsu
#
# Given an input txt file of the data, this function 
# will add the required keywords for reduction via NSDRP

from astropy.io import fits
import os
import warnings
warnings.filterwarnings("ignore")

def add_header(begin, end, date, imagetypes='object', debug=False):
	"""
	Purpose: to add the keywords in the header such as dispers and imagetypes.
	@Dino Hsu

	Inputs:
	begin (int): the begining number of the files, defined by different imagetypes of the data
	end (int): the ending number of the files, defined by different imagetypes of the data
	date (str): the date information in the filename
	imagetypes (str): can only be 'dark, flatlamp', or 'object'


	note:
		1. DISPERS is always 'high', adding to the header all the time
		2. IMAGETYPES: No 'arclamp', NSDRP doesn't deal with the arc lamp data.

	"""
	end = end + 1 #converge to the correct index of files

	for i in range(begin, end):
		files = date + 's0' + format(i, '03d') +'.fits'
		data, header = fits.getdata(files, header=True, ignore_missing_end=True)

		if ('IMAGETYP' in header) is False:
			header['IMAGETYP'] = imagetypes
		if ('DISPERS' in header) is False:
			header['DISPERS'] = 'high'

		#save the changes
		fits.writeto(files, data, header, clobber=True, output_verify='ignore')
		
		#check if the keywords were added to the header correctly
		if debug is True:
			if ('IMAGETYP' in header) is True and ('DISPERS' in header) is True:
				print('The imagetype {0} and the high dispersion are added to the {1}'.format(imagetypes, files))

def addKeyword(file=input, debug=False):
	"""
	Purpose: to read the input file for data reduction
	@Dino Hsu

	Input: 
	Txt format file. This need to specify the data folder
		and the darks, flats, sources

	Output:
	Keywords 'IMAGETYP' and 'DISPERS' required to put in 
		before proccessing by NSDRP

	example:
	from addKeyword import addKeyword
	addKeyword(file='input_reduction.txt')

	"""
	with open(file="input_reduction.txt", mode="r") as f:
		table = f.readlines()
	# cd to the datafolder
	originalpath = os.getcwd()
	datafolder = table[0]
	yrdate = table[1]
	path = datafolder.split('DATAFOLDER\t',1)[1].split('\n',1)[0]\
	+yrdate.split()[1]
	date = table[1].split()[1][2:]
	os.chdir(path)

	# add the keywords
	for i in range(2,len(table)):
		keyword = table[i]
		begin = int(keyword.split()[1].split("-")[0])
		end = int(keyword.split()[1].split("-")[1])
		imagetype = str(keyword.split()[0])
		if imagetype == 'DARKS':
			add_header(begin=begin, end=end, date=date, imagetypes='dark', debug=debug)
		elif imagetype == 'FLATS':
			add_header(begin=begin, end=end, date=date, imagetypes='flatlamp', debug=debug)
		elif imagetype == 'SOURCE':
			add_header(begin=begin, end=end, date=date, imagetypes='object', debug=debug)
	os.chdir(originalpath)
	print('Keywords have been added, ready to be reduced by NSDRP.')







