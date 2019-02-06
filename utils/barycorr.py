#!/usr/bin/env python
#
# History
# Feb. 01 2018 Dino Hsu
# Feb. 03 2019 Chris Theissen
# Feb. 05 2019 Dino Hsu
# The barycentric correction function using Astropy
#

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

# Keck information
# I refer to the telescope information from NASA
# https://idlastro.gsfc.nasa.gov/ftp/pro/astro/observatory.pro
# (longitude - observatory longitude in degrees *west*)
# Need to convert the longitude to the definition in pyasl.helcorr
# obs_long: Longitude of observatory (degrees, **eastern** direction is positive)

longitude = 360 - (155 + 28.7/60 ) # degrees
latitude  =  19 + 49.7/60 #degrees
altitude  = 4160.

keck = EarthLocation.from_geodetic(lat=latitude*u.deg, lon=longitude*u.deg, height=altitude*u.m)
#`~astropy.coordinates.Longitude` or float
#   Earth East longitude.  Can be anything that initialises an
#   `~astropy.coordinates.Angle` object (if float, in degrees)

def barycorr(header):
	"""
	Calculate the barycentric correction using Astropy.
	
	Input:
	header (fits header): using the keywords UT, RA, and DEC

	Output:
	barycentric correction (float*u(km/s))

	"""
	if float(header['DATE-OBS'][0:4]) >= 2019.0: # upgraded NIRSPEC
		ut  = header['DATE-OBS'] + 'T' + header['UT'] 
		ra  = header['RA']
		dec = header['DEC']
		sc  = SkyCoord('%s %s'%(ra, dec), unit=(u.hourangle, u.deg), equinox='J2000', frame='fk5')
	else:
		ut  = header['DATE-OBS'] + 'T' + header['UTC']
		ra  = header['RA']
		dec = header['DEC']
		sc  = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, equinox='J2000', frame='fk5')

	barycorr = sc.radial_velocity_correction(obstime=Time(ut, scale='utc'), location=keck)
	return barycorr.to(u.km/u.s)
