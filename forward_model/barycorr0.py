#!/usr/bin/env python
#
# Feb. 01 2018
# @Dino Hsu
#
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
latitude =  19 + 49.7/60 #degrees
altitude = 4160.

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
	ut = header['DATE-OBS'] + 'T' + header['UTC']
	ra = header['RA']
	dec= header['DEC']
	sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
	barycorr = sc.radial_velocity_correction(obstime=Time(ut), location=keck)
	return barycorr.to(u.km/u.s)
