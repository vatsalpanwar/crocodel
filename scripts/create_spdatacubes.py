"""Parse fits files to spdatacubes dictionaries that can be used by crocodel. 
Support for: CRIRES (old), IGRINS, CARMENES.
"""

import astropy
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.time import Time
import argparse 
import numpy as np
import sys

from crocodel.crocodel import astro_utils as aut
from crocodel.crocodel import data_parsing_utils as dput 

parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-df','--data_file_path', help = "Path to the data file containing the reduced spectra.",
                    type=str, required=True)
parser.add_argument('-inst','--instrument_name', help = "Name of the instrument.",
                    type=str, required=True)
parser.add_argument('-pln','--planet_name', help = "Name of the planet.",
                    type=str, required=True)
parser.add_argument('-svdir','--save_directory', help = "Directory where you want to save the spdatacube dictionaries.",
                    type=str, required=True)

## Below arguments for manually specifying T0 and Porb. Required for IGRINS as NASA Exo Archive query may not return the latest ephemerides. 
parser.add_argument('-T0','--T_mid_transit', help = "Predicted Mid transit time, latest for calculation of phase. (preferably in BJD_TDB)",
                    type=str, required=False)
parser.add_argument('-Porb','--P_orbital', help = "Orbital period (days)",
                    type=str, required=False)
parser.add_argument('-infstr','--infostring', help = "Information string.",
                    type=str, required=False)

args = vars(parser.parse_args())
data_file_path = args['data_file_path']
instrument_name = args['instrument_name']
planet_name = args['planet_name']
savedir = args['save_directory']

if instrument_name == 'crires':
    data = astropy.io.fits.getdata(data_file_path)
    spdatacube_dd = {}
    
    # Parse the datacubes and other auxiliary data from the files
    spdatacube_dd['spdatacube'] = data['SPEC'][0][:,:,:]
    spdatacube_dd['airmass'] = data['AIRM'][0]
    spdatacube_dd['phases'] = data['PH'][0]
    spdatacube_dd['wavsoln'] = data['WLEN'][0][:,0,:]
    # bary_RV = data['RVEL'][0]
    spdatacube_dd['time'] = data['MJD'][0]
    
    datestr = Time(spdatacube_dd['time'], format = 'mjd')[0].isot.split("T")[0]
    
    planet = NasaExoplanetArchive.query_criteria(table="pscomppars", where="pl_name like '" + planet_name + "'")
    
    ra, dec = planet['rastr'], planet['decstr']
    
    spdatacube_dd['bary_RV'] = -1. * np.array( aut.get_BERV(time_array = spdatacube_dd['time'] , time_format = 'mjd',
                ra_dec = [ra, dec],
                obs_location = [-24.617,-70.4,2635]) )   ## hardcode the instrument location here. CRIRES is at Paranal.
    
    np.save(savedir + datestr + '_spdd.npy', spdatacube_dd)
    
elif instrument_name == 'igrins': ## Needs testing, it is probably better to have a config file for creating spdatacubes as well instead of taking all inputs at command line.
    T0 = float(args['T0'])
    Porb = float(args['Porb'])
    infostring = args['infostring']
    
    infostring_final = planet_name + '_' +infostring
    dput.get_data_BJD_phase_Vbary_igrins(datadir = data_file_path,  T0 = T0, Porb = Porb, 
                                        save = True, savedir = savedir, infostring = infostring_final, retres = False)

