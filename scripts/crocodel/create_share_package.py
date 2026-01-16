#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import astropy.io.fits
import yaml
import sys 
import dynesty
import emcee 
import time
from shutil import copyfile
import corner
import datetime
import os
from multiprocessing import Pool
import pymultinest
from subprocess import call
import json
from astropy.io import ascii as asc
from astropy.table import Table

from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import cross_correlation_utils as crocut
from crocodel.crocodel import data
from crocodel.crocodel import model

import glob
import distinctipy
##############################################################################
### Define species you want to compute the individual cross-correlation maps for 
##############################################################################
# SP_INDIV = ['co', 'h2o']
# colors_all = distinctipy.get_colors( len(SP_INDIV), pastel_factor=0.5 )
# SP_COLORS = {SP_INDIV[i]:colors_all[i] for i in range(len(SP_INDIV))}

# SP_INDIV = ['logZ_planet', 'C_to_O'] #['co', 'h2o'] # , 'oh', 'fe', 'tio']
# SP_COLORS = {
#             # 'co':'black',
#             #  'h2o':'blue',
#             #  'oh':'red',
#             #  'fe':'green',
#             #  'tio':'orange'
#             'logZ_planet':'green',
#             'C_to_O':'orange'
#              }

##############################################################################
### Read in the path to the directory where the outputs of multinest are saved
##############################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-wdir','--workdir', help = "Path to the working directory where all the outputs of the multinest to be post processed are saved.",
                    type=str, required=True)
args = vars(parser.parse_args())
savedir = args['workdir']

##############################################################################
### If needed compute the corner plots etc from multinest samples 
##############################################################################
# infostring = os.path.basename(args['workdir'][:-1])
# call(["python", "multinest_marginals.py", args['workdir'] + infostring + 'multinest_output_'])

## Make a separate directory to load all the KpVsys maps
KpVsys_savedir = savedir + 'KpVsys_maps/' 
try:
    os.makedirs(savedir + 'KpVsys_maps/')
except OSError:
    KpVsys_savedir = savedir + 'KpVsys_maps/'

##############################################################################
### Define the path to the croc_config.yaml 
##############################################################################
config_file_path = args['workdir'] + 'croc_config.yaml'

with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)
# infostring = config_dd['infostring']['info'] + '_N_PCA-' + str(config_dd['infostring']['N_PCA_info']) + '_' + d1

##############################################################################
### Initialize the Data and Model classes: 
##############################################################################
planet_data = data.Data(config = config_file_path)
planet_model_dict = {}
for inst in config_dd['data'].keys():
    planet_model_dict[inst] = model.Model(config = config_file_path, inst = inst )

free_param_dict = config_dd['model']["free_params"]
##############################################################################
### Read the datadetrend dictionary  
##############################################################################
datadetrend_dd = np.load(args['workdir'] + 'datadetrend_dd.npy', allow_pickle = True).item()

##############################################################################
### Define global variables 
##############################################################################
INST_GLOBAL = 'igrins' ## could change this when running for multiple instruments
# INST_GLOBAL = 'crires' ## could change this when running for multiple instruments
# posterior_type_list = ['median'] # ['MAP'] # ['median'] # , '-1sigma', '+1sigma']
posterior_type = 'median'

dates = list(config_dd['data'][INST_GLOBAL]['dates'].keys())
order_inds = planet_data.get_use_order_inds(inst = INST_GLOBAL, date = dates[0])
config_dd_global = config_dd
datadetrend_dd_global = datadetrend_dd
planet_model_dict_global = planet_model_dict
Vsys_range_bound, Vsys_step = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_range'], config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_step']
Vsys_range_bound_trail = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_range_trail']
vel_window = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['vel_window']

Kp_range_bound, Kp_step = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Kp_range'], config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Kp_step']
Vsys_range = np.arange(Vsys_range_bound[0], Vsys_range_bound[1], Vsys_step)
Vsys_range_trail = np.arange(Vsys_range_bound_trail[0], Vsys_range_bound_trail[1], Vsys_step)
Kp_range = np.arange(Kp_range_bound[0], Kp_range_bound[1], Kp_step)

print('Vsys range', Vsys_range, '; Kp range: ', Kp_range)



##############################################################################
### Create a sharing package 
##############################################################################
## Make a separate directory to load all the KpVsys maps
print('Creating the share package ...')
share_pkg_savedir = savedir + 'share_package/' 
try:
    os.makedirs(share_pkg_savedir)
except OSError:
    share_pkg_savedir = share_pkg_savedir
    
####################### Chains from the retrieval ################################
free_param_dict = config_dd['model']["free_params"]
prefix = glob.glob( savedir  + '*_.txt' )[0][:-4]
stats_file = open(glob.glob(savedir  + '*_stats.json')[0])
stats_data = json.load(stats_file)
marginals = stats_data['marginals']
n_params = len(free_param_dict.keys())
a_obj = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
stats_mn = a_obj.get_stats()

fit_params = {}
for p, m in zip(free_param_dict.keys(), stats_mn['marginals']):
    fit_params[p] = {}
    lo, hi = m['1sigma']
    median = m['median']
    lo_sig, hi_sig = median - lo, hi - median
    
    fit_params[p]['median'] = median
    fit_params[p]['-1_sigma_err'] = lo_sig
    fit_params[p]['+1_sigma_err'] = hi_sig
np.save(share_pkg_savedir + 'fit_parameters.npy', fit_params)

####################### Best fit, and median ± 1 sigma uncertainties on the retrieval parameters as txt table ################################
data = Table()
data['parameter'] = np.array([pname for pname in fit_params.keys()])
data['median'] = np.array([fit_params[pname]['median']for pname in fit_params.keys()])
data['-1_sigma_err'] = np.array([fit_params[pname]['-1_sigma_err']for pname in fit_params.keys()])
data['+1_sigma_err'] = np.array([fit_params[pname]['+1_sigma_err']for pname in fit_params.keys()])
asc.write(data, share_pkg_savedir  + 'fit_parameters.csv', overwrite=True, format = 'csv')


#### Extract the chain 
chain = a_obj.get_equal_weighted_posterior()
chain_dd = {}
for i, pn in enumerate(free_param_dict.keys()):
    chain_dd[pn] = chain[:,i]
    
np.save(share_pkg_savedir + 'equal_weighted_posterior_chains.npy', chain_dd)

###################### Best fit model, for all species and individual species ################################
# copyfile(config_file_path, savedir + 'croc_config.yaml')
abund_dict_test = planet_model_dict_global[INST_GLOBAL].abundances_dict
SP_INDIV = [x for x in abund_dict_test.keys() if x != 'press_median']

model_share = {}
sp_names = ['all_species'] + list([x for x in SP_INDIV if x not in ['h2', 'he']])
#### Set parameters back to the median values from the posterior
for pname in fit_params.keys():
    if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**fit_params[pname]['median'])
    else:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, fit_params[pname]['median'])

print(sp_names)
for sp in sp_names:
    print('Computing model for: ', sp)
    plt.figure(figsize= (20,15))
    if sp  == 'all_species':
        exclude_species = None
    else:
        exclude_species = [x for x in sp_names if x not in [sp, 'all_species']]
    
    # if planet_model_dict_global[INST_GLOBAL].use_stellar_phoenix:
    if planet_model_dict_global[INST_GLOBAL].stellar_model is not None:
        
        model_wav, model_Fp_orig = planet_model_dict_global[INST_GLOBAL].get_Fp_spectra(exclude_species = exclude_species)    
        phoenix_model_lsf_broad = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=planet_model_dict_global[INST_GLOBAL].phoenix_model_flux)
        ### Rotationally broaden the planetary spectrum 
        model_Fp_orig_broadened, _ = planet_model_dict_global[INST_GLOBAL].rotation(vsini = planet_model_dict_global[INST_GLOBAL].vsini_planet, 
                                                    model_wav = model_wav, model_spec = model_Fp_orig)
        
        
        model_Fp = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
        plt.figure(figsize = (12,5))
        plt.plot(model_wav,
                model_Fp, color = 'xkcd:green', linewidth = 0.5 ) 
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Fp')
        plt.savefig(savedir + 'best_fit_model_only_Fp.pdf', format='pdf', bbox_inches='tight')

        model_FpFs = model_Fp/phoenix_model_lsf_broad
        model_share['wav_nm'], model_share[sp] = model_wav, model_FpFs
        
    else:
        model_wav, model_spec_orig = planet_model_dict_global[INST_GLOBAL].get_spectra(exclude_species = exclude_species)
        # Rotationally broaden the spectrum 
        model_spec_orig_broadened, _ = planet_model_dict_global[INST_GLOBAL].rotation(vsini = planet_model_dict_global[INST_GLOBAL].vsini_planet, 
                                        model_wav = model_wav, model_spec = model_spec_orig)   
        # Convolve the model to the instrument resolution already
        model_spec = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
    
        model_share['wav_nm'], model_share[sp] = model_wav, model_spec

    plt.plot(model_share['wav_nm'], model_share[sp], color = 'xkcd:green', linewidth = 0.5 ) 
    
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Fp/Fs')
    plt.title(sp)
    plt.savefig(share_pkg_savedir + 'model_'+sp+'.png', format='png', dpi = 300, bbox_inches='tight')
    
np.save(share_pkg_savedir + 'models.npy', model_share)

####################### TP profile ################################
copyfile(savedir + 'TP_dict.npy', share_pkg_savedir + 'TP.npy')
####################### Contribution function ################################
copyfile(savedir + 'contri_func_dict.npy', share_pkg_savedir + 'contribution_function.npy')
####################### Datacubes ################################
copyfile(savedir + 'datadetrend_dd.npy', share_pkg_savedir + 'data.npy')
####################### Abundance dictionary ################################
copyfile(savedir + 'abund_dict.npy', share_pkg_savedir + 'abundances.npy')
