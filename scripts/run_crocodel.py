#!/usr/bin/env python3
"""Script to run a MultiNest sampling for high-resolution cross-correlation spectroscopy data.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from shutil import copyfile
import datetime
import os
import pymultinest
import json

# Matplotlib rcparams
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Add the path to the code snippet modules
from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import data
from crocodel.crocodel import model
##########################################################################################################################
##########################################################################################################################
### Define the global name for the instrument (should match the instrument name specified in the croc_config.yaml file)
INST_GLOBAL = 'igrins' ## could change this when running for multiple instruments in future implementation.
##########################################################################################################################
##########################################################################################################################

now = datetime.datetime.now()
# Format the date and time
d1 = now.strftime("%d-%m-%YT%H-%M-%S")
print('Date tag for this run which will be used to save the results: ', d1)

################################################################
################################################################
"""
Take the path to the croc_config.yaml file as the input
"""
################################################################
################################################################
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=True)
args = vars(parser.parse_args())
config_file_path = args['config_file_path']
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)

################################################################
################################################################
"""
Extract the information string (infostring) which will be used in names of all results files 
"""
################################################################
################################################################
infostring = config_dd['infostring']['info'] + '_N_PCA-' + str(config_dd['infostring']['N_PCA_info']) + '_' + d1
savedir = config_dd['workdir']['results'] + infostring + '/'

"""Create the directory to save results."""
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

print('Saving files in directory: ', savedir)

### Save the config file in the savedir 
copyfile(config_file_path, savedir + 'croc_config.yaml')

################################################################
################################################################
"""
Initialize the Data and Model classes
"""
################################################################
################################################################
planet_data = data.Data(config = config_file_path)
planet_model_dict = {} ## While not implemented yet, the future plan is to have a model class for each instrument separately. Hence need a dictionary to store them all.
for inst in config_dd['data'].keys():
    planet_model_dict[inst] = model.Model(config = config_file_path, inst = inst )

"""Extract information on number of live points"""
N_live_points = config_dd['retrieval_setup']["N_live"]

"""Setup the free parameter dictionary from the information available in the croc_config.yaml file"""
free_param_dict = config_dd['model']["free_params"]

"""Setup the fix parameter dictionary"""
fix_param_dict = {}
for pname in free_param_dict.keys():
    fix_param_dict[pname] = free_param_dict[pname]["fix_test"]


################################################################
################################################################
"""
Do the PCA detrending for the data just once as it doesn't need to be repeated
"""
################################################################
################################################################
datadetrend_dd = {}
dates = list(config_dd['data'][INST_GLOBAL]['dates'].keys())
for date in dates:
    datadetrend_dd[date] = {}
order_inds = planet_data.get_use_order_inds(inst = INST_GLOBAL, date = dates[0]) ## Assuming that same number of orders are used for all dates.

"""
Loop over dates and orders/detectors, save results in datadetrend_dd dictionary
"""

print('Detrending data (including those for skipped orders for consistency)...') 
for date in dates:
    datadetrend_dd[date]['phases'] = planet_data.get_spdatacubes_dict[INST_GLOBAL][date]['phases']
    datadetrend_dd[date]['berv'] = planet_data.get_spdatacubes_dict[INST_GLOBAL][date]['bary_RV']
    
    post_pca_mask, colmask, data_wavsoln, datacube, datacube_mean_sub, datacube_fit, datacube_detrended, pca_eigenvectors = [],[],[],[],[],[],[],[]
    
    for ind in range(config_dd['data'][INST_GLOBAL]['N_order_all']):
        colmask_, post_pca_mask_, data_wavsoln_, datacube_, datacube_fit_, datacube_detrended_, datacube_mean_sub_, pca_eigenvectors_ = planet_data.pca_per_order_fast(inst = INST_GLOBAL, date = date, order_ind = ind)

        post_pca_mask.append(post_pca_mask_)
        colmask.append(colmask_) ## colmask is the same for all orders for now so don't append it more than once 
        data_wavsoln.append(data_wavsoln_)
        datacube.append(datacube_)
        datacube_fit.append(datacube_fit_)
        datacube_detrended.append(datacube_detrended_)
        datacube_mean_sub.append(datacube_mean_sub_) 
        pca_eigenvectors.append(pca_eigenvectors_)
    
    
    datadetrend_dd[date]['post_pca_mask'] = np.array(post_pca_mask) # Feel free to change this manually if you have a custom post_pca_mask (should be of shape (N_orders, N_wave))
    datadetrend_dd[date]['colmask'] = np.array(colmask)
    datadetrend_dd[date]['data_wavsoln'] = np.array(data_wavsoln)
    datadetrend_dd[date]['datacube'] = np.array(datacube)
    datadetrend_dd[date]['datacube_fit'] = np.array(datacube_fit)
    datadetrend_dd[date]['datacube_detrended'] = np.array(datacube_detrended)
    datadetrend_dd[date]['datacube_mean_sub'] = np.array(datacube_mean_sub)
    datadetrend_dd[date]['pca_eigenvectors'] = pca_eigenvectors

## Save the datadetrend_dd 
np.save(savedir+'datadetrend_dd.npy', datadetrend_dd)
print('Done!')

################################################################
################################################################
"""
Plot the detrended datacubes
"""
################################################################
################################################################
print('Plotting detrended datacubes...')
## Plot and save the original and detrended datacubes ; plot only those that will be eventually used for analysis
for date in datadetrend_dd.keys():
    for ind in order_inds:
        print('Date: ', date, 'Det: ', ind)
        fig, axx = plt.subplots(2, 1, figsize=(12, 5*2))
        plt.subplots_adjust(hspace=0.8)
        
        ## First plot the original datacube 
        hnd1 = stc.subplot_datacube(axis=axx[0], datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                    phases=datadetrend_dd[date]['phases'], 
                        wavsoln= datadetrend_dd[date]['data_wavsoln'][ind, :],
                        title='Original \n Date: ' + date + 'Detector: ' + str(ind), 
                        setxlabel=True,
                    vminvmax=None)
        
        fig.colorbar(hnd1, ax=axx[0])
        ## Plot the detrended datacube  
        hnd2 = stc.subplot_datacube(axis=axx[1], datacube = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                    phases=datadetrend_dd[date]['phases'], 
                        wavsoln= datadetrend_dd[date]['data_wavsoln'][ind, :],
                        title='Detrended \n Date: ' + date + 'Detector: ' + str(ind), 
                        setxlabel=True,
                    vminvmax=[-0.015,0.015])
        
        fig.colorbar(hnd2, ax=axx[1])

        plt.savefig(savedir + 'datacubes_original_detrend_date-' + date + '_det-'+ str(ind) + '.pdf', 
                    format='pdf', bbox_inches='tight')
        plt.close()
print('Done!')

################################################################
################################################################
"""
Define some global variables for running multinest 
"""
################################################################
################################################################
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

################################################################
################################################################
""" Set all free parameters to the fix parameters """
print(fix_param_dict.keys())
for pname in fix_param_dict.keys():
    if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
        print(pname, fix_param_dict[pname])
        setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**fix_param_dict[pname])
    else:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, fix_param_dict[pname])  
################################################################
################################################################

##### Uncomment below if you want to compute the trail matrix 
# print('Computing the trail matrix for initial parameters of the model class')
# planet_model_dict_global[INST_GLOBAL].get_ccf_trail_matrix(datadetrend_dd = datadetrend_dd_global, 
#                                                     order_inds = order_inds, 
#                              Vsys_range = Vsys_range_trail, plot = True, savedir = savedir)
# print('Done!')

################################################################
################################################################
"""
Plot some initial things like the TP profile and the corresponding forward model
"""
################################################################
################################################################
temp, press = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
plt.figure()
plt.plot(temp, press, color = 'k' )

plt.ylim(press.max(), press.min())
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
plt.savefig(savedir + 'init_TP_profile.pdf', format='pdf', dpi=300, bbox_inches='tight')

init_model_dd = {}
wav, spec = planet_model_dict_global[INST_GLOBAL].get_spectra()
init_model_dd['wav'], init_model_dd['spec'] = wav, spec
init_model_dd['spec_conv'] = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=spec)
        
plt.figure(figsize = (12,5))
plt.plot(wav, spec, 
         color = 'xkcd:green', label = 'Total', linewidth = 0.7 )
    
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fp/Fs')
plt.legend()
plt.savefig(savedir + 'init_model_all_species.pdf', format='pdf', bbox_inches='tight')
np.save(savedir + 'init_forward_models.npy', init_model_dd)


################################################################
################################################################
"""
Compute a fast Kp-Vsys map for the fix_test values in the free_param_dict
"""
################################################################
################################################################
print('Computing KpVsys maps ')
planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = None, posterior = None, 
                                                           datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                             Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = savedir, vel_window = vel_window)



################################################################
################################################################
"""
Define Likelihood and prior functions
"""
################################################################
################################################################
def prior_transform(cube, ndim, nparams):
    ### Just do uniform priors for all parameters within their bounds given in self.bounds
    """
    u is a list of sampled points for each parameter in the range from 0. to 1. This must be transformed to the bounds for your own parameter
    ### Assuming u is in the same order as the parameters in the free_param_dict
    """

    for i, nm in enumerate(free_param_dict.keys()):
        cube[i] = free_param_dict[nm]["bound"][0] + (free_param_dict[nm]["bound"][1] - free_param_dict[nm]["bound"][0])*cube[i]

def log_likelihood_multinest(cube, ndim, nparams):
    inst = INST_GLOBAL
    try:
        logL_total = planet_model_dict_global[inst].logL_fast(cube, datadetrend_dd = datadetrend_dd_global,
                                                                          order_inds=order_inds)
    except AssertionError:
        logL_total = -1e100 # -np.inf
    # print(logL_total)
    return logL_total

################################################################
################################################################
"""
Run the MultiNest sampler
"""
################################################################
################################################################

ndim = len(free_param_dict.keys())
parameters = list(free_param_dict.keys())
globalStart = time.time()
pymultinest.run(log_likelihood_multinest, 
                prior_transform, ndim, 
                outputfiles_basename=savedir + infostring + 'multinest_output_', 
                resume = False, verbose = True, n_live_points = N_live_points, 
                multimodal = False, importance_nested_sampling = False)
json.dump(parameters, open(savedir + infostring + 'multinest_output_' + 'params.json', 'w')) # save parameter names
globalEnd = time.time()
print('Total computation took {:5} seconds'.format(globalEnd-globalStart))