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
### Read in the parameter values from the multinest posteriors 
##############################################################################
prefix = glob.glob( args['workdir'] + '*_.txt' )[0][:-4]
stats_file = open(glob.glob(args['workdir'] + '*_stats.json')[0])
stats_data = json.load(stats_file)
marginals = stats_data['marginals']
n_params = len(free_param_dict.keys())

############## Read in the posterior chains and populate them in a chain dictionary ################
a_obj = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
stats_mn = a_obj.get_stats()
# import pdb
# pdb.set_trace()
try:
    MAP_param_vector = stats_mn["modes"][0]["maximum a posterior"]
except:
    MAP_param_vector = np.zeros(len(free_param_dict.keys()))
    

chain = a_obj.get_equal_weighted_posterior()

chain_dd = {}
for i, pn in enumerate(free_param_dict.keys()):
    chain_dd[pn] = chain[:,i]

############## Construct the fit parameter dictionary with median and ±1 sigma ################
fit_param_dict = {}
for i, pn in enumerate(free_param_dict.keys()):
    fit_param_dict[pn] = [ marginals[i]['median'], marginals[i]['1sigma'][0], marginals[i]['1sigma'][1], MAP_param_vector[i] ]  ### index 0 is -1sigma, and index 1 is +1sigma
np.save(savedir + 'fit_param_dict_models.npy', fit_param_dict)




##############################################################################
### Compute and save a plot of the TP profile 
##############################################################################
print('Computing the TP profile ...')
temp_list, press_list = {}, {} 
## First do for the median 
# posterior_type = posterior_type_list[0] # 'median' or 'MAP' , DOING ONLY ONE AT A TIME FOR NOW! 
if posterior_type == 'median':
    ind = 0
elif posterior_type == '-1sigma':
    ind = 1
elif posterior_type == '+1sigma':
    ind = 2
elif posterior_type == 'MAP':
    ind = 3
    
for pname in fit_param_dict.keys():
    if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**fit_param_dict[pname][ind])
    else:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, fit_param_dict[pname][ind])
        
temp, press = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
temp_list['median'] = temp
press_list['median'] = press

##################################################################################################
################## Randomly sample the chains of the posterior to construct 100 random TP profiles 
####################################################################################################
## Might need to put a check here if the TP params are in the fit param dictionary 
if config_dd['model']['TP_type'] in ['Linear', 'Linear_force_inverted', 'Linear_force_non_inverted']:
    chain_inds = np.random.randint(0, len(chain_dd['T1'])-1, 3000)
    P1_samp = chain_dd['P1'][chain_inds]
    T1_samp = chain_dd['T1'][chain_inds]
    P2_samp = chain_dd['P2'][chain_inds]
    T2_samp = chain_dd['T2'][chain_inds]

    temp_samp, press_samp = [], []
    for ind in chain_inds:
        for pname in ['P1', 'T1', 'P2', 'T2']:
            if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
            else:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])
        temp_, press_ = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
        temp_samp.append(temp_)
        press_samp.append(press_)
    temp_samp, press_samp = np.array(temp_samp), np.array(press_samp)


elif config_dd['model']['TP_type'] == 'Guillot':
    chain_inds = np.random.randint(0, len(chain_dd['T_irr'])-1, 3000)
    T_irr_samp = chain_dd['T_irr'][chain_inds]
    log_gamma_samp = chain_dd['log_gamma'][chain_inds]
    log_kappa_IR_samp = chain_dd['log_kappa_IR'][chain_inds]
    
    temp_samp, press_samp = [], []
    
    for ind in chain_inds:
        for pname in ['T_irr', 'log_gamma', 'log_kappa_IR']:
            if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
            else:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])
        temp_, press_ = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
        temp_samp.append(temp_)
        press_samp.append(press_)
    temp_samp, press_samp = np.array(temp_samp), np.array(press_samp)
    
elif config_dd['model']['TP_type'] == 'Madhusudhan_Seager':
    chain_inds = np.random.randint(0, len(chain_dd['T_set'])-1, 3000)
    # T_set_samp = chain_dd['T_set'][chain_inds]
    # log_gamma_samp = chain_dd['log_gamma'][chain_inds]
    # log_kappa_IR_samp = chain_dd['log_kappa_IR'][chain_inds]
    
    temp_samp, press_samp = [], []
    
    for ind in chain_inds:
        for pname in ['T_set', 'alpha1', 'alpha2', 'log_P1', 'log_P2', 'log_P3']:
            if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
            else:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])
        temp_, press_ = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
        temp_samp.append(temp_)
        press_samp.append(press_)
    temp_samp, press_samp = np.array(temp_samp), np.array(press_samp)
    
elif config_dd['model']['TP_type'] == 'Bezier_4_nodes':
    chain_inds = np.random.randint(0, len(chain_dd['T0'])-1, 3000)
    # T_set_samp = chain_dd['T_set'][chain_inds]
    # log_gamma_samp = chain_dd['log_gamma'][chain_inds]
    # log_kappa_IR_samp = chain_dd['log_kappa_IR'][chain_inds]
    
    temp_samp, press_samp = [], []
    for ind in chain_inds:
        for pname in ['T0', 'T1', 'log_P1', 'T2', 'log_P2', 'T3']:
            if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
            else:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])
        temp_, press_ = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
        temp_samp.append(temp_)
        press_samp.append(press_)
    temp_samp, press_samp = np.array(temp_samp), np.array(press_samp)
    
elif config_dd['model']['TP_type'] == 'Bezier_6_nodes':
    chain_inds = np.random.randint(0, len(chain_dd['T0'])-1, 3000)
    # T_set_samp = chain_dd['T_set'][chain_inds]
    # log_gamma_samp = chain_dd['log_gamma'][chain_inds]
    # log_kappa_IR_samp = chain_dd['log_kappa_IR'][chain_inds]
    
    temp_samp, press_samp = [], []
    for ind in chain_inds:
        # for pname in ["T0", "log_P0", "T1", "log_P1", "T2", "log_P2", "T3", "log_P3", "T4", "log_P4", "T5", "log_P5"]:
        for pname in ["T0", "T1", "log_P1", "T2", "log_P2", "T3", "log_P3", "T4", "log_P4", "T5"]:
            if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
            else:
                setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])
        temp_, press_ = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
        temp_samp.append(temp_)
        press_samp.append(press_)
    temp_samp, press_samp = np.array(temp_samp), np.array(press_samp)

### Compute the +- 1 sigma 
temp_plus_sig, temp_med_sig, temp_min_sig = np.zeros(len(temp_list['median'])), np.zeros(len(temp_list['median'])), np.zeros(len(temp_list['median']))
for it in range(temp_samp.shape[1]):
    temp_array = temp_samp[:,it]
    t_plus, t_med, t_min = corner.quantile(temp_array, [0.84,0.5, 0.16])
    temp_med_sig[it] = t_med
    temp_plus_sig[it] = t_plus
    temp_min_sig[it] = t_min
tp_dict = {}
tp_dict['temp_samp'] = temp_samp
tp_dict['press_samp'] = press_samp
tp_dict['temp_median'] = temp_list['median']
tp_dict['press_median'] = press_list['median']
tp_dict['temp_med_sig'] = temp_med_sig
tp_dict['temp_plus_sig'] = temp_plus_sig
tp_dict['temp_min_sig'] = temp_min_sig
np.save(savedir + 'TP_dict.npy', tp_dict)
print('Done! Plotting them now...')
##############################################
########### Plot the TP profile ##############
##############################################
plt.figure()
for i in range(len(chain_inds)):
    plt.plot(temp_samp[i], press_samp[i], color = 'r', alpha = 0.1, linewidth = 0.3)
# plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
plt.plot(temp_list['median'],press_list['median'], color = 'k' )

plt.ylim(press_list['median'].max(), press_list['median'].min())
# plt.xlim(1600.,3500.)
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
plt.savefig(savedir + 'TP_profile_all_samples.pdf', format='pdf', dpi=300, bbox_inches='tight')

### With the +- 1 sigma bounds  
plt.figure()
plt.fill_betweenx(tp_dict['press_median'], tp_dict['temp_min_sig'], tp_dict['temp_plus_sig'], color = 'r', alpha = 0.2)
# plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
plt.plot(tp_dict['temp_med_sig'],tp_dict['press_median'], color = 'r' )

plt.ylim(press_list['median'].max(), press_list['median'].min())
# plt.xlim(1600.,3500.)
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
plt.savefig(savedir + 'TP_profile_sigma_bounds.pdf', format='pdf', dpi=300, bbox_inches='tight')
print('Done!')
##############################################################################
### Compute, plot, and save the abundance dictionary 
##############################################################################
print('Computing abundances...')
chain_inds = np.random.randint(0, len(chain_dd['Kp'])-1, 3000)

abund_dict_test = planet_model_dict_global[INST_GLOBAL].abundances_dict
abund_dict_save = {}
abund_dict_save['press_median'] = tp_dict['press_median']
### Compile the poeterior sampled profiles first  
for sp in abund_dict_test.keys():    
    abund_dict_save[sp] = {}
    abund_dict_save[sp]['samp'] = np.zeros( (len(chain_inds), len(tp_dict['temp_median'])) )
    abund_dict_save[sp]['abund_med_sig'], abund_dict_save[sp]['abund_plus_sig'], abund_dict_save[sp]['abund_min_sig'] = np.zeros(len(tp_dict['temp_median'])), np.zeros(len(tp_dict['temp_median'])), np.zeros(len(tp_dict['temp_median']))

for indind, ind in enumerate(chain_inds):
    for pname in fit_param_dict.keys():
        if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
            setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**chain_dd[pname][ind])
        else:
            setattr(planet_model_dict_global[INST_GLOBAL], pname, chain_dd[pname][ind])

    abund_dict = planet_model_dict_global[INST_GLOBAL].abundances_dict
    for sp in abund_dict.keys():
        abund_dict_save[sp]['samp'][indind,:] = abund_dict[sp]
        
##### Get the list of species 
SP_INDIV = [x for x in abund_dict.keys() if x != 'press_median']
colors_all = distinctipy.get_colors( len(SP_INDIV), pastel_factor=0.2)
SP_COLORS = {SP_INDIV[i]:colors_all[i] for i in range(len(SP_INDIV))}

### Compute the +-1 sigma bounds
for sp in abund_dict.keys():
    for it in range(abund_dict_save[sp]['samp'].shape[1]):
        abund_array = abund_dict_save[sp]['samp'][:,it]
        a_plus, a_med, a_min = corner.quantile(abund_array, [0.84,0.5, 0.16])
        abund_dict_save[sp]['abund_med_sig'][it] = a_med
        abund_dict_save[sp]['abund_plus_sig'][it] = a_plus
        abund_dict_save[sp]['abund_min_sig'][it] = a_min
np.save(savedir + 'abund_dict.npy', abund_dict_save)
print('Done! Plotting them now...')


plt.figure()
for i_sp, sp in enumerate(abund_dict_save.keys()):
    # if sp not in ['h2', 'he', 'press_median']:
    if sp in SP_INDIV:
        plt.plot( abund_dict_save[sp]['abund_med_sig'], abund_dict_save['press_median'], color = SP_COLORS[sp], label = sp )
        plt.fill_betweenx(abund_dict_save['press_median'], abund_dict_save[sp]['abund_min_sig'], abund_dict_save[sp]['abund_plus_sig'], color = SP_COLORS[sp], alpha = 0.2)
        
plt.ylim(abund_dict_save['press_median'].max(), abund_dict_save['press_median'].min())
plt.yscale('log')
plt.xscale('log')
plt.xlabel('VMR')
plt.ylabel('Pressure [bar]')
plt.legend(fontsize = 8)
plt.savefig(savedir + 'abundances.pdf', format='pdf', dpi=300, bbox_inches='tight')
print('Done!')


##############################################################################
### Compute and save the total best fit model from ALL and individual species 
##############################################################################
model_ind_dd = {}
model_ind_dd['all_species'] = {}
#### Set parameters back to the median values from the posterior
ind = 0 ## For median 
for pname in fit_param_dict.keys():
    if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**fit_param_dict[pname][ind])
    else:
        setattr(planet_model_dict_global[INST_GLOBAL], pname, fit_param_dict[pname][ind])
        
if planet_model_dict_global[INST_GLOBAL].use_stellar_phoenix:
    
    model_wav, model_Fp_orig = planet_model_dict_global[INST_GLOBAL].get_Fp_spectra()    
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
    model_ind_dd['all_species']['wav_nm'], model_ind_dd['all_species']['spec'] = model_wav, model_FpFs
    
else:
    model_wav, model_spec_orig = planet_model_dict_global[INST_GLOBAL].get_spectra()
    # Rotationally broaden the spectrum 
    model_spec_orig_broadened, _ = planet_model_dict_global[INST_GLOBAL].rotation(vsini = planet_model_dict_global[INST_GLOBAL].vsini_planet, 
                                    model_wav = model_wav, model_spec = model_spec_orig)   
    # Convolve the model to the instrument resolution already
    model_spec = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
    model_ind_dd['all_species']['wav_nm'], model_ind_dd['all_species']['spec'] = model_wav, model_spec

np.save(savedir + 'model_spec_dict.npy', model_ind_dd)
plt.figure(figsize = (12,5))
plt.plot(model_ind_dd['all_species']['wav_nm'],
         model_ind_dd['all_species']['spec'], color = 'xkcd:green', linewidth = 0.5 ) 
    
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fp/Fs')
plt.savefig(savedir + 'best_fit_model_all_species.pdf', format='pdf', bbox_inches='tight')


##############################################################################
### Compute the contribution functions and then overplot them with the TP
##############################################################################
contri_dict = {}
contri_func, tau, P_array, P_tau = planet_model_dict_global[INST_GLOBAL].get_contribution_function()
contri_dict['wav_nm'] = model_ind_dd['all_species']['wav_nm']
contri_dict['tau'] = tau
contri_dict['P_array'] = P_array
contri_dict['P_tau'] = P_tau
contri_dict['contri_func'] = contri_func
np.save(savedir + 'contri_func_dict.npy', contri_dict) 

####### Plot the contribution function ################################################

##### First plot the 2D map of the optical depth 
plt.figure(figsize = (16,10))
plt.pcolormesh(model_ind_dd['all_species']['wav_nm'], P_array, np.log10(tau) )
plt.xlabel('Wavelength [nm]')
plt.ylabel('Pressure [bar]')
plt.yscale('log')
plt.ylim(max(P_array),min(P_array))
# plt.xlim(xmin = 2400., xmax = 2410.)
plt.colorbar(label = 'log$_{10}$tau')
plt.savefig(savedir + 'tau_map.png', dpi = 300, format = 'png', bbox_inches = 'tight')

##### Plot the 2D map of the contribution function (without blackbody)
# import pdb
# pdb.set_trace()
plt.figure(figsize = (16,10))
plt.pcolormesh(model_ind_dd['all_species']['wav_nm'], P_array[1:], contri_func.T )
plt.xlabel('Wavelength [nm]')
plt.ylabel('Pressure [bar]')
plt.yscale('log')
plt.ylim(max(P_array[1:]),min(P_array[1:]))
# plt.xlim(xmin = 2400., xmax = 2410.)
plt.colorbar(label = 'CF')
plt.savefig(savedir + 'cf_map.png', dpi = 300, format = 'png', bbox_inches = 'tight')

# Convert to HTML
# html_string = mpld3.fig_to_html(plt.gcf())

# # Save to a file
# with open(savedir + 'tau_map.html', 'w') as f:
#     f.write(html_string)


##### Plot the histogram of the pressure values 
plt.figure(figsize = (12,10))
plt.hist( P_tau, histtype = 'step', bins = 50, density = True, alpha = 1., color = 'k' )
plt.xscale('log')
plt.xlabel('Pressure for tau = 2/3 [bar]')
plt.ylabel('Probability Density')
plt.savefig(savedir + 'contribution_function_hist.pdf', dpi = 300, format = 'pdf', bbox_inches = 'tight')

##### Plot the tau = 2./3. pressure points across the wavelength range 
plt.figure(figsize = (12,10))
plt.plot( model_ind_dd['all_species']['wav_nm'], P_tau, color = 'k' )
plt.yscale('log')
plt.ylim(max(tp_dict['press_median']),min(tp_dict['press_median']))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Pressure for tau = 2/3 [bar]')
plt.savefig(savedir + 'P_tau_2by3_surface.pdf', dpi = 300, format = 'pdf', bbox_inches = 'tight')

####### Plot the TP profile ######### 
## with the histogram of P for tau = 2/3 across all the surfaces 
plt.figure()
plt.fill_betweenx(tp_dict['press_median'], tp_dict['temp_min_sig'], tp_dict['temp_plus_sig'], color = 'r', alpha = 0.2)
# plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
plt.plot(tp_dict['temp_med_sig'],tp_dict['press_median'], color = 'r' )

plt.ylim(max(tp_dict['press_median']), min(tp_dict['press_median']))
plt.xlim(1600.,3500.)
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
ax2 = plt.gca().twiny()
ax2.hist( P_tau, histtype = 'step', bins = 100, density = True, alpha = 1., color = 'k' , orientation = 'horizontal')
ax2.set_xlabel('Probability Density')
plt.savefig(savedir + 'TP_profile_contribution_function_hist.pdf', format='pdf', dpi=300, bbox_inches='tight')

## 2/3 pressures
plt.figure()
plt.fill_betweenx(tp_dict['press_median'],tp_dict['temp_min_sig'], tp_dict['temp_plus_sig'], color = 'r', alpha = 0.2)
# plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
plt.plot(tp_dict['temp_med_sig'],tp_dict['press_median'], color = 'r' )
plt.ylim(max(tp_dict['press_median']), min(tp_dict['press_median']))
plt.xlim(1600.,3500.)
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
ax2 = plt.gca().twiny()
ax2.plot(model_ind_dd['all_species']['wav_nm'], P_tau, color = 'k', alpha = 0.6)
ax2.set_xlabel('Wavelength [nm]')
plt.savefig(savedir + 'TP_profile_tau_2by3_pressures.pdf', format='pdf', dpi=300, bbox_inches='tight')

# exit()


# print('All species: ')
# # for posterior_type in posterior_type_list:
# # print('Computing: ', posterior_type)
# print('Using fast method first ...')
# planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = fit_param_dict, 
#                                     posterior = posterior_type, datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
#             Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = KpVsys_savedir, vel_window = vel_window)

# exit()

################################################################################
################################################################################
# print('Using slow method next ...')   
# KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = fit_param_dict, posterior = posterior_type, datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
#                             Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = KpVsys_savedir)


# planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = posterior_type, theta_fit_dd = fit_param_dict, savedir = KpVsys_savedir)

# #############################################################################
# ## Compute the 2D KpVsys maps and also plot them for individual species
# #############################################################################
# for spnm in SP_INDIV:
#     if spnm not in ['h2', 'he']:
#         print('Excluding:  ', spnm)
#         # spnm_exclude = []
#         # for spnm_ex in SP_INDIV:
#         #     if spnm_ex != spnm:
#         #         spnm_exclude.append(spnm_ex)
        
#         print('Using fast method first ...')
#         planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = fit_param_dict, 
#                                         posterior = posterior_type, datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
#                 Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = KpVsys_savedir, vel_window = vel_window,
#                 exclude_species = [spnm], species_info = spnm)
        
#         print('Using slow method next ...')
#         KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = fit_param_dict, posterior = 'median', datadetrend_dd = datadetrend_dd, 
#                                                                                 order_inds = order_inds, 
#                                     Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = KpVsys_savedir, 
#                                     exclude_species = [spnm], species_info = spnm)
#         planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = 'median', 
#                                                             theta_fit_dd = fit_param_dict, savedir = KpVsys_savedir, species_info = spnm)



