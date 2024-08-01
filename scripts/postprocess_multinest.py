import argparse
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import astropy.io.fits
import yaml
import sys 
#import arviz as az
#import pymc as pm
#import pytensor
#import pytensor.tensor as pt
import dynesty
import emcee 
import time
from shutil import copyfile
import corner
import cProfile
import datetime
import os
from multiprocessing import Pool
import pymultinest
from subprocess import call
import json
# sys.path.append('/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/sisiaph/')
import stellcorrection_utils as stc
import cross_correlation_utils as croc
import crocodile as crocodile
import glob
import pdb
##############################################################################
### Define species you want to compute the individual cross-correlation maps for 
##############################################################################
# SP_INDIV = ['co', 'h2o', 'oh', 'fe', 'tio']
# SP_COLORS = {'co':'black',
#              'h2o':'blue',
#              'oh':'red',
#              'fe':'green',
#              'tio':'orange'}
SP_INDIV = ['co', 'h2o'] # , 'oh', 'fe', 'tio']
SP_COLORS = {'co':'black',
             'h2o':'blue',
            #  'oh':'red',
            #  'fe':'green',
            #  'tio':'orange'
             }

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
planet_data = crocodile.Data(config = config_file_path)
planet_model_dict = {}
for inst in config_dd['data'].keys():
    planet_model_dict[inst] = crocodile.Model(config = config_file_path, inst = inst )

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
posterior_type_list = ['median'] # ['MAP'] # ['median'] # , '-1sigma', '+1sigma']

dates = list(config_dd['data'][INST_GLOBAL]['dates'].keys())
order_inds = planet_data.get_use_order_inds(inst = INST_GLOBAL, date = dates[0])
config_dd_global = config_dd
datadetrend_dd_global = datadetrend_dd
planet_model_dict_global = planet_model_dict
Vsys_range_bound, Vsys_step = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_range'], config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_step']
Vsys_range_bound_trail = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['Vsys_range_trail']

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
    fit_param_dict[pn] = [ marginals[i]['median'], marginals[i]['1sigma'][0], marginals[i]['1sigma'][1], MAP_param_vector[i] ]

##############################################################################
### Compute and save the best fit model
##############################################################################
## All species 
wav_nm, spec, spec_conv = [], [], []

model_ind_dd = {}
model_ind_dd['all_species'] = {}
## Individual species 
for spnm in SP_INDIV:
    model_ind_dd[spnm] = {}
    model_ind_dd[spnm]['wav'], model_ind_dd[spnm]['spec'], model_ind_dd[spnm]['spec_conv'] = [], [], []
species_list = planet_model_dict_global[INST_GLOBAL].species

for posterior_type in posterior_type_list:
    print('Computing the model spectrum for : ', posterior_type)
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
            
    wav_nm_, spec_ = planet_model_dict_global[INST_GLOBAL].get_spectra()
    spec_conv_= planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=spec_)
    
    wav_nm.append(wav_nm_), spec.append(spec_), spec_conv.append(spec_conv_)
    
    ## Computing models for individual species 
    for spnm in SP_INDIV:
        ## Calculating the model with only contributions from this species 
        # print("Computing forward model for only ", spnm)
        # print("Setting abundance of ", spnm, "to ", str(10.**fit_param_dict[spnm][ind]) )
        setattr(planet_model_dict_global[INST_GLOBAL], spnm, 10.**fit_param_dict[spnm][ind])
        
        ## Set all other species to very low abundances 
        for spnm_ex in SP_INDIV:
            # print("Excluding abundance of ", spnm, "by setting its abundance to ", str(10.**fit_param_dict[spnm][ind]) )
            if spnm_ex != spnm:
                setattr(planet_model_dict_global[INST_GLOBAL], spnm_ex, 10.**-30.)
        
        wav_sp, spec_sp = planet_model_dict_global[INST_GLOBAL].get_spectra()
        spec_conv_sp = planet_model_dict_global[INST_GLOBAL].convolve_spectra_to_instrument_resolution(model_spec_orig=spec_sp)
        
        model_ind_dd[spnm]['wav'].append(wav_sp)
        model_ind_dd[spnm]['spec'].append(spec_sp)
        model_ind_dd[spnm]['spec_conv'].append(spec_conv_sp)
            
wav_nm, spec, spec_conv = np.array(wav_nm), np.array(spec), np.array(spec_conv)
model_ind_dd['all_species']['wav'], model_ind_dd['all_species']['spec'], model_ind_dd['all_species']['spec_conv'] = np.array(wav_nm)[0], np.array(spec)[0], np.array(spec_conv)[0]

for spnm in SP_INDIV:
    model_ind_dd[spnm]['wav'], model_ind_dd[spnm]['spec'], model_ind_dd[spnm]['spec_conv'] = np.array(model_ind_dd[spnm]['wav']), np.array(model_ind_dd[spnm]['spec']), np.array(model_ind_dd[spnm]['spec_conv'])


plt.figure(figsize = (12,5))
plt.plot(wav_nm[0],spec[0], color = 'xkcd:green', label = 'Total', linewidth = 0.7 ) ## The index 0 is just referring to the posterior type here.

for ii, spnm in enumerate(SP_INDIV):
    plt.plot(model_ind_dd[spnm]['wav'][0], model_ind_dd[spnm]['spec'][0]-(ii+1)*0.0002, color = SP_COLORS[spnm], label = spnm, linewidth = 0.7 )
    
plt.xlabel('Wavelength [nm]')
plt.ylabel('Fp/Fs')
plt.legend()
plt.savefig(savedir + 'best_fit_model_all_species.pdf', format='pdf', bbox_inches='tight')

np.save(savedir + 'forward_models.npy', model_ind_dd)

##############################################################################
### Compute and save a plot of the TP profile 
##############################################################################

temp_list, press_list = {}, {} 
## First do for the median 
posterior_type = posterior_type_list[0] # 'median' or 'MAP' , DOING ONLY ONE AT A TIME FOR NOW! 
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
if config_dd['model']['TP_type'] == 'Linear':
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

tp_dict = {}
tp_dict['temp_samp'] = temp_samp
tp_dict['press_samp'] = press_samp
tp_dict['temp_median'] = temp_list['median']
tp_dict['press_median'] = press_list['median']
np.save(savedir + 'TP_dict.npy', tp_dict)

##############################################
########### Plot the TP profile ##############
##############################################
# import pdb
# pdb.set_trace()
plt.figure()

for i in range(len(chain_inds)):
    plt.plot(temp_samp[i], press_samp[i], color = 'r', alpha = 0.1, linewidth = 0.3)
# plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
plt.plot(temp_list['median'],press_list['median'], color = 'k' )

plt.ylim(press_list['median'].max(), press_list['median'].min())
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [bar]')
plt.savefig(savedir + 'TP_profile.pdf', format='pdf', dpi=300, bbox_inches='tight')
# exit()
##############################################################################
### Compute the 2D KpVsys maps and also plot them for all species 
##############################################################################
# print('All species: ')
for posterior_type in posterior_type_list:
    print('Computing: ', posterior_type)
    KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = fit_param_dict, posterior = posterior_type, datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                                Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = KpVsys_savedir)
    print('Plotting: ', posterior_type)
    planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = posterior_type, theta_fit_dd = fit_param_dict, savedir = KpVsys_savedir)

##############################################################################
### Compute the 2D KpVsys maps and also plot them for all species 
##############################################################################
for spnm in SP_INDIV:
    print('Only ', spnm)
    
    spnm_exclude = []
    for spnm_ex in SP_INDIV:
        if spnm_ex != spnm:
            spnm_exclude.append(spnm_ex)
        
    KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = fit_param_dict, posterior = 'median', datadetrend_dd = datadetrend_dd, 
                                                                              order_inds = order_inds, 
                                Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = KpVsys_savedir, 
                                exclude_species = spnm_exclude, species_info = spnm)
    planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = 'median', 
                                                           theta_fit_dd = fit_param_dict, savedir = KpVsys_savedir, species_info = spnm)

    