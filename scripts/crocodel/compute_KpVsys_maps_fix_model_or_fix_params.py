#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
import datetime
import os
import astropy.io.ascii as asc
import distinctipy
# os.environ["OMP_NUM_THREADS"] = "1"

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
# sys.path.append('/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/sisiaph/')
from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import astro_utils as aut
from crocodel.crocodel import data
from crocodel.crocodel import model
##########################################################################################################################
##########################################################################################################################


  
##########################################################################################################################
##########################################################################################################################


parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=True)
parser.add_argument('-dt','--date_tag', help = "(Optional) Date tag of a previous run on which you want to run parts of the script.",
                    type=str, required=False)
parser.add_argument('-mmet','--map_calc_method', help = "Method used to calculate the map, 'fast' or 'slow'. Fast is done without model reprocessing using the fast method, and slow is done with model reprocessing.",
                    type=str, required=True)
parser.add_argument('-inst','--instrument', help = "Name of the instrument.",
                    type=str, required=True)

args = vars(parser.parse_args())
config_file_path = args['config_file_path']
KpVsys_method = args['map_calc_method']
INST_GLOBAL = args['instrument'] ## can be either igrins, crires_plus, or spirou
# INST_GLOBAL = 'igrins' ## could change this when running for multiple instruments in future implementation.
# INST_GLOBAL = 'crires_plus' ## could change this when running for multiple instruments in future implementation.

if args['date_tag'] is None:
    now = datetime.datetime.now()
    # Format the date and time
    d1 = now.strftime("%d-%m-%YT%H-%M-%S")
    print('Date tag for this run: ', d1)
else:
    d1 = args['date_tag']

with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)
    
infostring = config_dd['infostring']['info'] + '_N_PCA-' + str(config_dd['infostring']['N_PCA_info']) + '_' + d1

savedir = config_dd['workdir']['KpVsys_maps_fixed_params'] + infostring + '/'

try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

print('Saving files in directory: ', savedir)

### Save the config file in the savedir 
copyfile(config_file_path, savedir + 'croc_config.yaml')

############################################################################################################################################################################
############################################################################################################################################################################
############################################# Initialize the Data and Model classes: #######################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
####### Do you have a fixed model computed already? If this is set true, all the model params in the config file will be ignored and only this model will be used.
fix_model_info = config_dd['model']['fix_model_info']
fix_model = fix_model_info['fix_model'] ## True or False 


planet_data = data.Data(config = config_file_path)
planet_model_dict = {}
for inst in config_dd['data'].keys():
    planet_model_dict[inst] = model.Model(config = config_file_path, inst = inst )

if not fix_model:
    free_param_dict = config_dd['model']["free_params"]
    fix_param_dict = {}
    if free_param_dict != 'None':
        for pname in free_param_dict.keys():
            fix_param_dict[pname] = free_param_dict[pname]["fix_test"]

############################################################################################################################################################################
############################################################################################################################################################################
## Do the PCA detrending for the data just once as it doesn't need to be repeated for ######################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
datadetrend_dd = {}
dates = list(config_dd['data'][INST_GLOBAL]['dates'].keys())
for date in dates:
    datadetrend_dd[date] = {}
order_inds = planet_data.get_use_order_inds(inst = INST_GLOBAL, date = dates[0])# [0,1,2] # Keep this for now, assuming we are using same number of detectors for all dates. 
######################################################################################
######################################################################################
# Do following for each date and order and save , do only for one instrument for now
######################################################################################
######################################################################################

if not os.path.isfile(savedir+'datadetrend_dd.npy'):
    print('Detrending data (including those for skip order inds for consistency)...') 

    for date in dates:
        datadetrend_dd[date]['phases'] = planet_data.get_spdatacubes_dict[INST_GLOBAL][date]['phases']
        datadetrend_dd[date]['berv'] = planet_data.get_spdatacubes_dict[INST_GLOBAL][date]['bary_RV']
        print('Date: ', date)
        print('phases: ', datadetrend_dd[date]['phases'])
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
        
        
        datadetrend_dd[date]['post_pca_mask'] = np.array(post_pca_mask) # This will work if all orders are of the same length 
        datadetrend_dd[date]['colmask'] = np.array(colmask)
        datadetrend_dd[date]['data_wavsoln'] = np.array(data_wavsoln)
        datadetrend_dd[date]['datacube'] = np.array(datacube)
        datadetrend_dd[date]['datacube_fit'] = np.array(datacube_fit)
        datadetrend_dd[date]['datacube_detrended'] = np.array(datacube_detrended)
        datadetrend_dd[date]['datacube_mean_sub'] = np.array(datacube_mean_sub)
        datadetrend_dd[date]['pca_eigenvectors'] = pca_eigenvectors
    ## Save the datadetrend_dd 
    np.save(savedir+'datadetrend_dd.npy', datadetrend_dd)

    print('Computed datadetrend_dd!')
    print('Plotting detrended datacubes...')
    ######################################################################################
    ######################################################################################
    ## Plot and save the original and detrended datacubes 
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
            hnd2 = stc.subplot_datacube(axis=axx[1], datacube = datadetrend_dd[date]['datacube_mean_sub'][ind, :, :], 
                                        phases=datadetrend_dd[date]['phases'], 
                            wavsoln= datadetrend_dd[date]['data_wavsoln'][ind, :],
                            title='Detrended \n Date: ' + date + 'Detector: ' + str(ind), 
                            setxlabel=True,
                        vminvmax=[-0.08,0.08])
            
            fig.colorbar(hnd2, ax=axx[1])

            plt.savefig(savedir + 'datacubes_original_detrend_date-' + date + '_det-'+ str(ind) + '.pdf', 
                        format='pdf', bbox_inches='tight')
            plt.close()
    print('Done!')

else:
    print('datadetrend_dd has already been computed, using that...')
    datadetrend_dd = np.load(savedir+'datadetrend_dd.npy', allow_pickle = True).item()

######################################################################################
######################################################################################
######################################################################################
######################################################################################
##### Define global variables ######################################################################################
######################################################################################
######################################################################################
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
try:
    phase_range = config_dd_global['data'][INST_GLOBAL]['cross_correlation_params']['phase_range']
except:
    phase_range = None ### will summ across all phases in the data.

############################################################################################################################################################################
############################################################################################################################################################################
############ Compute the KpVsys maps ########################################################################################################################################## 
############################################################################################################################################################################
############################################################################################################################################################################

##############################################################################################
## Computing the total model and models for individual species ###################################################
##############################################################################################
if fix_model: #### the model should already have been calculated before using a separate script.
    fix_model_path = fix_model_info['fix_model_path']
    model_dd = np.load(fix_model_path + 'spec_dict.npy', allow_pickle = True).item()
    
    if config_dd_global['data'][INST_GLOBAL]['method'] == 'emission':
        if planet_model_dict_global[INST_GLOBAL].use_stellar_phoenix:   
            wav_nm, spec = model_dd['wav_nm'], model_dd['Fp']
        else:
            wav_nm, spec = model_dd['wav_nm'], model_dd['spec']
    
    elif config_dd_global['data'][INST_GLOBAL]['method'] == 'transmission':
        wav_nm, spec = model_dd['wav_nm'], model_dd['spec']
        
    #### For now just copy the abund_dict and TP_dict files; in future can just use them to comput ethe model here instead of precomputing elsewhere.
    copyfile(fix_model_info['fix_model_path'] + 'TP_dict.npy', savedir + 'TP_dict.npy')
    copyfile(fix_model_info['fix_model_path'] + 'abund_dict.npy', savedir + 'abund_dict.npy')
    copyfile(fix_model_info['fix_model_path'] + 'spec_dict.npy', savedir + 'spec_dict.npy')
    copyfile(fix_model_info['fix_model_path'] + 'contri_func_dict.npy', savedir + 'contri_dict.npy')
    
    TP_dict = np.load(fix_model_info['fix_model_path'] + 'TP_dict.npy', allow_pickle = True).item()
    abund_dict = np.load(fix_model_info['fix_model_path'] + 'abund_dict.npy', allow_pickle = True).item()
    
    SP_INDIV = [x for x in abund_dict.keys() if x != 'press_median']
    colors_all = distinctipy.get_colors( len(SP_INDIV), pastel_factor=0.2)
    SP_COLORS = {SP_INDIV[i]:colors_all[i] for i in range(len(SP_INDIV))}
    ####################### OLD ################################################# 
    wav_nm, spec = planet_model_dict_global[INST_GLOBAL].get_spectra()
    abund_dict = planet_model_dict_global[INST_GLOBAL].abundances_dict
    SP_INDIV = [x for x in config_dd_global['model']['abundances'].keys() if x != 'he']
    colors_all = distinctipy.get_colors( len(SP_INDIV), pastel_factor=0.5 )
    SP_COLORS = {SP_INDIV[i]:colors_all[i] for i in range(len(SP_INDIV))}

        
    ########### Plot the model only ########### 
    plt.figure(figsize = (12,5))
    plt.plot(wav_nm, spec, color = 'xkcd:green', linewidth = 0.7 )

    # for ii, spnm in enumerate(SP_INDIV):
    #     plt.plot(model_ind_dd[spnm]['wav'][0], model_ind_dd[spnm]['spec'][0]-(ii+1)*0.0002, color = SP_COLORS[spnm], label = spnm, linewidth = 0.7 )
    # plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Fp/Fs')
    plt.savefig(savedir + 'best_fit_model_all_species.pdf', format='pdf', bbox_inches='tight')

    np.save(savedir + 'wav_nm.npy', wav_nm)
    np.save(savedir + 'spec.npy', -spec)
        
else:
    model_ind_dd = {}
    model_ind_dd['all_species'] = {}
    
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
spec_plot = 1. - model_ind_dd['all_species']['spec']
plt.plot(model_ind_dd['all_species']['wav_nm'],
         spec_plot, color = 'xkcd:green', linewidth = 0.5 ) 
    
plt.xlabel('Wavelength [nm]')
if config_dd_global['data'][INST_GLOBAL]['method'] == 'transmission':
    plt.ylabel('(Rp/Rs)^2')
    plt.ylim(ymin = 0.02, ymax = 0.025)
    plt.xlim(xmin = 1060, xmax = 1080)
else:
    plt.ylabel('Fp/Fs')

plt.savefig(savedir + 'best_fit_model_all_species.pdf', format='pdf', bbox_inches='tight')
exit()

# ############ Plot the model and the data across all orders ########### 
# plt.figure(figsize = (12,5))
# # spnm = 'fe' ## Just plot one of the species 
# # plt.plot(model_ind_dd[spnm]['wav'][0], model_ind_dd[spnm]['spec'][0]-(ii+1)*0.0002, color = SP_COLORS[spnm], label = spnm, linewidth = 0.7 )
# plt.plot(wav_nm, spec, color = 'r', linewidth = 0.7 )
# dk = list(datadetrend_dd.keys())[0]
# for iord in range(datadetrend_dd[dk]['datacube_mean_sub'].shape[0]):
#     data_order_time_av = np.mean(datadetrend_dd[dk]['datacube_mean_sub'][iord,:,:], axis = 0)
#     plt.plot(datadetrend_dd[dk]['data_wavsoln'][iord,:], data_order_time_av, color = 'xkcd:azure')
# plt.savefig(savedir + 'data_and_model_all_orders_average.png', format='png', dpi=300, bbox_inches='tight')
# plt.close('all')


##########################################################################################
################### Compute and plot the TP profile and abundance profiles  ######################################
##########################################################################################
if fix_model:
    plt.figure()
    plt.plot(TP_dict['temp_median'], TP_dict['press_median'], color = 'k' )
    plt.ylim( max(TP_dict['press_median']), min(TP_dict['press_median']) )
    plt.yscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [bar]')
    plt.savefig(savedir + 'TP_profile.png', format='png', dpi=300, bbox_inches='tight')
    
    plt.figure()
    for i_sp, sp in enumerate(abund_dict.keys()):
        # if sp not in ['h2', 'he', 'press_median']:
        if sp in SP_INDIV:
            plt.plot( abund_dict[sp]['abund_med_sig'], abund_dict['press_median'], color = SP_COLORS[sp], label = sp )
            # plt.fill_betweenx(abund_dict['press_median'], abund_dict[sp]['abund_min_sig'], abund_dict[sp]['abund_plus_sig'], color = SP_COLORS[sp], alpha = 0.2)
        
    plt.ylim(abund_dict['press_median'].max(), abund_dict['press_median'].min())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('VMR')
    plt.ylabel('Pressure [bar]')
    plt.legend(fontsize = 8)
    plt.savefig(savedir + 'abundances.pdf', format='pdf', dpi=300, bbox_inches='tight')

else:
    ############ Compute and plot the TP profile ############## 
    temp, press = planet_model_dict_global[INST_GLOBAL].get_TP_profile()
    tp_dict = {}
    tp_dict['temp_median'] = temp
    tp_dict['press_median'] = press
    np.save(savedir + 'TP_dict.npy', tp_dict)
    plt.figure()
    plt.plot(temp,press, color = 'k' )
    plt.ylim(press.max(), press.min())
    plt.yscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [bar]')
    plt.savefig(savedir + 'TP_profile.png', format='png', dpi=300, bbox_inches='tight')
    
    ############ Compute and plot the abundances ############## 
    abund_dict_test = planet_model_dict_global[INST_GLOBAL].abundances_dict
    abund_dict_save = {}
    abund_dict_save['press_median'] = tp_dict['press_median']
    SP_INDIV = [x for x in abund_dict_test.keys()]
    for sp in abund_dict_test.keys():
        abund_dict_save[sp] = {}
        abund_dict_save[sp]['abund_med_sig'] = abund_dict_test[sp]
    np.save(savedir + 'abund_dict.npy', abund_dict_save)
    
    colors_all = distinctipy.get_colors( len(SP_INDIV), pastel_factor=0.2)
    SP_COLORS = {SP_INDIV[i]:colors_all[i] for i in range(len(SP_INDIV))}
    
    plt.figure()
    for i_sp, sp in enumerate(abund_dict_save.keys()):
        # if sp not in ['h2', 'he', 'press_median']:
        if sp in SP_INDIV:
            plt.plot( abund_dict_save[sp]['abund_med_sig'], abund_dict_save['press_median'], color = SP_COLORS[sp], label = sp )
            # plt.fill_betweenx(abund_dict_save['press_median'], abund_dict_save[sp]['abund_min_sig'], abund_dict_save[sp]['abund_plus_sig'], color = SP_COLORS[sp], alpha = 0.2)
            
    plt.ylim(abund_dict_save['press_median'].max(), abund_dict_save['press_median'].min())
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('VMR')
    plt.ylabel('Pressure [bar]')
    plt.legend(fontsize = 8)
    plt.savefig(savedir + 'abundances.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print('Done!')
        
    ##############################################################################
### Compute the contribution functions and then overplot them with the TP
    ##############################################################################
    # contri_dict = {}
    # contri_func, tau, P_array, P_tau = planet_model_dict_global[INST_GLOBAL].get_contribution_function()
    # contri_dict['wav_nm'] = model_ind_dd['all_species']['wav_nm']
    # contri_dict['tau'] = tau
    # contri_dict['P_array'] = P_array
    # contri_dict['P_tau'] = P_tau
    # contri_dict['contri_func'] = contri_func
    # np.save(savedir + 'contri_func_dict.npy', contri_dict) 

    ####### Plot the contribution function ################################################

    ##### First plot the 2D map of the optical depth 
    # plt.figure(figsize = (16,10))
    # plt.pcolormesh(model_ind_dd['all_species']['wav_nm'], P_array, np.log10(tau) )
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Pressure [bar]')
    # plt.yscale('log')
    # plt.ylim(max(P_array),min(P_array))
    # # plt.xlim(xmin = 2400., xmax = 2410.)
    # plt.colorbar(label = 'log$_{10}$tau')
    # plt.savefig(savedir + 'tau_map.png', dpi = 300, format = 'png', bbox_inches = 'tight')

    ##### Plot the 2D map of the contribution function (without blackbody)
    # plt.figure(figsize = (16,10))
    # plt.pcolormesh(model_ind_dd['all_species']['wav_nm'], P_array[1:], contri_func.T )
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Pressure [bar]')
    # plt.yscale('log')
    # plt.ylim(max(P_array[1:]),min(P_array[1:]))
    # # plt.xlim(xmin = 2400., xmax = 2410.)
    # plt.colorbar(label = 'CF')
    # plt.savefig(savedir + 'cf_map.png', dpi = 300, format = 'png', bbox_inches = 'tight')

    # ##### Plot the histogram of the pressure values 
    # plt.figure(figsize = (12,10))
    # plt.hist( P_tau, histtype = 'step', bins = 50, density = True, alpha = 1., color = 'k' )
    # plt.xscale('log')
    # plt.xlabel('Pressure for tau = 2/3 [bar]')
    # plt.ylabel('Probability Density')
    # plt.savefig(savedir + 'contribution_function_hist.pdf', dpi = 300, format = 'pdf', bbox_inches = 'tight')

    # ##### Plot the tau = 2./3. pressure points across the wavelength range 
    # plt.figure(figsize = (12,10))
    # plt.plot( model_ind_dd['all_species']['wav_nm'], P_tau, color = 'k' )
    # plt.yscale('log')
    # plt.ylim(max(tp_dict['press_median']),min(tp_dict['press_median']))
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Pressure for tau = 2/3 [bar]')
    # plt.savefig(savedir + 'P_tau_2by3_surface.pdf', dpi = 300, format = 'pdf', bbox_inches = 'tight')

    ####### Plot the TP profile ######### 
    ## with the histogram of P for tau = 2/3 across all the surfaces 
    plt.figure()
    # plt.fill_betweenx(tp_dict['press_median'], tp_dict['temp_min_sig'], tp_dict['temp_plus_sig'], color = 'r', alpha = 0.2)
    # plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
    plt.plot(tp_dict['temp_median'],tp_dict['press_median'], color = 'r' )

    plt.ylim(max(tp_dict['press_median']), min(tp_dict['press_median']))
    plt.xlim(1600.,3500.)
    plt.yscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [bar]')
    ax2 = plt.gca().twiny()
    # ax2.hist( P_tau, histtype = 'step', bins = 100, density = True, alpha = 1., color = 'k' , orientation = 'horizontal')
    ax2.set_xlabel('Probability Density')
    plt.savefig(savedir + 'TP_profile_contribution_function_hist.pdf', format='pdf', dpi=300, bbox_inches='tight')

    ## 2/3 pressures
    plt.figure()
    # plt.fill_betweenx(tp_dict['press_median'],tp_dict['temp_min_sig'], tp_dict['temp_plus_sig'], color = 'r', alpha = 0.2)
    # plt.fill_betweenx(press_list[0], temp_list[1], temp_list[2], color = 'r', alpha = 0.2)
    plt.plot(tp_dict['temp_median'],tp_dict['press_median'], color = 'r' )
    plt.ylim(max(tp_dict['press_median']), min(tp_dict['press_median']))
    plt.xlim(1600.,3500.)
    plt.yscale('log')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [bar]')
    ax2 = plt.gca().twiny()
    # ax2.plot(model_ind_dd['all_species']['wav_nm'], P_tau, color = 'k', alpha = 0.6)
    ax2.set_xlabel('Wavelength [nm]')
    plt.savefig(savedir + 'TP_profile_tau_2by3_pressures.pdf', format='pdf', dpi=300, bbox_inches='tight')

##############################################################################
### Compute the 2D KpVsys maps and also plot them for all species included
##############################################################################

###### Make sure everything is set to initial params 
if not fix_model:
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    ###### ###### Computing KpVsys maps for fixed model parameters ###### ###### ###### ######
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    print('Computing KpVsys maps for fixed parameters ...')
    print(fix_param_dict.keys())
    for pname in fix_param_dict.keys():
        if pname in planet_model_dict_global[INST_GLOBAL].species or pname in ['P1','P2']:
            print(pname, fix_param_dict[pname])
            setattr(planet_model_dict_global[INST_GLOBAL], pname, 10.**fix_param_dict[pname])
        else:
            setattr(planet_model_dict_global[INST_GLOBAL], pname, fix_param_dict[pname])   

    print('Computing KpVsys maps ...')
    if KpVsys_method == 'slow':
        print('Using fast method as test first ...')
        planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = None, posterior = None, 
                                                    datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                        Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = savedir, vel_window = vel_window, phase_range = phase_range)
        
        print('Using slow method ...')
        print('Computing the trail matrix with model reprocessing ...')
        # planet_model_dict_global[INST_GLOBAL].get_ccf_trail_matrix_with_model_reprocess(datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
        #                      Vsys_range = Vsys_range, savedir = savedir, Kp_range = Kp_range)
        # exit()
        KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = None, posterior = '_', 
                                                                                    datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                                    Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = savedir, phase_range = phase_range)

        planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = '_', theta_fit_dd = None, savedir = savedir)
    elif KpVsys_method == 'fast':
        print('Using fast method ...')
        planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = None, posterior = None, 
                                                            datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                                Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = savedir, vel_window = vel_window, phase_range = phase_range)

else:
    print('Computing KpVsys maps for fixed model ...')
    if KpVsys_method == 'slow':
        print('Using fast method as test first ...')
        planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = None, posterior = None, 
                                                    datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                        Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = savedir, vel_window = vel_window, fixed_model_wav = wav_nm, fixed_model_spec = spec)
        
        print('Using slow method ...')
        KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = None, posterior = '_', 
                                                                                    datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                                    Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = savedir, fixed_model_wav = wav_nm, fixed_model_spec = spec)

        planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = None, posterior = '_', theta_fit_dd = None, savedir = savedir)
    
    elif KpVsys_method == 'fast':
        print('Using fast method ...')
        planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map_fast_without_model_reprocess(theta_fit_dd = None, posterior = None, 
                                                            datadetrend_dd = datadetrend_dd, order_inds = order_inds, 
                                Vsys_range = Vsys_range_trail, Kp_range = Kp_range, savedir = savedir, vel_window = vel_window, fixed_model_wav = wav_nm, fixed_model_spec = spec)
    
       
       
##############################################################################
### Compute the 2D KpVsys maps and also plot them for all species individually ; following only implemented for slow method right now.
##############################################################################     
# for spnm in SP_INDIV:
#     print('Only ', spnm)
#     print('Setting ', spnm, 'to ', 10.**fix_param_dict[spnm])
#     setattr(planet_model_dict_global[INST_GLOBAL], spnm, 10.**fix_param_dict[spnm])
    
#     spnm_exclude = []
#     for spnm_ex in SP_INDIV:
#         if spnm_ex != spnm:
#             spnm_exclude.append(spnm_ex)
#             setattr(planet_model_dict_global[INST_GLOBAL], spnm_ex, 10.**-30.)
#     ### Zero out abundance params for spnm_ex
    
    
#     KpVsys_save = planet_model_dict_global[INST_GLOBAL].compute_2D_KpVsys_map(theta_fit_dd = None, posterior = '_', datadetrend_dd = datadetrend_dd, 
#                                                                               order_inds = order_inds, 
#                                 Vsys_range = Vsys_range, Kp_range = Kp_range, savedir = savedir, 
#                                 exclude_species = spnm_exclude, species_info = spnm)
#     planet_model_dict_global[INST_GLOBAL].plot_KpVsys_maps(KpVsys_save = KpVsys_save, posterior = '_', 
#                                                            theta_fit_dd = None, savedir = savedir, species_info = spnm)


