'''
Compute a single total Kp-Vsys for a set of parameters and a given injected model. Takes simulate_config.yaml as an input and also includes stellar and telluric signals (inherent in ITC outputs), and performs a PCA before recovering the signal through cross-correlation.

'''
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import matplotlib.cm as cm
import astropy.io.ascii as asc
import argparse
import yaml
import datetime
import os
import sys
import time
import json
import re

### Set some matplotlib rc params for easy plotting later on 
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
import matplotlib.pyplot as plt
from shutil import copyfile

sys.path.insert(0,"/Users/parkerl/Documents/crocodel")
from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import cross_correlation_utils as crocut
#from crocodel.crocodel import astro_utils as aut
from crocodel.planner import planit

######## Read in the config file ###### 
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the simulate_config.yaml.",
                    type=str, required=True)
args = vars(parser.parse_args())
config_file_path = args['config_file_path']
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)

match = re.search(r'/([^/]+)_b/', config_file_path)
if match: target = match.group(1)

print(target)
########################################

instrument = config_dd['instrument']
#HD179949 51_peg HD75289 HD217107 tau_boo

if instrument == 'crires': ## This is for checking and making sure the simulation is done for igrins. Best option for another instrument is to just create another script instead of adding another 'else' condition here in order to keep the script short and manageable.
    ######### Define the path and load the ITC outputs 

    crires_itc_output = config_dd['snr_info']['output_path']
    etc_override = False
    if not os.path.exists(f'{crires_itc_output}OUTPUT_{target}_crires_etc.json') or etc_override:
        print('Calling ETC...')
        #### I have to run the ETC here.
        os.system(f'python etc_cli.py crires {crires_itc_output}{target}_crires_etc_input.json -o  {crires_itc_output}OUTPUT_{target}_crires_etc.json')
        #### THIS IS THE ETC INPUT SECTION. H_wav, K_Wav, etc need to be the CRIREES+ intputs
        print('COMPLETED')
    

    file_path = f'{crires_itc_output}OUTPUT_{target}_crires_etc.json'
    with open(file_path, 'r') as f:
        etc_output = json.load(f)   # not json.loads

    print(np.array(etc_output['data']['orders']).shape[0])
    snr_data = np.zeros((3, int(np.array(etc_output['data']['orders']).shape[0]), 2048))
    wave_data = np.zeros((3, int(np.array(etc_output['data']['orders']).shape[0]), 2048))
    star_data = np.zeros((3, int(np.array(etc_output['data']['orders']).shape[0]), 2048))

    for ord in range(np.array(etc_output['data']['orders']).shape[0]):
        
        for det in range(3):
            snr_data[det, ord] = etc_output['data']['orders'][ord]['detectors'][det]['noise_components']['snr']
            star_data[det, ord] = etc_output['data']['orders'][ord]['detectors'][det]['noise_components']['target']
            wave_data[det, ord] = etc_output['data']['orders'][ord]['detectors'][det]['wavelength']


    """
    H_sig = asc.read(crires_itc_output + 'H_signal.txt')
    H_sig_bkg = asc.read(crires_itc_output + 'H_signal_bkg.txt')
    H_snr_1exp = asc.read(crires_itc_output + 'H_SNR_single_exp.txt')
    H_snr_fin = asc.read(crires_itc_output + 'H_SNR_final.txt')

    K_sig = asc.read(crires_itc_output + 'K_signal.txt')
    K_sig_bkg = asc.read(crires_itc_output + 'K_signal_bkg.txt')
    K_snr_1exp = asc.read(crires_itc_output + 'K_SNR_single_exp.txt')
    K_snr_fin = asc.read(crires_itc_output + 'K_SNR_final.txt')

    H_wav, K_wav = H_sig['col1'], K_sig['col1']
    """



    
    ############## Extract more info from the file ###### 
    T_exp = config_dd['snr_info']['T_exp']
    overhead_per_exp = config_dd['snr_info']['overhead_per_exp']
    
    #only for transit
    #transit_duration = config_dd['planet_info']['transit_duration'] ## hours 
    
    planet_params = {}
    planet_params["P_orb"] = config_dd['planet_info']['P_orb'] ## days 
    #planet_params["RpRs"] = config_dd['planet_info']['RpRs'] ## days 
    #planet_params["a_Rs"] = config_dd['planet_info']['a_Rs'] ## days 
    planet_params["inc"] = config_dd['planet_info']['inc'] ## days 
    planet_params["ecc"] = config_dd['planet_info']['ecc'] ## days 
    planet_params["w"] = config_dd['planet_info']['w'] ## days 
    
    phase_range = config_dd['planet_info']['phase_range']
    Kp_true = config_dd['planet_info']['Kp_true']
    Vsys_true = config_dd['planet_info']['Vsys_true']
    
    Kp_range = np.arange(config_dd['planet_info']['Kp_range'][0], config_dd['planet_info']['Kp_range'][1], config_dd['planet_info']['Kp_step'] )
    Vsys_range = np.arange(config_dd['planet_info']['Vsys_range'][0], config_dd['planet_info']['Vsys_range'][1], config_dd['planet_info']['Vsys_step'] )
    vel_window = config_dd['planet_info']['vel_window']
    
    method = config_dd['planet_info']['method']
    
    infostring = config_dd['simulation_info']['infostring']
    model_infostring = config_dd['planet_info']['inject_model_infostring'] + config_dd['planet_info']['cross_correlate_model_infostring']
    N_PCA, N_visit = config_dd['simulation_info']['N_PCA'], config_dd['simulation_info']['N_visit']
    boost_signal_factor = config_dd['simulation_info']['boost_signal_factor']
    boost_signal_factor_string = str(boost_signal_factor)
    infostring_save = f'{infostring}_{model_infostring}_NPCA_{N_PCA}_Nvisit_{N_visit}_boost{boost_signal_factor_string}x_'
    
    noise = config_dd['simulation_info']['noise']
    include_star = config_dd['simulation_info']['include_star']
    include_tellurics = config_dd['simulation_info']['include_tellurics']
    savedir = config_dd['simulation_info']['savedir']
    snr_thresh = config_dd['simulation_info']['snr_thresh']
    N_visit = config_dd['simulation_info']['N_visit']
    boost_signal_factor = config_dd['simulation_info']['boost_signal_factor']
    CC_species = config_dd['simulation_info']['CC_species']
    
    """Create the directory to save results."""
    now = datetime.datetime.now()
    # Format the date and time
    d1 = now.strftime("%d-%m-%YT%H-%M-%S")
    savedir_fin = savedir + infostring_save + d1 + '/'
    
    try:
        os.makedirs(savedir_fin)
    except OSError:
        savedir_fin = savedir_fin

    print('Saving files in directory: ', savedir_fin)

    copyfile(config_file_path, savedir_fin + 'simulation_config.yaml')
    
    ############## Print Obs details ###### 
    P_orb = planet_params["P_orb"]
    #print("Transit occurs between phases: ", -(0.5*transit_duration)/(P_orb*24.),(0.5*transit_duration)/(P_orb*24.) )
    #print("Secondary eclipse occurs between phases: ", 0.5- ((0.5*transit_duration)/(P_orb*24.)), 0.5+ ((0.5*transit_duration)/(P_orb*24.)) )

    

    ######### Define the path to the 1D model to be injected 
    model_dd_path_inject = config_dd['planet_info']['model_dir'] + config_dd['planet_info']['inject_model_path'] + 'spec_dict.npy'
    model_dd_inject = np.load(model_dd_path_inject, allow_pickle = True).item()
    
    ######### Define the path to the 1D model to be cross_correlated with 
    model_dd_path_cross_correlate = config_dd['planet_info']['model_dir'] + config_dd['planet_info']['cross_correlate_model_path'] + 'spec_dict.npy'
    model_dd_cross_correlate = np.load(model_dd_path_cross_correlate, allow_pickle = True).item()
    
    ########################################################################################################################
    ##### Loop over the species defined in CC_species and simulate the data and get the Kp-Vsys maps for each ######
    ########################################################################################################################
    
    for sp in CC_species:
        print('Doing: ', sp)
        model_spec_inject = model_dd_inject['spec']
        model_wav_inject = model_dd_inject['wav_nm']
        model_wav_cross_correlate = model_dd_cross_correlate['wav_nm']
        if sp == 'all':
            model_spec_cross_correlate = model_dd_cross_correlate['spec']
        else:
            model_spec_cross_correlate = model_dd_cross_correlate[sp]

        #override for single species injections
        model_spec_inject = model_spec_cross_correlate
        model_wav_inject = model_wav_cross_correlate
            
        plt.figure()
        plt.plot(model_wav_inject, model_spec_inject, label = 'Inject')
        plt.plot(model_wav_cross_correlate, model_spec_cross_correlate, label = 'Cross-correlate')
        plt.xlabel('Wavelength [nm]')
        if method == 'transmission':
            plt.ylabel('(Rp/Rs)$^{2}$')
        else:
            plt.ylabel('Fp/Fs')
            
        plt.legend()
        plt.title(sp)
        plt.savefig(savedir_fin + f'models_{sp}.png', format='png', dpi=300, bbox_inches='tight')
        


        n_ords = snr_data.shape[1]
        n_dets = snr_data.shape[0]

        KpVsys_total = {}

        for det in range(n_dets):
            for ord in range(n_ords):

                if ord ==3: pass
                elif sp=='sio' and ord not in [2]: pass # M-band hardcoding of the molecules to include! (for SiO etc)
                elif sp=='co' and ord not in [4]: pass
                elif sp=='h2o' and ord not in [0,1,2,4]: pass

                else:



                    print(f'Doing: det {det}, Order {ord}')
                    ############## Simulate data ######## 
                    

                    snr_inp = snr_data[det,ord]
                    stell_signal = star_data[det, ord]
                    data_wavsoln_inp = wave_data[det, ord]
                    
                    if ~include_star and ~include_tellurics:
                        
                        ######### Keep editing below and save H and K separately, as well as together summed ; in KpVsys save, also save the max logL for each map for later use 
                    
                        data_wavsoln, datacube_sim, model_cube, phases, wav_mask, RV_all = planit.get_simulated_data_2D_crires_stell(T_exp = T_exp, 
                                                    overheads = overhead_per_exp, snr_array_1D = snr_inp, 
                                                    data_wavsoln = data_wavsoln_inp, model_planet_spec_1D = model_spec_inject, 
                                                    model_planet_wav_1D = model_wav_inject, 
                                                    method = method, stell_signal = stell_signal, 
                                                    Kp = Kp_true, Vsys = Vsys_true, phase_range = phase_range, N_visit = N_visit, include_eclipse = False, 
                                                    planet_params = planet_params, plot_datacube = True, savedir = savedir_fin, 
                                                    boost_signal_factor = boost_signal_factor)

                        

                        ############### Run the cross-correlations to get 2D Kp-Vsys maps ############ 
                        band = f'{det}_{ord}'
                        KpVsys_save = planit.get_simulated_2D_CCF_crires_stell(datacube = datacube_sim, model_spec = model_spec_cross_correlate, 
                                                            data_wavsoln = data_wavsoln, 
                                                        model_wavsoln = model_wav_cross_correlate, Kp_range = Kp_range, 
                                                        Vsys_range = Vsys_range,
                                                        vel_window = vel_window, phases = phases,
                                                        snr_array_1D = snr_inp, snr_thresh = snr_thresh, wav_mask = wav_mask, 
                                                        savedir = savedir_fin, method = method, bandinfo = band, Kp_true = Kp_true, 
                                                            Vsys_true = Vsys_true, N_PCA = config_dd['simulation_info']['N_PCA'])
                        
                        KpVsys_total[f'{det}_{ord}'] = KpVsys_save
            

        
        ######## Plot the total logL KpVsys and its sigma contours ###### 



        KpVsys_total_logL = np.nansum(np.stack([v["logL"] for v in KpVsys_total.values()]), axis=0)

        KpVsys_total_CC = np.nansum(np.stack([v["cc"] for v in KpVsys_total.values()]), axis=0)
        KpVsys_total_CC_norm = (KpVsys_total_CC)/np.std((KpVsys_total_CC))
        
        KpVsys_logL_sigma = crocut.get_sigma_contours(logL_KpVsys=KpVsys_total_logL, dof=2) 
        
        KpVsys_total['logL_total'] = KpVsys_total_logL
        KpVsys_total['logL_max'] = np.max(KpVsys_total_logL)
        KpVsys_total['logL_sigma'] = KpVsys_logL_sigma
        KpVsys_total['CC'] = KpVsys_total_CC
        KpVsys_total['CC_norm'] = KpVsys_total_CC_norm

        try: wind = KpVsys_total['0_4']['Vsys_range_windowed']
        except: wind = KpVsys_total['0_2']['Vsys_range_windowed']
        ####### Save ##### 
        np.save(savedir_fin + 'KpVsys_total_'+ sp+ '.npy', KpVsys_total)
        
        ###### Plot the above ######## 
        
        plt.figure(figsize=(8,8))
        axx = plt.gca()
        _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                    matrix_2D= np.array(KpVsys_total_logL),
                                    Y=Kp_range,
                                    X= wind,
                                    title= 'logL : all_dets_all_orders' ,
                                    setxlabel=True, plot_type = 'pcolormesh')
        plt.colorbar(hnd1, ax=axx)
        plt.ylabel('K$_{P}$ [km/s]')
        plt.xlabel('V$_{sys}$ [km/s]')
        plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
        plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
        
        plt.savefig(savedir_fin + 'simulated_logL_total_' + sp + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,8))
        axx = plt.gca()
        _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                    matrix_2D = KpVsys_logL_sigma,
                                    Y=Kp_range,
                                    X= wind,
                                            title= 'log L sigma : all_dets_all_orders' ,
                                    setxlabel=True, plot_type = 'contourf', colormap = 'inferno_r')
        plt.colorbar(hnd1, ax=axx)
        plt.ylabel('K$_{P}$ [km/s]')
        plt.xlabel('V$_{sys}$ [km/s]')
        plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
        plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
        
        plt.savefig(savedir_fin + 'simulated_logL_sigma_total_'+sp+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,8))
        axx = plt.gca()
        _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                    matrix_2D = np.array(KpVsys_total_CC),
                                    Y=Kp_range,
                                    X= wind,
                                            title= 'CC : all_dets_all_orders' ,
                                    setxlabel=True, plot_type = 'pcolormesh', colormap = 'inferno')
        plt.colorbar(hnd1, ax=axx)
        plt.ylabel('K$_{P}$ [km/s]')
        plt.xlabel('V$_{sys}$ [km/s]')
        plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
        plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
        
        plt.savefig(savedir_fin + 'simulated_CC_total_norm_'+sp+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,8))
        axx = plt.gca()




        _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                    matrix_2D = np.array(KpVsys_total_CC - np.nanmedian(KpVsys_total_CC))/np.nanstd(KpVsys_total_CC[-15:, -15:]),
                                    Y=Kp_range,
                                    X= wind,
                                            title= 'CC SNR: all_dets_all_orders' ,
                                    setxlabel=True, plot_type = 'pcolormesh', colormap = 'inferno')
        plt.colorbar(hnd1, ax=axx)
        plt.ylabel('K$_{P}$ [km/s]')
        plt.xlabel('V$_{sys}$ [km/s]')
        plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
        plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
        
        plt.savefig(savedir_fin + 'simulated_CC_total_norm_SNR_'+sp+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print('Done: ', sp)
        
print('Done ALL, check: ', savedir_fin)
    
    ######## If still debugging, plot data and model to check ##### 

# ### Play system beep when done     
# from AppKit import NSBeep
# NSBeep()