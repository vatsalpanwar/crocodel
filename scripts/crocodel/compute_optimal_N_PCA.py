#!/home/vatsal/anaconda3/bin/python3
import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import astropy.io.ascii as asc
from astropy.modeling import models
from astropy import units as un
from scipy import interpolate
import datetime
import os
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from scipy.interpolate import splev, splrep
from tqdm import tqdm
import gc
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
import sys
from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import cross_correlation_utils as croc
from crocodel.crocodel import astro_utils as aut
import math
import argparse
import yaml

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### #####  Load the config_dd ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
parser = argparse.ArgumentParser(description='Read the user inputs.')
parser.add_argument('-cfg','--config_file_path', help = "Path to the croc_config.yaml.",
                    type=str, required=True)
args = vars(parser.parse_args())
config_file_path = args['config_file_path']
with open(config_file_path) as f:
    config_dd = yaml.load(f,Loader=yaml.FullLoader)

### Info
infostring = config_dd['infostring']
result_dir = config_dd['directories']['result_dir']

### PCA
N_PCA_range = np.arange(config_dd['PCA_params']['N_PCA_range'][0], config_dd['PCA_params']['N_PCA_range'][1], 1)
post_pca_threshold_type = config_dd['PCA_params']['post_pca_threshold_type']

### Model
MODEL_TYPE = config_dd['telluric_model']['model_type']
vel_range = np.arange(config_dd['telluric_model']['vel_range'][0],config_dd['telluric_model']['vel_range'][1], 
                      config_dd['telluric_model']['vel_step'])
model_path = config_dd['telluric_model']['model_path']
model_resolution = config_dd['telluric_model']['model_resolution']
### Data
datadir_dict = config_dd['data']['dates']
inst_name = config_dd['data']['inst_name']
resolution = config_dd['data']['resolution']
method = config_dd['data']['method']
ndet = config_dd['data']['ndet'] ## Number of detectors or orders 
badcolmask_inds = config_dd['data']['badcolmask_inds']

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### Define the directories where to save the result ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

now = datetime.datetime.now()
# Format the date and time
d1 = now.strftime("%d-%m-%YT%H-%M-%S")
infostring_all = 'OPT_PCA_CALC_'+d1+'_' + MODEL_TYPE + '_' + infostring
savedir = result_dir + '/telluric_cross_correlations/' + infostring_all + '/'
try:
    os.makedirs(savedir)
except OSError:
    savedir = savedir

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### Create a tree of directories for each detector ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
savedir_dd = {}
for idate, date in enumerate(datadir_dict.keys()):
    os.makedirs(savedir + '/' + date + '/')
    savedir_dd[date] = {}
    for idet in range(ndet):
        os.makedirs(savedir + '/' + date + '/det_' + str(idet) + '/')
        savedir_dd[date][idet] = savedir + '/' + date + '/det_' + str(idet) + '/'

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### Adapt the model to the instrument  ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
print('Convolving the ESO SkyCalc model to the instrument resolution...')
if MODEL_TYPE == 'eso_skycalc':
    esosc = astropy.io.fits.getdata(model_path)
    ####### Convolve the ESO SkyCalc model to the instrument resolution 
    # Convolve the model to the instrument resolution 
    # plt.figure()
    # plt.plot(esosc['lam'], esosc['trans'])
    # plt.show()
    delwav_by_wav_instrument = resolution
    print('delwav_by_wav_instrument', delwav_by_wav_instrument)
    # delwav_by_wav_model = np.mean(np.diff(esosc['lam'])/esosc['lam'][1:])
    delwav_by_wav_model = model_resolution
    print('delwav_by_wav_model', delwav_by_wav_model)
    FWHM_kernel = np.mean(delwav_by_wav_instrument/delwav_by_wav_model)
    sig_kernel = FWHM_kernel / (2. * np.sqrt(2. * np.log(2.) ) )
    print('sig_kernel', sig_kernel)
    gauss1d_kernel = Gaussian1DKernel(stddev=sig_kernel)
    # model_tell = convolve(esosc['trans']/np.median(esosc['trans']), gauss1d_kernel, boundary='extend')
    ## Trying without dividing by the median (Vatsal : 03-12-2023)
    model_tell = convolve(esosc['trans'], gauss1d_kernel, boundary='extend')
    model_tell_wavsoln = np.array(esosc['lam']) ## should also be in nm
    print(model_tell_wavsoln)
    # plt.figure()
    # plt.plot(model_tell_wavsoln, model_tell)
    # plt.show()

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# for N_PCA in tqdm(N_PCA_range):
#     for idate, date in enumerate(datadir_dict.keys()):

#         print('Starting date: ', date)
#         if inst_name == 'crires':
#             # Read in the data for this date
#             data = astropy.io.fits.getdata(datadir_dict[date])
#             # Parse the datacubes and other auxiliary data from the files
#             spdatacube = data['SPEC'][0][:,:,:]
#             airmass = data['AIRM'][0]
#             phases = data['PH'][0]
#             wavsoln = data['WLEN'][0][:,0,:]
#             # bary_RV = data['RVEL'][0]
#             time = data['MJD'][0]
#             bary_RV = -1. * np.array( aut.get_BERV(time_array = time, time_format = 'mjd',
#                         ra_dec = ['13:47:15.46','+17:26:42.64'],
#                         obs_location = [-24.617,-70.4,2635]) ) # By default the velocity of the barycenter in the observer's rest frame
        
#         elif inst_name == 'igrins':
#             sp_dd = np.load(datadir_dict[date][0], allow_pickle = True).item()
#             spdatacube = sp_dd['spdatacube']
#             airmass =sp_dd['airmass']
#             phases = sp_dd['phases']
#             wavsoln = sp_dd['wavsoln']
#             time = sp_dd['time'] ## BJD_TDB
#             bary_RV = sp_dd['bary_RV']
            
#             # plt.figure()
#             # plt.plot(model_tell_wavsoln, model_tell, color = 'r')
#             # plt.plot(wavsoln[0,:],spdatacube[0,0,:], color = 'k' )
#             # plt.show()
        
#         # nspec, nwav = spdatacube.shape[1], spdatacube.shape[2]
#         # # Run the PCA to obtain the eigenvectors
#         # pca_eigenvectors = np.empty((ndet,nspec,N_PCA+1)) ## Need to change this to allow different PCA per component 
#         # spdatacube_fit = np.empty((ndet, nspec, nwav))
#         # spdatacube_detrended = np.empty((ndet, nspec, nwav))
        
#         ### Loop over orders ####
#         for idet in tqdm(range(ndet)):
#             colmask_info = badcolmask_inds[date][idet]
#             _ = stc.get_telluric_trail_matrix_per_detector(datacube=spdatacube[idet, :, :], 
#                                                            data_wavsoln=wavsoln[idet, :], 
#                                                            model_tell=model_tell , 
#                                                            model_tell_wavsoln=model_tell_wavsoln,
#                                                             vel_range = vel_range, 
#                                                             berv=bary_RV, 
#                                                             phases = phases, 
#                                                             N_PCA = N_PCA,
#                                             save = True, savedir = savedir_dd[date][idet], 
#                                             date = date, idet = idet, colmask_info=colmask_info,
#                                             return_trail_matrix = False, post_pca_threshold_type=post_pca_threshold_type)
#             plt.close('all')
#             gc.collect()
            
            
            
print('Done computing PCA optimization, constructing summary plots...')

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### Demo plots ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

########## First construct a tree of paths ; this may not be the same as for computing because of time stamp being different (if compute and demo figure is not run at the same time) ############ 

trail_matrix_dd = {}
rootdir = result_dir + 'telluric_cross_correlations/'
cases = {
    MODEL_TYPE:'OPT_PCA_CALC_26-04-2024T03-45-46_eso_skycalc_IGRINS_WASP-122b_emission_wave_recal' # infostring_all
}

# for case in cases:
#     trail_matrix_dd[case] = {}
    
#     for date in datadir_dict.keys():
#         trail_matrix_dd[case][date] = {}
#         for det in range(ndet):
#             det = str(det)
#             trail_matrix_dd[case][date][det] = {}
#             suffix = '_trail_matrix_date-'+date+'_idet-'+det+'.npy'
#             for N_PCA in N_PCA_range:
#                 trail_matrix_dd[case][date][det][N_PCA] = rootdir + cases[case] +'/' + date + '/det_'+ det + '/N_PCA-'+str(N_PCA)+suffix

# wav_phase_dd = {}
# for date in datadir_dict.keys():
#     dat = np.load(datadir_dict[date][0], allow_pickle = True).item()
#     wav_phase_dd[date] = {}
#     for det in range(ndet):
#         det = str(det)
#         wav_phase_dd[date][det] = {}
#         wav_phase_dd[date][det]['phases'] = dat['phases']
#         wav_phase_dd[date][det]['wavsoln'] = dat['wavsoln'][int(det), : ]
# ####################################################################
# ################# Make the demo plots #################### #################
# ####################################################################
# summary_dd_all = {}
# for case in cases:
#     summary_dd_all[case] = {}
#     for date in datadir_dict.keys():
#         summary_dd_all[case][date] = {}
#         for det in range(ndet):
#             det = str(det)
#             N_PCA_range = N_PCA_range
#             savedir_all = rootdir + cases[case] +'/' + date + '/det_'+ det + '/'
#             dd_ = stc.make_paper_figure_PCA_optim_demo(case = case, date = date, det = det, N_PCA_range = N_PCA_range, 
#                                                        trail_matrix_dd = trail_matrix_dd,
#                                      savefig = True, savedir = savedir_all, wav_phase_dd=wav_phase_dd)
#             summary_dd_all[case][date][det] = dd_
#         plt.close('all')
# np.save(rootdir + cases[case] + 'PCA_optimization_summary_dd.npy', summary_dd_all)

# print('PCA optimization summary computed and saved.')
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### Make the summary plot (one plot per detector of both sigma_tell and Delta sigma_tell)  ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
case = MODEL_TYPE
summary_dd_all = np.load(rootdir + cases[case] + 'PCA_optimization_summary_dd.npy', allow_pickle = True).item()
for date in datadir_dict.keys():
    for det in range(ndet):
        det = str(det)
        fig, ax = plt.subplots(2,1,figsize = (10,16))
        plt.subplots_adjust(wspace=0.3)
        
        savedir = rootdir + cases[case] +'/' + date  + '/det_'+ det + '/'
        savedir_ = rootdir + cases[case] +'/' + date  + '/' #+ '/det_'+ det + '/'  
        ax[0].plot(N_PCA_range, 
                    [summary_dd_all[case][date][det][N_PCA]['hist_sig'] for N_PCA in N_PCA_range], 
                    marker = 'o', color = 'k')

        ## Difference from one sigma to next 
        ax[1].plot(N_PCA_range[1:], 
                    np.diff([summary_dd_all[case][date][det][N_PCA]['hist_sig'] for N_PCA in N_PCA_range]), 
                    marker = 'o', color = 'k' )
        
        plt.title('Order: ' + det)
        
        ax[0].set_ylabel('$\sigma _{tell}$') 
        ax[1].set_ylabel('$\Delta \sigma _{tell}$')

        plt.savefig(savedir + case + '_det-' + det +'_telluric_CCF_hist_vs_NPCA.png', 
                    format='png', bbox_inches='tight', dpi = 300)
        
        plt.savefig(savedir_ + case + '_det-' + det +'_telluric_CCF_hist_vs_NPCA.png', 
                    format='png', bbox_inches='tight', dpi = 300)
        plt.clf()
        plt.close('all')






##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

