import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import scipy
from astropy import units as un
from astropy import constants as con
from scipy.interpolate import splev, splrep
from astropy.modeling import models
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from tqdm import tqdm
from scipy import interpolate

from crocodel.crocodel import stellcorrection_utils as stc
from crocodel.crocodel import cross_correlation_utils as crocut
import skycalc_ipy

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
############## Modules to do basic calculations for high resolution spectroscopy proposals ###########
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 


####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
####### ####### ####### #######  IGRINS ####### ####### ####### ####### ####### ####### ####### ####### 
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 

####### ####### ####### ####### #######  Only photon noise ####### ####### ####### ####### ####### ####### 
def get_simulated_data_2D_igrins_photon_noise(P_orb = None, T_exp = None, overheads = None, snr_array_1D = None, data_wavsoln = None, 
                          model_planet_spec_1D = None, model_planet_wav_1D = None, method = 'emission',
                          star_only_signal = None, Kp = 188, Vsys = 35., phase_range = [0.45, 0.48] ):


    T_dur = P_orb * abs(phase_range[1] - phase_range[0]) * 24. ## in hours 
    
    print("Duration of observations: (hours)", T_dur)
    
    T_exp = T_exp ## In seconds ; Spandan said that the output he is getting from his tool for 100 seconds is closer to 200 seconds from the ETC  
    overheads =  overheads ## E.g. for crires 2.4 + 28/2. ## In seconds, 2.4 seconds is the readout, and 28 seconds is the overhead per nodding cycle i.e. per AB pair, so per exp it is 28/2 
    
    N_exp = int((T_dur * 3600)/(T_exp+overheads))
    
    print("Number of exposures", N_exp)
    
    phases = np.linspace(phase_range[0], phase_range[1], N_exp)
    print(phases)
    N_wav = len(data_wavsoln)
    
    # damp_fac = 1./np.sqrt(Nexp*N_visit)
    
    damp_fac = 1. ## set it to 1 for now 
    
    ######### Arrays to fill 
    model_cube = np.ones((N_exp, N_wav))
    datacube_sim = np.ones((N_exp, N_wav))
    RV_all = np.ones((N_exp,))
    
    ## First convolve model to the IGRINS resolution 
    delwav_by_wav = 1/45000 # igrins
    delwav_by_wav_model = np.diff(model_planet_wav_1D)/model_planet_wav_1D[1:]
    FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )
    model_planet_spec_conv = convolve(model_planet_spec_1D, Gaussian1DKernel(stddev=sig), boundary='extend')
    
    ## Resample to the data wavelength solution 
    model_spl = splrep(model_planet_wav_1D, model_planet_spec_conv)
    
    #### For each exposure, doppler shift the model by the Vp + Vsys, and then inject to the data. 
    if star_only_signal is None:
        star_only_signal = np.ones((N_wav))
    
    # inf_mask_all = []
    
    for iexp in range(N_exp):
        
        RV = Kp * np.sin( 2. * np.pi * phases[iexp] ) + Vsys
        RV_all[iexp] = RV
        
        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * RV)
        
        model_planet_spec_shift = splev(data_wavsoln_shift, model_spl)
        
        model_cube[iexp,:] = model_planet_spec_shift
        
        ## Inject the model to the data 
        
        if method == 'emission':
            star_and_planet = star_only_signal*(1. + 1e6*model_planet_spec_shift) ## Ftot = Fs * (1. + Fp/Fs)
        elif method == 'transmission':
            star_and_planet = star_only_signal*(1. - model_planet_spec_shift) ## Ftot = Fs * (1. - (Rp/Rs)**2. )
        # print('star_and_planet', np.where(star_and_planet<0.))
        # print('snr_array_1D', np.where(snr_array_1D<0.))
        noise = damp_fac*np.random.normal(0., abs(star_and_planet/snr_array_1D ))
        
        datacube_sim[iexp, :] =  (star_and_planet + noise)/star_only_signal

    #     inf_mask_all.append(~np.isinf(datacube_sim[iexp,:]))
    
    # inf_mask_comb = np.logical_or.reduce(np.array(inf_mask_all), axis = 0)
    
    ## CHECK above 
        
    #### Compute the mask for which the data will not be analysed 
    
    wav_mask = np.ones(N_wav, dtype = bool)
    # minw, maxw = np.min(data_wavsoln), np.max(data_wavsoln)
    
    for iw in range(N_wav):
        # if data_wavsoln[iw] < 1450 or (1790 < data_wavsoln[iw] < 1970) or data_wavsoln[iw] > 2420:
        if data_wavsoln[iw] < 1450 or data_wavsoln[iw] > 1760 or data_wavsoln[iw] > 2420:
            wav_mask[iw] = False
    
    #### Combine the wav mask with the inf mask 
    # wav_mask_fin = np.logical_and(wav_mask, inf_mask_comb) 
    wav_mask_fin = wav_mask

    return data_wavsoln, datacube_sim, model_cube, phases, wav_mask_fin, RV_all
        
    
def get_simulated_2D_CCF_igrins_photon_noise(datacube = None, model_spec = None, data_wavsoln = None, 
                                model_wavsoln = None, Kp_range = None, Vsys_range = None,
                                vel_window = None, phases = None,
                                snr_array_1D = None, snr_thresh = 20, wav_mask = None, savedir = None, method = 'emission',
                                Kp_true = None, Vsys_true = None, bandinfo = 'H'):
    """Function to simulate 1D CCF and logL for a simulated observed data and a given model spectrum.  

    :param datacube: _description_, defaults to None
    :type datacube: _type_, optional
    :param model_spec: _description_, defaults to None
    :type model_spec: _type_, optional
    :param data_wavsoln: _description_, defaults to None
    :type data_wavsoln: _type_, optional
    :param model_wavsoln: _description_, defaults to None
    :type model_wavsoln: _type_, optional
    :param velocity_range: _description_, defaults to None
    :type velocity_range: _type_, optional
    :param snr_array_1D: _description_, defaults to None
    :type snr_array_1D: _type_, optional
    :param snr_thresh: _description_, defaults to 20
    :type snr_thresh: int, optional
    :return: _description_
    :rtype: _type_
    """
    
    ## First convolve model to the IGRINS resolution 
    delwav_by_wav = 1/45000 # igrins
    delwav_by_wav_model = np.diff(model_wavsoln)/model_wavsoln[1:]
    FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )  
    print(sig)         
    model_spec_conv = convolve(model_spec, Gaussian1DKernel(stddev=sig), boundary='extend')
    
    ## Create a spline interpolation for the model 
    if method == 'transmission':
        model_spl = splrep(model_wavsoln, 1. - model_spec_conv)
    elif method == 'emission':
        model_spl = splrep(model_wavsoln, model_spec_conv)
        
    nspec, N_wav = datacube.shape
    nKp, nVsys = len(Kp_range), len(Vsys_range)

    cc_matrix, logL_matrix = np.zeros((nspec, nVsys)), np.zeros((nspec, nVsys))
    
    snr_mask = snr_array_1D > snr_thresh
    
    use_mask = np.logical_or(wav_mask, snr_mask)
    
    
    #############################################################
    ###### Optionally perform PCA here on the data first ########
    #############################################################    
    
    ############ Get trail matrix first ############
    
    for it in tqdm(range(nspec)):
        ## Loop over velocities 
        for iv, vel in enumerate(Vsys_range):
            # First Doppler shift the data wavelength solution to -vel
            data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * vel)
            # Evaluate the model to the data_wavsoln_shifted by -vel,
            # Effectively Doppler shifting the model by +vel
            model_spec_flux_shift = splev(data_wavsoln_shift, model_spl)
            # Subtract the mean from the model
            model_spec_flux_shift = model_spec_flux_shift - crocut.fast_mean(model_spec_flux_shift)
            # Compute the cross correlation value between the shifted model and the mean subtracted data
            
            inf_mask = ~np.isinf(datacube[it,:])
            
            data_mean_sub = datacube[it,:] - crocut.fast_mean(datacube[it,inf_mask])

            use_mask_fin = np.logical_and(use_mask,inf_mask)
            
            _, cc_matrix[it,iv], logL_matrix[it,iv] = crocut.fast_cross_corr(data = data_mean_sub[use_mask_fin], 
                                                                          model = model_spec_flux_shift[use_mask_fin])
            
                 
    ########### Shift rows in trail matrix to get Kp-Vsys maps ####### 
    
    CC_KpVsys, logL_KpVsys = np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
    for iKp, Kp in enumerate(Kp_range):
        CC_matrix_shifted, logL_matrix_shifted = np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
        for it in range(nspec):
            Vp = Kp * np.sin(2. * np.pi * phases[it])
            
            Vsys_shifted = Vsys_range + Vp

            func_CC = interpolate.interp1d(Vsys_range, cc_matrix[it, :] ) 
            func_logL = interpolate.interp1d(Vsys_range, logL_matrix[it, :] )
            
            CC_matrix_shifted[it,:] = func_CC(Vsys_shifted[vel_window[0]:vel_window[1]])
            logL_matrix_shifted[it,:] = func_logL(Vsys_shifted[vel_window[0]:vel_window[1]])
        
        CC_KpVsys[iKp,:], logL_KpVsys[iKp,:] = np.sum(CC_matrix_shifted, axis = 0), np.sum(logL_matrix_shifted, axis = 0)

    
    KpVsys_save = {}
    KpVsys_save['logL'] = logL_KpVsys
    KpVsys_save['cc'] = CC_KpVsys
    KpVsys_save['Kp_range'] = Kp_range
    KpVsys_save['Vsys_range'] = Vsys_range
    KpVsys_save['vel_window'] = vel_window
    KpVsys_save['Vsys_range_windowed'] = Vsys_range[vel_window[0]:vel_window[1]]
    
    # np.save(savedir + 'KpVsys_fast_no_model_reprocess_dict.npy', KpVsys_save)

    # fig, axx = plt.subplots(subplot_num, 1, figsize=(8, 8*subplot_num))
    # plt.subplots_adjust(hspace=0.6)
    plt.figure(figsize=(8,8))
    axx = plt.gca()
    _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                matrix_2D=KpVsys_save['cc'],
                                Y=Kp_range,
                                X=KpVsys_save['Vsys_range_windowed'],
                                ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                title= 'CCF' ,
                                setxlabel=True, plot_type = 'pcolormesh')
    plt.colorbar(hnd1, ax=axx)
    plt.ylabel('K$_{P}$ [km/s]')
    plt.xlabel('V$_{sys}$ [km/s]')
    plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
    plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
    
    plt.savefig(savedir + 'simulated_CC_'+bandinfo+'.png', format='png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(8,8))
    axx = plt.gca()
    _, hnd1 = crocut.plot_2D_cmap(axis=axx,
                                matrix_2D=KpVsys_save['logL'],
                                Y=Kp_range,
                                X=KpVsys_save['Vsys_range_windowed'],
                                ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                title= 'log L' ,
                                setxlabel=True, plot_type = 'pcolormesh')
    plt.colorbar(hnd1, ax=axx)
    plt.ylabel('K$_{P}$ [km/s]')
    plt.xlabel('V$_{sys}$ [km/s]')
    plt.axvline(x = Vsys_true, linestyle = 'dotted', color = 'w')
    plt.axhline(y = Kp_true, linestyle = 'dotted', color = 'w')
    
    plt.savefig(savedir + 'simulated_logL_'+bandinfo+'.png', format='png', dpi=300, bbox_inches='tight')

    return KpVsys_save

####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######  
####### ####### ####### ####### ####### Add tellurics, stellar spectrum, and do PCA to above for more realistic simulations ####### ####### 
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 





####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
####### ####### ####### ####### ####### ####### ####### ####### 
###### Below CRIRES+ ####### ####### ####### ####### ####### 
####### ####### ####### ####### ####### ####### ####### ####### 
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
## write separate functions for individual instruments for now, combine them later in one if possible 

def get_simulated_1D_CCF_crires(datacube = None, model_spec = None, data_wavsoln = None, model_wavsoln = None, 
                                velocity_range = None, snr_cube = None, case = None, skip_orders = None, snr_thresh = 20):
    """Function to simulate 1D CCF and logL for a simulated observed data and a given model spectrum.  

    :param datacube: 2D datacube of simulated observed data (star + planet) with shape (Ndet, Nwav) where n_det is number of orders/detectors, and n_wav is the number of wavelength channels, defaults to None
    :type datacube: ndarray
    :param model: 1D model spectrum, defaults to None
    :type model: ndarray
    :param data_wavsoln: Wavelength solution of the data, defaults to None
    :type data_wavsoln: ndarray
    :param model_wavsoln: Wavelength solution of the model, defaults to None 
    :type model_wavsoln: ndarray
    :param velocity_range: Range of velocities in km/s, defaults to None 
    :type model_wavsoln: ndarray
    :param snr_cube: 2D cube of snr, defaults to None 
    :type snr_cube: ndarray
    :param case: is it transmission or emission?, defaults to None 
    :type snr_cube: str
    """
    
    ## First convolve model to the CRIRES resolution 
    delwav_by_wav = 1/100000 # crires 
    delwav_by_wav_model = np.diff(model_wavsoln)/model_wavsoln[1:]
    FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )  
    print(sig)         
    model_spec_conv = convolve(model_spec, Gaussian1DKernel(stddev=sig), boundary='extend')
    
    ## Create a spline interpolation for the model 
    if case == 'transmission':
        model_spl = splrep(model_wavsoln, 1. - model_spec_conv)
    elif case == 'emission':
        model_spl = splrep(model_wavsoln, model_spec_conv)
        
    Ndet, Nwav = datacube.shape
    
    R, CC, logL = np.empty((Ndet, len(velocity_range))), np.empty((Ndet, len(velocity_range))), np.empty((Ndet, len(velocity_range))) 
    
    if skip_orders is None:
        skip_orders = []
    
    ## Loop over the orders 
    for idet in range(Ndet):
        
        if idet in skip_orders:
            for ivel in range(len(velocity_range)):
                R[idet,ivel], CC[idet,ivel], logL[idet,ivel] = 0.,0.,0.
        else:
                
            snr_mask = snr_cube[idet, :] > snr_thresh
            data_masked, data_wavsoln_masked = datacube[idet, snr_mask], data_wavsoln[idet, snr_mask]
            
            for ivel in range(len(velocity_range)):
                
                ### Doppler shift the wavelength solution of the data by negative velocity 
                data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln_masked, velocity=-1. * velocity_range[ivel])
                
                ### Shift the model to this Doppler shifted wavelength solution
                # print(idet)
                # print(data_wavsoln_shift) 
                # import pdb
                # pdb.set_trace()
                model_spec_shift = splev(data_wavsoln_shift, model_spl)
                
                ### Mean subtract the model 
                model_spec_shift_norm = model_spec_shift - crocut.fast_mean(model_spec_shift)
                
                ### Normalize the data by continuum 
                # data_masked_norm_cont = data_masked-np.poly1d( np.polyfit(data_wavsoln_masked, data_masked, 2) )(data_wavsoln_masked)
                
                data_masked_norm_cont = data_masked - crocut.fast_mean(data_masked)
                
                if ivel == 0:
                    plt.figure()
                    plt.plot(model_wavsoln, model_spec)
                    # plt.plot(data_wavsoln_masked, data_masked, alpha = 0.2, color = 'k')
                    # plt.plot(data_wavsoln_masked, data_masked_norm_cont, alpha = 0.5, color = 'b')
                    # plt.plot(data_wavsoln_masked, model_spec_shift_norm, alpha = 0.5, color = 'r')
                    plt.show()
                    
                
                R[idet,ivel], CC[idet,ivel], logL[idet,ivel] = crocut.fast_cross_corr(data = data_masked_norm_cont, model = model_spec_shift_norm)


    return R, CC, logL
            
###########################################################################################################################################################################     
#################################### Function to take a 2D grid of parameters for which models have been calculated, and compute a 2D grid of logL and their contours ##### 
###########################################################################################################################################################################

def get_2D_logL_grid_contours(logL_max_dd = None, param1_arr = None, param2_arr = None, param1_name = None, param2_name = None, savedir = None, infostring = None,
                              xlims = [-5,-1], ylims = [-5,-3],levels = 14):
    
           
    ## Make the plot and save 
    maxmk = max(logL_max_dd, key=logL_max_dd.get)
    
    logL_sigma_grid = {}
    logL_sigma_grid_info = {}
    logL_sigma_grid_info['param1_name'] = param1_name
    logL_sigma_grid_info['param1_arr'] = param1_arr
    logL_sigma_grid_info['param2_name'] = param2_name
    logL_sigma_grid_info['param2_arr'] = param2_arr
 
    for mk in logL_max_dd.keys():
        
        DeltalogL = logL_max_dd[mk] - logL_max_dd[maxmk]

        chi2 = -2. * DeltalogL

        p_one_tail = 0.5 * scipy.stats.chi2.sf(chi2, 2)

        # sigma_levels = scipy.stats.norm.ppf(1-p_one_tail, dof) ## Note that norm.ppf(0.95) is same as norm.isf(0.05)
        sigma_level = scipy.stats.norm.isf(p_one_tail) ## Gives better precision
        
        logL_sigma_grid[mk] = sigma_level
    
    Z = np.empty((len(param1_arr),len(param2_arr)))
    for i in range(len(param1_arr)):
        for j in range(len(param2_arr)):
            Z[i, j] = logL_sigma_grid[(param1_arr[i], param2_arr[j])]  ## logL_max_dd must have keys formed from grid combination of param1_arr and param2_arr 
 
    fig, hnd = crocut.plot_2D_cmap(axis=None, matrix_2D=Z, 
             Y=param1_arr, X=param2_arr, 
             title=None,
                      setxlabel=param2_name, setylabel=param1_name, 
                      plot_type = 'contourf',xlims = xlims, ylims = ylims, levels = levels )
    fig.colorbar(hnd)
    plt.savefig(savedir + param1_name + '_' + param2_name + '_2D_logL_grid_' + infostring + '.png', format = 'png', dpi = 300)
    plt.show()
    return logL_sigma_grid, logL_sigma_grid_info
    
            
def get_CbyO_MbyH(logH2O = None, logCO = None):
    """
    Given a single value of water and CO VMR, compute the C/O and metallicity.
    """         
    ## Lodders et al. (2009) proto-Sun values
    C0 = 2.77E-4
    N0 = 8.19E-5
    O0 = 6.07E-4
    
    ####### C/O ratio ######## 
    nC = 10**(logCO)
    nO = 10**(logCO)+10**(logH2O)
    coRatio = nC/nO
    
    ####### M/H ###############
    n_H2He = 1. - 10.**logCO - 10.**logH2O  # Number density of the H2-He mixture
    n_H2 = n_H2He / (1.0 + 0.176)           # Number density of the H2 gas
    H_corr = 2.0 * n_H2   
    
    nMetals = 10**logCO*2 + 10**logH2O
    denom = H_corr*(C0+O0) ## Add N0 only when including the nitrogen species above 
    
    metall = nMetals / denom
    logMet = np.log10(metall)
    
    return coRatio, logMet
    

        
        
        
        
    
    
    

