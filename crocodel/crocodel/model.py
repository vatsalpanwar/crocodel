import sys 
# sys.path.insert(0, "/home/astro/phsprd/code/genesis/code") 
# sys.path.insert(0, "/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/genesis/code/opac")

# sys.path.insert(0, "/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/genesis/code/")

import numpy as np
import yaml
import genesis
# from opac.opac import Opac
import scipy.constants as sc
import stellcorrection_utils as stc
from . import cross_correlation_utils as crocut
from scipy.interpolate import splev, splrep
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from . import astro_utils as aut
from tqdm import tqdm 
import time
import pdb
from astropy.io import ascii as ascii_reader

class Model:
    # Defined per instrument and per date.
    def __init__(self, *args, **kwargs):
        with open(kwargs.pop('config')) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # """Setup the planetary system for genesis for a particular instrument and date by 
        # setting the star and planetary properties. Use the initial 
        # values defined in the croc_config.yaml file.
        
              # From the keyword arguments, can be taken from the config file but that needs to be specified outside 
        self.inst = kwargs.pop('inst') 
        # self.date = kwargs.pop('date')
        
        # From the config file 
        self.method = self.config['data'][self.inst]['method']
        # From the config file (can also run the following in just one loop using setattr)
        self.TP_type = self.config['model']['TP_type']
        
        self.R_star = self.config['model']['R_star']
        self.T_eff = self.config['model']['T_eff']
        self.P_min = self.config['model']['P_min']
        self.P_max = self.config['model']['P_max']
        self.N_layers = self.config['model']['N_layers']
        self.lam_min = self.config['model']['lam_min']
        self.lam_max = self.config['model']['lam_max']
        self.resolving_power = self.config['model']['R_power']
        self.spacing = self.config['model']['spacing']
        
        if self.TP_type == 'Linear':
            self.P2 = 10.**self.config['model']['P2']
            self.T2 = self.config['model']['T2']
            self.P1 = 10.**self.config['model']['P1']
            self.T1 = self.config['model']['T1']
            
        elif self.TP_type == 'Guillot':
            self.T_int= self.config['model']['T_int']
            self.T_irr= self.config['model']['T_irr']
            self.log_gamma= self.config['model']['log_gamma']
            self.log_kappa_IR= self.config['model']['log_kappa_IR']
            self.f_global= self.config['model']['f_global'] # Setting this to one let's it be folded into T_eq
        elif self.TP_type == 'custom_fixed':
            self.TP_data = ascii_reader.read(self.config['model']['TP_path'])
        elif self.TP_type == 'Madhusudhan_Seager':
            self.T_set = self.config['model']['T_set']
            self.alpha1 = self.config['model']['alpha1']
            self.alpha2 = self.config['model']['alpha2']
            self.log_P1 = self.config['model']['log_P1']
            self.log_P2 = self.config['model']['log_P2']
            self.log_P3 = self.config['model']['log_P3']
        elif self.TP_type == 'Bezier_4_nodes': ## In order of increasing pressures (or going from top to down in altitude) : P3,P2,P1,P0 
            self.T0 = self.config['model']['T0']
            self.log_P0 = self.config['model']['log_P0']
            self.T1 = self.config['model']['T1']
            self.log_P1 = self.config['model']['log_P1']
            self.T2 = self.config['model']['T2']
            self.log_P2 = self.config['model']['log_P2']
            self.T3 = self.config['model']['T3']
            self.log_P3 = self.config['model']['log_P3']
            
        
        self.R_planet= self.config['model']['R_planet']
        self.log_g= self.config['model']['log_g']
        self.P_ref= self.config['model']['P_ref']
        self.cl_P = self.config['model']['cl_P']
        self.log_fs = self.config['model']['log_fs'] ## model scale factor 
        self.phase_offset = self.config['model']['phase_offset']
        self.Kp = self.config['model']['Kp']
        self.Vsys = self.config['model']['Vsys']
        
        self.Kp_pred = self.Kp
        self.Vsys_pred = self.Vsys

        self.free_params_dict = self.config['model']['free_params'] 
        
        # Initialize the molecular/atomic abundances as model parameters
        self.species = list(self.config['model']['abundances'].keys())
        for sp in self.species:
            setattr(self, sp, 10.**self.config['model']['abundances'][sp])
        # Set hydrogen as everything besides other species 
        setattr(self, 'h2', 1. - sum([10.**self.config['model']['abundances'][x] for x in self.species]) )

        ## First define an Atm object, and then define an initial opac object here
        # self.Opac_init = Opac() 
        # self.atm_init = self.gen.atm
        # self.atm_init.atm_set(self.gen.P, self.gen.T, self.gen.n, self.gen.r, self.gen.rho, self.gen.g, self.gen.Rp, self.gen.lam, self.abundances_dict)
        # self.Opac_init.opacity(self.atm_init) # This runs all the opac_check functions that read in the x-sections of species in the abundance dictionary once.        
        
        self.Genesis_instance = genesis.Genesis(self.P_min, self.P_max, self.N_layers, self.lam_min, self.lam_max, self.resolving_power, self.spacing, method = self.method)
        
        
    # @property
    # def parameter(self, param_name):
    #     return self.param_name
    
    @property
    def gen(self):
        """Get the Genesis object based on the latest value of parameters.

        :return: _description_
        :rtype: _type_
        """
        # gen_ = genesis.Genesis_no_opac_check(self.P_min, self.P_max, self.N_layers, self.lam_min, self.lam_max, self.resolving_power, self.spacing, method = self.method,
        #                              Opac_obj = self.Opac_init)
        
        gen_ = self.Genesis_instance
        
        if self.TP_type == 'Linear':
            # From Sid'e email and looking at set_T function, 
            # Order is (P1,T1),(P2,T2),[P0=,T0=], i.e. down to top. P1 must be greater than P2!
            gen_.set_T(self.P1, self.T1, self.P2, self.T2) # This part should have options to choose different kinds of TP profile.
        elif self.TP_type == 'Guillot':
            gen_.T = aut.guillot_TP(pressure_levels = gen_.P.copy()/1e5, # All input pressure values should be in bars for this (the function will convert to Pascals itself)
                                   T_int = self.T_int, 
                                   T_eq = None, 
                                   gamma = 10.**self.log_gamma, 
                                   gravity = 10.**self.log_g, 
                                    kappa_IR = 10.**self.log_kappa_IR, 
                                    f_global = self.f_global,
                                    T_irr = self.T_irr)
        elif self.TP_type == 'custom_fixed':
            tempS, presS = self.TP_data['T[K]'], self.TP_data['P[bar]']
            csS = interp1d(presS[::-1], tempS[::-1], fill_value='extrapolate')
            pOut = gen_.P.copy() / 1E5
            tOut = csS(pOut)
            gen_.T = tOut
        elif self.TP_type == 'Madhusudhan_Seager':
            gen_.T = aut.madhusudhan_seager_TP(pressure_levels = gen_.P.copy()/1e5, # All input pressure values should be in bars for this (the function will convert to Pascals itself)
                                               log_Pset = 0., Tset = self.T_set, 
                                               alpha1 = self.alpha1, alpha2 = self.alpha2,
                                               log_P1 = self.log_P1, log_P2 = self.log_P2, 
                                               log_P3 = self.log_P3, beta = 0.5)
        
        elif self.TP_type == 'Bezier_4_nodes':
            gen_.T = aut.PTbez(logParr = np.log10(gen_.P.copy()/1e5),
                                Ps = [self.log_P3, self.log_P2, self.log_P1, self.log_P0],
                                Ts = [self.T3, self.T2, self.T1, self.T0]) 
                               # All input pressure values should be in bars for this (the function will convert to Pascals itself)

            
        gen_.profile(self.R_planet, self.log_g, self.P_ref) #Rp (Rj), log(g) cgs, Pref (log(bar))
        
        return gen_
    
    def get_TP_profile(self):
        """
        Return TP profile, as array of T [K] and P [bars].
        """
        gen_ = self.Genesis_instance
        if self.TP_type == 'Linear':
            # From Sid'e email and looking at set_T function, 
            # Order is (P1,T1),(P2,T2),[P0=,T0=], i.e. down to top. P1 must be greater than P2!
            gen_.set_T(self.P1, self.T1, self.P2, self.T2) # This part should have options to choose different kinds of TP profile.
        elif self.TP_type == 'Guillot':
            gen_.T = aut.guillot_TP(pressure_levels = gen_.P.copy()/1e5, # Pressure should be in bars for this 
                                   T_int = self.T_int, 
                                   T_eq = None, 
                                   gamma = 10.**self.log_gamma, 
                                   gravity = 10.**self.log_g, 
                                    kappa_IR = 10.**self.log_kappa_IR, 
                                    f_global = self.f_global,
                                    T_irr = self.T_irr)
        elif self.TP_type == 'custom_fixed':
            tempS, presS = self.TP_data['T[K]'], self.TP_data['P[bar]']
            csS = interp1d(presS[::-1], tempS[::-1], fill_value='extrapolate')
            pOut = gen_.P.copy() / 1E5
            tOut = csS(pOut)
            gen_.T = tOut
        elif self.TP_type == 'Madhusudhan_Seager':
            gen_.T = aut.madhusudhan_seager_TP(pressure_levels = gen_.P.copy()/1e5, 
                                               log_Pset = 0., Tset = self.T_set, 
                                               alpha1 = self.alpha1, alpha2 = self.alpha2,
                                               log_P1 = self.log_P1, log_P2 = self.log_P2, 
                                               log_P3 = self.log_P3, beta = 0.5)
            
        elif self.TP_type == 'Bezier_4_nodes':
            gen_.T = aut.PTbez(logParr = np.log10(gen_.P.copy()/1e5),
                                Ps = [self.log_P3, self.log_P2, self.log_P1, self.log_P0],
                                Ts = [self.T3, self.T2, self.T1, self.T0]) 
            
        return gen_.T, gen_.P.copy() / 1E5 

    
    @property
    def abundances_dict(self):
        
        # self.set_abundances() # this is necessary for the abundances to be converted to their power of 10 (since we are sampling in log10 of VMR)
        
        X = {}
        for sp in self.species:
            if sp not in ["h2", "he"]:
                X[sp] = np.full(len(self.gen.P), getattr(self, sp))
        X["he"] = np.full(len(self.gen.P), self.he)
        
        metals = np.full(len(self.gen.P), 0.)
        for sp in self.species:
            if sp != "h2":
                metals+=X[sp]
        
        X["h2"] = 1.0 - metals
        
        assert all(X["h2"] >= 0.) 
        
        return X # make sure that the abundance is not going negative!   
    
    def get_spectra(self):
        
        if self.method == "transmission":
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict, cl_P = self.cl_P)
            spec = self.gen.genesis(self.abundances_dict, cl_P = self.cl_P)
            spec /= ((self.R_star*6.96e8)**2.0)
            # spec = 1.-spec
            spec = -spec
        elif self.method == 'emission':
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict)
            spec = self.gen.genesis(self.abundances_dict)
            spec /= self.planck_lam_star(self.R_star, self.gen.lam, self.T_eff)
        
        return (10**9) * self.gen.lam, 10**self.log_fs * spec
    
    def convolve_spectra_to_instrument_resolution(self,model_spec_orig=None):
        ####### Convolve model to instrument resolution
        delwav_by_wav = 1/self.config['data'][self.inst]['resolution'] # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
        delwav_by_wav_model = 1./self.config['model']['R_power']   ### np.diff(model_wav)/model_wav[1:]
        
        ############ Convolve to instrument resolution 
        FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
        model_spec = convolve(model_spec_orig, Gaussian1DKernel(stddev=sig), boundary='extend')
        return model_spec
    
    def get_spectra_for_species(self, species_name = None ):
        """
        Method to get spectrum with contribution from only one of the species: co, ch4, or h2o, with the other two zeroed out. 
        """
        print('Species: ', species_name)
        for sp in ['co', 'ch4', 'h2o']:
            if sp != species_name: 
                print('Setting ', sp, 'to 1e-30')
                setattr(self, sp, 10.**-30.)

        # print('Computing model for:', species_name)
        # print(np.mean(self.abundances_dict[species_name]))
        if self.method == "transmission":
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict, cl_P = self.cl_P)
            spec = self.gen.genesis(self.abundances_dict, cl_P = self.cl_P)
            spec /= ((self.R_star*6.96e8)**2.0)
        elif self.method == 'emission':
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict)
            spec = self.gen.genesis(self.abundances_dict)
            spec /= self.planck_lam_star(self.R_star, self.gen.lam, self.T_eff)
        
        return (10**9) * self.gen.lam, 10**self.log_fs * spec
    
    def get_reprocessed_modelcube(self, model_spec = None, model_wav = None, datacube = None, datacube_detrended = None, data_wavsoln = None,
                                  pca_eigenvectors = None, colmask = None, post_pca_mask = None,
                                  phases = None, berv = None):
        """Given the datacubes (original and detrended) for a single order, 
        and the pca_eigenvectors, for a single value of Kp and Vsys, 
        compute the reprocessed modelcube corresponding to each exposure as those in the observations.

        :param datacube: _description_, defaults to None
        :type datacube: _type_, optional
        :param datacube_detrended: _description_, defaults to None
        :type datacube_detrended: _type_, optional
        :param data_wavsoln: _description_, defaults to None
        :type data_wavsoln: _type_, optional
        :param pca_eigenvectors: _description_, defaults to None
        :type pca_eigenvectors: _type_, optional
        :param colmask: _description_, defaults to None
        :type colmask: _type_, optional
        :param phases: _description_, defaults to None
        :type phases: _type_, optional
        :param berv: _description_, defaults to None
        :type berv: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        ### Check shapes of input 
        # print('datacube', datacube.shape)
        # print('datacube_detrended', datacube_detrended.shape)
        # print('data_wavsoln', data_wavsoln.shape)
        # print('pca_eigenvectors', pca_eigenvectors.shape)
        # print('colmask', colmask.shape)
        # print('phases', phases.shape) 
        # print('berv', berv.shape)
        
        ####### Convolve model to instrument resolution
        delwav_by_wav = 1/self.config['data'][self.inst]['resolution'] # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
        delwav_by_wav_model = 1./self.config['model']['R_power']
        FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
        model_spec = convolve(model_spec, Gaussian1DKernel(stddev=sig), boundary='extend')
        # import pdb
        # pdb.set_trace()
        ################ 
        nspec, nwav = datacube.shape[0], datacube.shape[1]
            # Initialize the reprocessed modelcube
        model_reprocess = np.empty((nspec, nwav))
        datamodel = np.empty((nspec, nwav))
        datamodel_fit = np.empty((nspec, nwav))
        datamodel_detrended = np.empty((nspec, nwav))
        
        model_spl = splrep(model_wav, model_spec)

        model_spec_shift_cube = np.empty((nspec, nwav))
        
        for it in range(nspec):
            RV = self.Kp * np.sin(2. * np.pi * (phases[it] + self.phase_offset)) + self.Vsys + berv[it]
            
            data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[:], velocity=-1. * RV)
            model_spec_shift_exp = splev(data_wavsoln_shift, model_spl)
            model_spec_shift_cube[it, :] = model_spec_shift_exp

        # Inject the model to the data
        datamodel = datacube * (1. + model_spec_shift_cube)

        datamodel_fit = stc.linear_regression(X=pca_eigenvectors,
                                                Y=datamodel)

        datamodel_detrended = datamodel/datamodel_fit - 1.

        # Zero out the same channels/columns as done for the datacube (including the ones post PCA)
        datamodel_detrended[:, post_pca_mask] = 0.
        # Also zero out the bad columns 
        datamodel_detrended[:, colmask] = 0.
        
        model_reprocess = datamodel_detrended - datacube_detrended
        # Zero out the model for channels you want to mask 
        model_reprocess[:, post_pca_mask] = 0.
        model_reprocess[:, colmask] = 0.

        avoid_mask = np.logical_or(post_pca_mask, colmask) 
        
        # import pdb
        # pdb.set_trace()
        return model_reprocess, avoid_mask # see line 684 in croc on how to use this output further 
        
                 
    def planck_lam_star(self, Rs, lam, T):
        """Compute the Blackbody flux from the star.

        :param Rs: Stellar radius, in terms of solar radius.
        :type Rs: float
        :param lam: Wavelength range, in micron.
        :type lam: array
        :param T: Effective stellar temperature.
        :type T: float
        :return: Blackbody flux of the star.
        :rtype: array
        """
        lam_5 = lam*lam*lam*lam*lam
        Bs = (2.0*sc.h*sc.c*sc.c)/(lam_5*(np.expm1((sc.h*sc.c)/(lam*sc.k*T))))
        return Bs*np.pi*Rs*Rs*6.957e8*6.957e8
    
    def logL_fast(self, theta, datadetrend_dd = None, order_inds = None):
        """Function to calculate the total logL for data from a single instrument and all dates as per the 
        initial specifications of the Model class. The separation of date and instruments here has been done to 
        allow flexibility in the dates and instruments you might want to include in a retrieval. 

        :param theta: Array of free parameter values. 
        :type theta: array
        :param data: An instance of the Data class defined above. 
        :type data: crocodile.Data()
        """
        # First set the parameters of the model based on the list of free parameters in the config file. 
        # The order of free params in the theta vector should be EXACTLY the same 
        # as that defined in the yaml file. Note that in python version >=3.7, dictionaries are by default always ordered.
        # data : should be an instance of the data class already defined outside 
        # start = time.time()
        #######################################################################
        ################# SET THE SAMPLED FREE PARAMETERS FOR THE MODEL #######
        for i, pname in enumerate(self.free_params_dict.keys()):
            if pname in self.species or pname in ['P1','P2']:
                setattr(self, pname, 10.**theta[i])
                # print('Setting ', pname, ': ', 10.**theta[i])
            else:
                setattr(self, pname, theta[i])
                # print('Setting ', pname, ': ', theta[i])
        #######################################################################
        #######################################################################
        
        ####### Test TP profile for this set of params, if it is not appropriate return very low logL value 
        
        
        
        ### Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        model_wav, model_spec_orig = self.get_spectra()
        # plt.figure()
        # plt.plot(model_wav, model_spec, 'r')
        # plt.show()
        
        # order_inds = data.get_use_order_inds(inst = self.inst, date = self.date)
        
        datelist = list(datadetrend_dd.keys()) 
        
        logL_per_date = np.empty(len(datelist))
        
        for dt, date in enumerate(datelist):
            
            logL_per_ord = np.empty(len(order_inds))
            
            for num_ind, ind in zip(range(len(order_inds)),order_inds):
                # First run the PCA detrending for this particular order # this particular step only needs doing once so do it outside! 
                # phases, berv, colmask, data_wavsoln, datacube, datacube_fit, datacube_detrended, pca_eigenvectors = data.pca_per_order(inst = self.inst, date = self.date, 
                #                                                                                 order_ind = ind)
                
                nspec = datadetrend_dd[date]['datacube'][ind, :, :].shape[0]
                
                model_reprocess, avoid_mask = self.get_reprocessed_modelcube(model_spec = model_spec_orig, model_wav = model_wav, 
                                                datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                                datacube_detrended = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                                data_wavsoln = datadetrend_dd[date]['data_wavsoln'][ind, :],
                                        pca_eigenvectors = datadetrend_dd[date]['pca_eigenvectors'][ind][:], 
                                        colmask = datadetrend_dd[date]['colmask'][ind, :],
                                        post_pca_mask = datadetrend_dd[date]['post_pca_mask'][ind, :],
                                        phases = datadetrend_dd[date]['phases'], berv = datadetrend_dd[date]['berv'])

                R_per_spec, C_per_spec, logL_per_spec = np.empty(nspec), np.empty(nspec), np.empty(nspec)
                # plt.figure()
                for it in range(nspec):
                    model_spec_flux_shift = model_reprocess[it, :]
                    # Mean subtract the model
                    model_spec_flux_shift = crocut.sub_mask_1D(model_spec_flux_shift, avoid_mask)
                                        
                    # R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, :],
                    #                                                                     model=model_spec_flux_shift)
                    ## The following where you mask both the data and model takes in the correct amount of data points (only those that actually contribute to the CCF finitely)
                    R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, ~avoid_mask],
                                                                                        model=model_spec_flux_shift[~avoid_mask])
                    
                    # if dt == 0 and it == 0:
                    #     plt.figure()
                    #     plt.plot(datadetrend_dd[date]['data_wavsoln'][ind, :], datadetrend_dd[date]['datacube_mean_sub'][ind, it, :], 'k', alpha = 0.1)
                    #     plt.plot(datadetrend_dd[date]['data_wavsoln'][ind, :], model_spec_flux_shift, 'r', alpha = 0.1)
                    #     # plt.ylim(-0.005, 0.005)
                    #     plt.savefig('/home/astro/phsprd/wasp122b/results/retrievals/scratch/order_'+str(ind)+'_data_model_inspect.png', dpi = 300, format = 'png', bbox_inches = 'tight')
                    #     plt.close()
                    # print('data wav', datadetrend_dd[date]['data_wavsoln'][ind, :])
                # print('model wav', model_wav)
                # plt.plot(model_wav, model_spec_orig, 'k', alpha = 0.5)
                # plt.title('logL: ' + str(np.sum(logL_per_spec)) + '\n Kp: ' + str(round(self.Kp,2)) + 'Vsys: ' + str(round(self.Vsys,2)) )
                # plt.axvline(x = 2287.467163054678)
                # plt.axvline(x = 2300.1590974811365)
                # plt.show()
                    # sg2 = crocut.fast_var(model_spec_flux_shift)
                
                # print('order: ',ind,'R: ', np.mean(R_per_spec), 'C: ', np.mean(C_per_spec),'logL: ', np.mean(logL_per_spec))
                ## probably using all orders, we don't need to be using all orders!! 
                logL_per_ord[num_ind] = np.sum(logL_per_spec) # np.dot(logL_per_spec, np.ones(nspec)) # Sum over all the orders
            
            logL_per_date[dt] = np.sum(logL_per_ord) # np.dot(logL_per_ord, np.ones(len(order_inds)))

        logL_total = np.sum(logL_per_date)
        # end = time.time()
        # print('One logL done in {:5} seconds'.format(end-start))
                
        return logL_total
    
    def get_ccf_trail_matrix(self, datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, plot = False, savedir = None):
        """
        For each date and each detector, take a model spectrum, and cross correlate with each exposure, 
        construct the CCF trail matrix and return. 
        """
        #######################################################################
        ################# SET THE SAMPLED FREE PARAMETERS FOR THE MODEL #######
        # for i, pname in enumerate(self.free_params_dict.keys()):
        #     if pname in self.species or pname in ['P1','P2','log_fs']:
        #         setattr(self, pname, 10.**theta[i])
        #         # print('Setting ', pname, ': ', 10.**theta[i])
        #     else:
        #         setattr(self, pname, theta[i])
        #         # print('Setting ', pname, ': ', theta[i])
        #######################################################################
        #######################################################################
        
        ### Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        model_wav, model_spec_orig = self.get_spectra()
        ####### Convolve model to instrument resolution
        delwav_by_wav = 1/self.config['data'][self.inst]['resolution'] # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
        delwav_by_wav_model = 1./self.config['model']['R_power']
        FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
        model_spec = convolve(model_spec_orig, Gaussian1DKernel(stddev=sig), boundary='extend')
        model_spl = splrep(model_wav, model_spec)
        # import pdb
        # pdb.set_trace()
        # plt.figure()
        # plt.plot(model_wav, model_spec, 'r')
        # plt.show()
        
        # order_inds = data.get_use_order_inds(inst = self.inst, date = self.date)
        
        ccf_trail_matrix_dd = {}
        
        for dt, date in tqdm(enumerate(datadetrend_dd.keys())):
            
            logL_per_ord = np.empty(len(order_inds))
            ccf_trail_matrix_dd[date] = {}

            for ind in tqdm(order_inds):
                
                # First run the PCA detrending for this particular order # this particular step only needs doing once so do it outside! 
                # phases, berv, colmask, data_wavsoln, datacube, datacube_fit, datacube_detrended, pca_eigenvectors = data.pca_per_order(inst = self.inst, date = self.date, 
                #                                                                                 order_ind = ind)
                
                nspec, nvel  = datadetrend_dd[date]['datacube'][ind, :, :].shape[0], len(Vsys_range)
                
                ccf_trail_matrix_dd[date][ind] = np.empty((nspec, nvel))
                
                for iVsys, Vsys in enumerate(Vsys_range):
                    for it in range(nspec):
                        RV = Vsys + datadetrend_dd[date]['berv'][it]
                        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=datadetrend_dd[date]['data_wavsoln'][ind, :], 
                                                                        velocity=-1. * RV)
                        model_spec_shift_exp = splev(data_wavsoln_shift, model_spl)
                        model_spec_shift_exp = model_spec_shift_exp - crocut.fast_mean(model_spec_shift_exp)
                        
                        ccf_trail_matrix_dd[date][ind][it, iVsys], _, _ = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, :],
                                                                                        model=model_spec_shift_exp)
        ####### Sum across all orders for each date 
        for date in datadetrend_dd.keys():
            ccf_trail_total = np.zeros((nspec,nvel))
            for ind in order_inds:
                ccf_trail_total+=ccf_trail_matrix_dd[date][ind]
            ccf_trail_matrix_dd[date]['total'] = ccf_trail_total
                        
        if plot:
            #### First plot individual dates and orders 
            for date in ccf_trail_matrix_dd.keys():
                plot_type = 'pcolormesh'
                subplot_num = len(ccf_trail_matrix_dd[date].keys()) ## Make a subplot for each detector
                
                fig, axx = plt.subplots(subplot_num, 1, figsize=(5, 5*subplot_num))

                plt.subplots_adjust(hspace=0.8)

                for axis_ind, ind in zip(range(subplot_num), order_inds):
                    hnd1 = crocut.subplot_cc_matrix(axis=axx[axis_ind],
                                                cc_matrix=ccf_trail_matrix_dd[date][ind],
                                                phases=datadetrend_dd[date]['phases'],
                                                velocity_shifts=Vsys_range,
                                                ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                                title= 'Date: '+ date +'; Detector: ' + str(ind) ,
                                                setxlabel=True, plot_type = plot_type)
                    fig.colorbar(hnd1, ax=axx[axis_ind])

                    # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
                    axx[axis_ind].set_ylabel(r'$\phi$')
                    axx[axis_ind].set_xlabel(r'V$_{rest}$ [km/s]')

                plt.savefig(savedir + 'ccf_trail_date-' + date + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
                plt.close()
            
                #### Plot the total trail matrix across all dates and detectors 
                plt.figure(figsize = (10,5))
                ax = plt.gca()
                hnd1 = crocut.subplot_cc_matrix(axis=ax,
                                cc_matrix=ccf_trail_matrix_dd[date]['total'],
                                phases=datadetrend_dd[date]['phases'],
                                velocity_shifts=Vsys_range,
                                ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                title= date+', Total' ,
                                setxlabel=True, plot_type = plot_type)
                # fig.colorbar(hnd1, ax)
                ax.set_ylabel(r'$\phi$')
                ax.set_xlabel(r'V$_{rest}$ [km/s]')
                plt.savefig(savedir + 'ccf_trail_total_'+date+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
            
        
        else:
            return ccf_trail_matrix_dd
    
    def compute_2D_KpVsys_map(self, theta_fit_dd = None, posterior = 'median', datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, Kp_range = None, savedir = None, exclude_species = None, species_info = None):
        """
        For a set of parameters inferred from the cornerplot theta_fit_dd, 
        compute the 2D cross-corerlation map for a range of Kp and Vsys.
        
        posterior: 'median', '+1sigma', '-1sigma'
        """
        ## Define the index of the theta_fit_dd to use depending on if you are doing the computation for median, +1sigma, or -1sigma values of the posterior. 
        ## If theta_fit_dd is None, then don't change anything and leave the parameters to the ones set originally when intializing the planet model instance.
        if theta_fit_dd is not None:
            if posterior == 'median':
                postind = 0
            elif posterior == '-1sigma':
                postind = 1
            elif posterior == '+1sigma':
                postind = 2
            
            
            
            #######################################################################
            ################# SET THE SAMPLED FREE PARAMETERS FOR THE MODEL #######
            for pname in theta_fit_dd.keys():
                if pname in self.species or pname in ['P1','P2']:
                    setattr(self, pname, 10.**theta_fit_dd[pname][postind])
                else:
                    setattr(self, pname, theta_fit_dd[pname][postind])
            #######################################################################
            #######################################################################
            ################# Due you want to zero out certain species to get the contribution of others? ########
            ########## For fix param condition (when theta_fit_dd is set to None), this happens outside. 
            if exclude_species is not None:
                for spnm in exclude_species:
                    setattr(self, spnm, 10.**-30.)
        
        
        
        nKp, nVsys = len(Kp_range), len(Vsys_range)

        
        
        ### Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        model_wav, model_spec_orig = self.get_spectra()
        # plt.figure()
        # plt.plot(model_wav, model_spec, 'r')
        # plt.show()
        
        # order_inds = data.get_use_order_inds(inst = self.inst, date = self.date)
        
        datelist = list(datadetrend_dd.keys()) 
        
        R_dd, C_dd, logL_dd = {}, {}, {}
        
        for dt, date in enumerate(datelist):
            print('Date: ', date)
            R_dd[date], C_dd[date], logL_dd[date] = {}, {}, {}
            
            for ind in tqdm(order_inds):
                
                R_dd[date][ind], C_dd[date][ind], logL_dd[date][ind] = np.empty((nKp, nVsys)), np.empty((nKp, nVsys)), np.empty((nKp, nVsys))
                
                for iKp, Kp_val in enumerate(Kp_range): 
                    for iVsys, Vsys_val in enumerate(Vsys_range):
                        ## Set the Kp and Vsys value 
                        setattr(self, 'Kp', Kp_val)
                        setattr(self, 'Vsys', Vsys_val) 
                        nspec = datadetrend_dd[date]['datacube'][ind, :, :].shape[0]
                        
                        model_reprocess, avoid_mask = self.get_reprocessed_modelcube(model_spec = model_spec_orig, model_wav = model_wav, 
                                                        datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                                        datacube_detrended = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                                        data_wavsoln = datadetrend_dd[date]['data_wavsoln'][ind, :],
                                                pca_eigenvectors = datadetrend_dd[date]['pca_eigenvectors'][ind][:], 
                                                colmask = datadetrend_dd[date]['colmask'][ind, :],
                                                post_pca_mask = datadetrend_dd[date]['post_pca_mask'][ind, :],
                                                phases = datadetrend_dd[date]['phases'], berv = datadetrend_dd[date]['berv'])
                        
                        R_per_spec, C_per_spec, logL_per_spec = np.empty(nspec), np.empty(nspec), np.empty(nspec)

                        for it in range(nspec):
                            model_spec_flux_shift = model_reprocess[it, :]
                            # Mean subtract the model
                            # model_spec_flux_shift = model_spec_flux_shift - crocut.fast_mean(model_spec_flux_shift)
                            model_spec_flux_shift = crocut.sub_mask_1D(model_spec_flux_shift, avoid_mask)
                                                
                            # R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, :],
                            #                                                                     model=model_spec_flux_shift)
                            ## The following where you mask both the data and model takes in the correct amount of data points (only those that actually contribute to the CCF finitely)
                            R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, ~avoid_mask],
                                                                                                model=model_spec_flux_shift[~avoid_mask])
                            
                            
                            
                        R_dd[date][ind][iKp, iVsys], C_dd[date][ind][iKp, iVsys], logL_dd[date][ind][iKp, iVsys] = np.sum(R_per_spec), np.sum(C_per_spec), np.sum(logL_per_spec) 

                    
        ## Sum across each detector for all dates 
        logL_dd_per_date, logL_sigma_dd_per_date = {} , {}
        C_dd_per_date = {}
        
        ## For each date, first sum across indices 
        for date in logL_dd.keys():
            logL_per_date = np.array([logL_dd[date][ind] for ind in order_inds])
            logL_dd_per_date[date] = np.sum(logL_per_date, axis = 0)
            
            C_per_date = np.array([C_dd[date][ind] for ind in order_inds])
            C_dd_per_date[date] = np.sum(C_per_date, axis = 0)

            logL_sigma_dd_per_date[date] = crocut.get_sigma_contours(logL_KpVsys=logL_dd_per_date[date], dof=2) 
                
        ## Sum across all dates 
        logL_all_dates = np.array( [logL_dd_per_date[date] for date in logL_dd_per_date.keys()] )
        logL_total = np.sum(logL_all_dates, axis = 0)
        
        C_all_dates = np.array( [C_dd_per_date[date] for date in C_dd_per_date.keys()] )
        C_total =  np.sum(C_all_dates, axis = 0)
        
        logL_total_sigma = crocut.get_sigma_contours(logL_KpVsys=logL_total, dof=2) 
        
         ## Save the KpVsys matrix for this model along with all the other relevant info
        KpVsys_save = {}
        KpVsys_save['Kp_range'] = Kp_range
        KpVsys_save['Vsys_range'] = Vsys_range
        
        KpVsys_save['R_dd'] = R_dd
        KpVsys_save['C_dd'] = C_dd
        KpVsys_save['logL_dd'] = logL_dd
        
        KpVsys_save['total'] = {}
        KpVsys_save['all_dates'] = {}
        KpVsys_save['all_dates']['logL'] = logL_dd_per_date
        KpVsys_save['all_dates']['logL_sigma'] = logL_sigma_dd_per_date
        KpVsys_save['all_dates']['cc'] = C_dd_per_date
        
        KpVsys_save['total']['logL'] = logL_total
        KpVsys_save['total']['logL_sigma'] = logL_total_sigma
        KpVsys_save['total']['cc'] = C_total
        
        if species_info is None:
            np.save(savedir + 'KpVsys_dict_' + posterior, KpVsys_save)
        else:
            np.save(savedir + 'KpVsys_dict_' + posterior + '_' + species_info, KpVsys_save)
        
        return KpVsys_save
    
    def compute_2D_KpVsys_map_fast_without_model_reprocess(self, theta_fit_dd = None, posterior = None, 
                                                           datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, Kp_range = None, savedir = None, exclude_species = None, 
                             species_info = None, vel_window = None):
        """
        For a set of parameters inferred from the cornerplot theta_fit_dd, or for the initial parameters (by setting theta_fit_dd to None), 
        compute the 2D cross-corerlation map for a range of Kp and Vsys. This does 
        
        posterior: 'median', '+1sigma', '-1sigma'
        """
        ## Define the index of the theta_fit_dd to use depending on if you are doing the computation for median, +1sigma, or -1sigma values of the posterior. 
        ## If theta_fit_dd is None, then don't change anything and leave the parameters to the ones set originally when intializing the planet model instance.
        if theta_fit_dd is not None:
            if posterior == 'median':
                postind = 0
            elif posterior == '-1sigma':
                postind = 1
            elif posterior == '+1sigma':
                postind = 2
            
            #######################################################################
            ################# SET THE SAMPLED FREE PARAMETERS FOR THE MODEL #######
            for pname in theta_fit_dd.keys():
                if pname in self.species or pname in ['P1','P2']:
                    setattr(self, pname, 10.**theta_fit_dd[pname][postind])
                else:
                    setattr(self, pname, theta_fit_dd[pname][postind])
            #######################################################################
            #######################################################################

        nKp, nVsys = len(Kp_range), len(Vsys_range)
        ################# Due you want to zero out certain species to get the contribution of others? ######## NOT IMPLMENTED YET : 13-06-2024
        if exclude_species is not None:
            for spnm in exclude_species:
                setattr(self, spnm, 10.**-30.)
        
        
        ### Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        model_wav, model_spec_orig = self.get_spectra()
        print('Model calculation done, convolving to instrument resolution ...')
         ####### Convolve model to instrument resolution
        delwav_by_wav = 1/self.config['data'][self.inst]['resolution'] # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
        delwav_by_wav_model = 1./self.config['model']['R_power']   ### np.diff(model_wav)/model_wav[1:]
        
        ############ Convolve to instrument resolution 
        FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
        model_spec = convolve(model_spec_orig, Gaussian1DKernel(stddev=sig), boundary='extend')
        model_spl = splrep(model_wav, model_spec)
        # plt.figure()
        # plt.plot(model_wav, model_spec, 'r')
        # plt.show()        
        datelist = list(datadetrend_dd.keys()) 
        
        #####################################################################################################################################################################
        #####################################################################################################################################################################
        ########### Loop over dates, and get trail matrix (summed across all detectors) and then convert that to KpVsys maps for that date by shift by Kp and interpolation 
        #####################################################################################################################################################################
        #####################################################################################################################################################################
        CC_matrix_all_dates , logL_matrix_all_dates = {}, {}
        for dt, date in enumerate(datelist):
            
            datacube_mean_sub = datadetrend_dd[date]['datacube_mean_sub']
            data_wavsoln = datadetrend_dd[date]['data_wavsoln']
            phases = datadetrend_dd[date]['phases']
            berv = datadetrend_dd[date]['berv']
            nspec = datacube_mean_sub.shape[1]
            
            cc_matrix_all_orders, logL_matrix_all_orders = np.zeros((len(order_inds), nspec, nVsys)), np.zeros((len(order_inds), nspec, nVsys))
            ## Loop over orders
            
            # for ind in tqdm(order_inds):
            ### THIS NEEDS FIXING 
            # ind_map = {}
            # for i_ind, ind in enumerate(order_inds):
                
            for i_ind, ind in tqdm(enumerate(order_inds)): 
                avoid_mask = np.logical_or(datadetrend_dd[date]['colmask'][ind, :],datadetrend_dd[date]['post_pca_mask'][ind, :])
                ## Loop over time 
                for it in tqdm(range(nspec)):
                    ## Loop over velocities 
                    for iv, vel in enumerate(Vsys_range):
                        # First Doppler shift the data wavelength solution to -vel
                        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[ind,:], velocity=-1. * vel)
                        # Evaluate the model to the data_wavsoln_shifted by -vel,
                        # Effectively Doppler shifting the model by +vel
                        model_spec_flux_shift = splev(data_wavsoln_shift, model_spl)
                        # Subtract the mean from the model
                        model_spec_flux_shift = model_spec_flux_shift - crocut.fast_mean(model_spec_flux_shift)
                        # Compute the cross correlation value between the shifted model and the data
                        
                        if it == 0 and iv == 0:
                            plt.figure(figsize = (10,8))
                            plt.plot(data_wavsoln[ind,~avoid_mask], datacube_mean_sub[ind,it,~avoid_mask], color = 'k', label = 'data')
                            plt.plot(data_wavsoln_shift[~avoid_mask], model_spec_flux_shift[~avoid_mask], color = 'r', label = 'model')
                            plt.legend()
                            plt.savefig(savedir + 'data_model_comp_order_'+str(ind)+'.png', format='png', dpi=300, bbox_inches='tight')    
                        plt.close('all')
                        _, cc_matrix_all_orders[i_ind,it,iv], logL_matrix_all_orders[i_ind,it,iv] = crocut.fast_cross_corr(data=datacube_mean_sub[ind,it,~avoid_mask], 
                                                                                                                     model=model_spec_flux_shift[~avoid_mask])
                        
            CC_matrix_all_dates[dt] , logL_matrix_all_dates[dt] = np.sum(cc_matrix_all_orders, axis = 0), np.sum(logL_matrix_all_orders, axis = 0)              

        # CC_matrix_total, logL_matrix_total = np.zeros((nspec, nVsys)), np.zeros((nspec, nVsys))
        # import pdb
        # pdb.set_trace()
        # for dt, date in enumerate(datelist):
        #     CC_matrix_total+=CC_matrix_all_dates[dt]
        #     logL_matrix_total+=logL_matrix_all_dates[dt]
        
        ###### Plot the CC trail matrix to test 
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (10,8))
            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=CC_matrix_all_dates[dt],
                                        phases=datadetrend_dd[date]['phases'],
                                        velocity_shifts=Vsys_range,
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total ; Date: '+ date ,
                                        setxlabel=True, plot_type = 'pcolormesh')
            fig.colorbar(hnd1, ax=axx)

            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'$\phi$')
            axx.set_xlabel(r'V$_{rest}$ [km/s]')
            plt.savefig(savedir + 'ccf_total_trail_matrix_fast_date-' + date + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
        ###### Plot the logL trail matrix to test
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (10,8))
            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=logL_matrix_all_dates[dt],
                                        phases=datadetrend_dd[date]['phases'],
                                        velocity_shifts=Vsys_range,
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total ; Date: '+ date ,
                                        setxlabel=True, plot_type = 'pcolormesh')
            fig.colorbar(hnd1, ax=axx)

            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'$\phi$')
            axx.set_xlabel(r'V$_{rest}$ [km/s]')
            plt.savefig(savedir + 'logL_total_trail_matrix_fast_date-' + date + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
        
        #####################################################################################################################################################################
        #####################################################################################################################################################################
        ########### Shift rows in trail matrix by Kp values and compute the KpVsys maps 
        #####################################################################################################################################################################
        #####################################################################################################################################################################
        
        CC_KpVsys_total, logL_KpVsys_total = np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
        ## Start loop over dates 
        for dt, date in enumerate(datelist):
            CC_KpVsys, logL_KpVsys = np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
            phases = datadetrend_dd[date]['phases']
            nspec = len(phases)
            for iKp, Kp in enumerate(Kp_range):
                CC_matrix_shifted, logL_matrix_shifted = np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
                for it in range(nspec):
                    Vp = Kp * np.sin(2. * np.pi * phases[it])
                    print("NOT INCLUDING BERV")
                    Vsys_shifted = Vsys_range + Vp  + berv[it] 
                    # print('Vp:', Vp, 'BERV: ', berv[it])
                    # print('Vsys_shifted ', max(Vsys_shifted), min(Vsys_shifted) )
                    func_CC = interpolate.interp1d(Vsys_range, CC_matrix_all_dates[dt][it, :]) 
                    func_logL = interpolate.interp1d(Vsys_range, logL_matrix_all_dates[dt][it, :])
                    
                    CC_matrix_shifted[it,:] = func_CC(Vsys_shifted[vel_window[0]:vel_window[1]])
                    logL_matrix_shifted[it,:] = func_logL(Vsys_shifted[vel_window[0]:vel_window[1]])
                
                CC_KpVsys[iKp,:], logL_KpVsys[iKp,:] = np.sum(CC_matrix_shifted, axis = 0), np.sum(logL_matrix_shifted, axis = 0)
            CC_KpVsys_total+=CC_KpVsys
            logL_KpVsys_total+=logL_KpVsys
            
        KpVsys_save = {}
        KpVsys_save['logL'] = logL_KpVsys_total
        KpVsys_save['cc'] = CC_KpVsys_total
        KpVsys_save['Kp_range'] = Kp_range
        KpVsys_save['Vsys_range'] = Vsys_range
        KpVsys_save['vel_window'] = vel_window
        KpVsys_save['Vsys_range_windowed'] = Vsys_range[vel_window[0]:vel_window[1]]
        
        if species_info is None:
            np.save(savedir + 'KpVsys_fast_no_model_reprocess_dict.npy', KpVsys_save)
        else:
            np.save(savedir + 'KpVsys_fast_no_model_reprocess_dict_' + species_info + '.npy', KpVsys_save)
        
        ####### Plot and save 
        subplot_num = 2
        fig, axx = plt.subplots(subplot_num, 1, figsize=(8, 8*subplot_num))
        plt.subplots_adjust(hspace=0.6)

        hnd1 = crocut.subplot_cc_matrix(axis=axx[0],
                                    cc_matrix=KpVsys_save['cc'],
                                    phases=Kp_range,
                                    velocity_shifts=KpVsys_save['Vsys_range_windowed'],
                                    ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                    title= 'Total CC' ,
                                    setxlabel=True, plot_type = 'pcolormesh')
        fig.colorbar(hnd1, ax=axx[0])

        hnd1 = crocut.subplot_cc_matrix(axis=axx[1],
                                    cc_matrix=KpVsys_save['logL'],
                                    phases=Kp_range,
                                    velocity_shifts=KpVsys_save['Vsys_range_windowed'],
                                    ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                    title= 'Total logL' ,
                                    setxlabel=True, plot_type = 'pcolormesh')
        fig.colorbar(hnd1, ax=axx[1])

        if theta_fit_dd is not None:
            for ip in [0,1]:
                axx[ip].vlines(x=theta_fit_dd['Vsys'][ind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][ind]-5., color='k', linestyle='dashed')
                axx[ip].vlines(x=theta_fit_dd['Vsys'][ind], ymin=theta_fit_dd['Kp'][ind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                axx[ip].hlines(y=theta_fit_dd['Kp'][ind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][ind]-5., color='k', linestyle='dashed')
                axx[ip].hlines(y=theta_fit_dd['Kp'][ind], xmin=theta_fit_dd['Vsys'][ind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')
        else:
            for ip in [0,1]:
                axx[ip].vlines(x=self.Vsys_pred, ymin=KpVsys_save['Kp_range'][0], ymax=self.Kp_pred-5., color='w', linestyle='dashed')
                axx[ip].vlines(x=self.Vsys_pred, ymin=self.Kp_pred+5., ymax=KpVsys_save['Kp_range'][-1], color='w', linestyle='dashed')

                axx[ip].hlines(y=self.Kp_pred, xmin=KpVsys_save['Vsys_range_windowed'][0], xmax=self.Vsys_pred-5., color='w', linestyle='dashed')
                axx[ip].hlines(y=self.Kp_pred, xmin=self.Vsys_pred+5., xmax=KpVsys_save['Vsys_range_windowed'][-1], color='w', linestyle='dashed')

            

        axx[0].set_ylabel(r'K$_{P}$ [km/s]')
        axx[0].set_xlabel(r'V$_{rest}$ [km/s]')
        axx[1].set_ylabel(r'K$_{P}$ [km/s]')
        axx[1].set_xlabel(r'V$_{rest}$ [km/s]')
        
        if species_info is None:
            plt.savefig(savedir + 'KpVsys_fast_no_model_reprocess.png', format='png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savedir + 'KpVsys_fast_no_model_reprocess' + species_info + '.png', format='png', dpi=300, bbox_inches='tight')

                    
    
    ##############################################################################################################
    ##############################################################################################################
    ## Function to plot KpVsys maps ##############################################################################
    ##############################################################################################################
    ##############################################################################################################
    def plot_KpVsys_maps(self, KpVsys_save = None, posterior = 'median', theta_fit_dd = None, savedir = None, species_info = None):
        
        if KpVsys_save is None:
            if species_info is None:
                KpVsys_save = np.load(savedir + 'KpVsys_dict_' + posterior + '.npy', allow_pickle = True).item()
            else:
                KpVsys_save = np.load(savedir + 'KpVsys_dict_' + posterior + '_' + species_info + '.npy', allow_pickle = True).item()
        
        datelist = list(KpVsys_save['all_dates']['logL'].keys())
        print(datelist)
        
        if theta_fit_dd is not None:
        ## Define the index of the theta_fit_dd to use depending on if you are doing the computation for median, +1sigma, or -1sigma values of the posterior. 
            if posterior == 'median':
                ind = 0
            elif posterior == '-1sigma':
                ind = 1
            elif posterior == '+1sigma':
                ind = 2
        ############### Plot and save the individual date cc matrices first ##############
        for mk in ['cc', 'logL','logL_sigma']:

            if mk == 'logL_sigma':
                plot_type = 'contourf'
            else:
                plot_type = 'pcolormesh'

            subplot_num = len(datelist)
            fig, axx = plt.subplots(subplot_num, 1, figsize=(12, 5*subplot_num))
            print(subplot_num)
            plt.subplots_adjust(hspace=0.8)

            if len(datelist) == 1:
                axx = [axx]
            
            for dd,date in enumerate(datelist):
                
                hnd1 = crocut.subplot_cc_matrix(axis=axx[dd],
                                            cc_matrix=KpVsys_save['all_dates'][mk][date],
                                            phases=KpVsys_save['Kp_range'],
                                            velocity_shifts=KpVsys_save['Vsys_range'],
                                            ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                            title= 'Date: '+ date,
                                            setxlabel=True, plot_type = plot_type)
                fig.colorbar(hnd1, ax=axx[dd])
                if theta_fit_dd is not None:
                    axx[dd].vlines(x=theta_fit_dd['Vsys'][ind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][ind]-5., color='k', linestyle='dashed')
                    axx[dd].vlines(x=theta_fit_dd['Vsys'][ind], ymin=theta_fit_dd['Kp'][ind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                    axx[dd].hlines(y=theta_fit_dd['Kp'][ind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][ind]-5., color='k', linestyle='dashed')
                    axx[dd].hlines(y=theta_fit_dd['Kp'][ind], xmin=theta_fit_dd['Vsys'][ind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')

                # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
                axx[dd].set_ylabel(r'K$_{P}$ [km/s]')
                axx[dd].set_xlabel(r'V$_{rest}$ [km/s]')

                # axx[dd].set_xlim(xmin = velocity_shifts_win[0], xmax = velocity_shifts_win[-1])
                # axx[dd].set_ylim(ymin=Kp_range[0], ymax=Kp_range[-1])

            if species_info is None:
                plt.savefig(savedir + 'all_dates_' + mk + '_'+posterior+'_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savedir + 'all_dates_' + mk + '_'+posterior+'_'+species_info+'_.pdf', format='pdf', dpi=300, bbox_inches='tight')

        ############### Plot the total cc matrices ##############
        for mk in ['cc', 'logL','logL_sigma']:
            if mk == 'logL_sigma':
                plot_type = 'contourf'
            else:
                plot_type = 'pcolormesh'

            subplot_num = 1
            fig, axx = plt.subplots(subplot_num, 1, figsize=(12, 8))

            plt.subplots_adjust(hspace=0.8)

            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=KpVsys_save['total'][mk],
                                        phases=KpVsys_save['Kp_range'],
                                        velocity_shifts=KpVsys_save['Vsys_range'],
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total '+ mk ,
                                        setxlabel=True, plot_type = plot_type)
            fig.colorbar(hnd1, ax=axx)

            if theta_fit_dd is not None:
            
                axx.vlines(x=theta_fit_dd['Vsys'][ind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][ind]-5., color='k', linestyle='dashed')
                axx.vlines(x=theta_fit_dd['Vsys'][ind], ymin=theta_fit_dd['Kp'][ind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                axx.hlines(y=theta_fit_dd['Kp'][ind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][ind]-5., color='k', linestyle='dashed')
                axx.hlines(y=theta_fit_dd['Kp'][ind], xmin=theta_fit_dd['Vsys'][ind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')


            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'K$_{P}$ [km/s]')
            axx.set_xlabel(r'V$_{rest}$ [km/s]')

            # axx.set_xlim(xmin=velocity_shifts_win[0], xmax=velocity_shifts_win[-1])
            # axx.set_ylim(ymin=Kp_range[0], ymax=Kp_range[-1])
            
            if species_info is None:
                plt.savefig(savedir + 'total_'+ mk + '_'+ posterior+ '_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savedir + 'total_'+ mk + '_'+ posterior+ '_' + species_info + '_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            
        plt.close('all')