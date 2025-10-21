import numpy as np
import yaml
import pyfastchem
import sys
# sys.path.insert(0, "/home/astro/phsprd/code/genesis/code")  ## Add path to genesis in your machine (point to the code subdirectory which contains genesis.py)
# import genesis
from genesis import genesis
from astropy.io import fits
import scipy.constants as sc
from . import stellcorrection_utils as stc
from . import cross_correlation_utils as crocut
# from scipy.interpolate import splev, splrep
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from astropy.convolution import Gaussian1DKernel, convolve
from scipy.signal import fftconvolve
from . import astro_utils as aut
from tqdm import tqdm 
from astropy.io import ascii as ascii_reader
import copy
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


# from astropy import units as un

FAST_CHEM_DIR = '/home/astro/phsprd/code/crocodel/fastchem_inputs/'


class Model:
    """Model class to for modeling and performing cross-correlation and log-likelihood computations for 
    typical high-resolution cross-correlation spectroscopy dataset. This should be define on a 'per instrument' basis, with the instrument name 
    specified in the kwarg 'inst' when instantiating the class, and the info for the respective instrument extracted from the croc_config.yaml file. Use one croc_config.yaml file to 
    store info for all instruments, and use one model class for one instrument.
    """
    def __init__(self, *args, **kwargs):
        
        ##### Define some constants 
        self.vel_c_SI = 299792458.0
        self.k_B_cgs = 1.380649e-16
        
        with open(kwargs.pop('config')) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Instrument name        
        self.inst = kwargs.pop('inst') 
        
        # Method ('transmission' or 'emission')
        self.method = self.config['data'][self.inst]['method']
        
        # Type of TP profile parametrization
        self.TP_type = self.config['model']['TP_type']
        
        # Stellar properties
        self.R_star = self.config['model']['R_star'] # Stellar radius, in terms of R_Sun
        self.T_eff = self.config['model']['T_eff'] # Stellar effective temperature, in K        
        self.vsini = self.config['model']['vsini']
        
        # Other model properties
        self.P_min = self.config['model']['P_min'] # Minimum pressure level for model calculation, in bars 
        self.P_max = self.config['model']['P_max'] # Maximum pressure level for model calculation, in bars 
        self.N_layers = self.config['model']['N_layers'] # Number of pressure layers in the atmosphere 
        self.lam_min = self.config['model']['lam_min'] # Minimum wavelength for model calculation, in microns
        self.lam_max = self.config['model']['lam_max'] # Maximum wavelength for model calculation, in microns
        self.resolving_power = self.config['model']['R_power'] # Resolving power for the model calculation (use 250000 which will later be convolved down)
        self.spacing = self.config['model']['spacing'] # Wavelength grid spacing, use 'R' for constant resolving power
        
        self.fix_MMW = self.config['model']['fix_MMW']
        self.MMW_value = self.config['model']['MMW_value']
        
        self.chemistry = self.config['model']['chemistry']
        
        # Load the names of all the absorbing species to be included in the model
        self.species = np.array(list(self.config['model']['abundances'].keys()))
        self.species_name_fastchem = self.config['model']['species_name_fastchem']
        if self.config['model']['include_cia'] is not None:
            self.include_cia = self.config['model']['include_cia']
        else:
            self.include_cia = True
        
        if self.chemistry == 'eq_chem':
            #create a FastChem object
            #it needs the locations of the element abundance and equilibrium constants files
            self.include_condensation = self.config['model']['include_condensation']
            
            
            if self.include_condensation:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR + 'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR +'input/logK/logK.dat',
                FAST_CHEM_DIR +'input/logK/logK_condensates.dat',
                1)
            else:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR +'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR +'input/logK/logK.dat',
                1)
            
            # Make a copy of the solar abundances from FastChem
            self.solar_abundances = np.array(self.fastchem.getElementAbundances())
            
            
            self.logZ_planet = self.config['model']['logZ_planet']
            self.C_to_O = self.config['model']['C_to_O']
            self.use_C_to_O = self.config['model']['use_C_to_O']
            
            self.index_C = self.fastchem.getElementIndex('C')
            self.index_O = self.fastchem.getElementIndex('O')
            
            # Create the input and output structures for FastChem
            self.input_data = pyfastchem.FastChemInput()
            self.output_data = pyfastchem.FastChemOutput()
            if self.include_condensation:
                self.input_data.equilibrium_condensation = True
            else:
                self.input_data.equilibrium_condensation = False
            
            self.species_fastchem_indices = {}
            for sp in self.species: ## Only do this for the species we are including in the model
                # if sp != "h_minus":
                self.species_fastchem_indices[sp] = self.fastchem.getGasSpeciesIndex(self.species_name_fastchem[sp])
            ## "h2" is usually not in the free abundances so get its index as well separately.
            self.species_fastchem_indices["h2"] = self.fastchem.getGasSpeciesIndex("H2")
            
        elif self.chemistry == 'eq_chem_ER':
            #create a FastChem object
            #it needs the locations of the element abundance and equilibrium constants files
            self.include_condensation = self.config['model']['include_condensation']
            
            if self.include_condensation:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR + 'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR +'input/logK/logK.dat',
                FAST_CHEM_DIR +'input/logK/logK_condensates.dat',
                1)
            else:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR +'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR +'input/logK/logK.dat',
                1)
            
            # Make a copy of the solar abundances from FastChem
            self.solar_abundances = np.array(self.fastchem.getElementAbundances())
            self.elemental_ratios = self.config['model']['elemental_ratios']
            self.log10_O_to_H_by_O_to_H_solar = self.elemental_ratios['log10_O_to_H_by_O_to_H_solar']
            self.log10_C_to_H_by_C_to_H_solar = self.elemental_ratios['log10_C_to_H_by_C_to_H_solar']
            self.log10_R_to_H_by_R_to_H_solar = self.elemental_ratios['log10_R_to_H_by_R_to_H_solar']
            self.refractories = self.config['model']['refractories']
            self.use_C_to_O = False ## By default set for False ALWAYS for this case.
            
            self.index_C = self.fastchem.getElementIndex('C')
            self.index_O = self.fastchem.getElementIndex('O')
            self.index_H = self.fastchem.getElementIndex('H')
            
            # for refrac in self.refractories:
            #     refrac_fastchem_name = self.species_name_fastchem[refrac]
            #     ind_refrac = self.fastchem.getElementIndex(refrac_fastchem_name)
            #     setattr(self, 'index_' + refrac, ind_refrac)
            # element_abundances = np.copy(self.solar_abundances)
            # self.C_to_H_solar = element_abundances[self.index_C]/element_abundances[self.index_H]
            # self.O_to_H_solar = element_abundances[self.index_O]/element_abundances[self.index_H]
            
            # self.R_total_solar = 0
            for refrac in self.refractories:
                refrac_fastchem_name = self.species_name_fastchem[refrac]
                ind_refrac = self.fastchem.getElementIndex(refrac_fastchem_name)
                setattr(self, 'index_' + refrac, ind_refrac)
                # self.R_total_solar =+ element_abundances[ind_refrac]
            # self.R_to_H_solar = self.R_total_solar/element_abundances[self.index_H]
            
            # Create the input and output structures for FastChem
            self.input_data = pyfastchem.FastChemInput()
            self.output_data = pyfastchem.FastChemOutput()
            if self.include_condensation:
                self.input_data.equilibrium_condensation = True
            else:
                self.input_data.equilibrium_condensation = False
            
            self.species_fastchem_indices = {}
            for sp in self.species: ## Only do this for the species we are including in the model
                # if sp != "h_minus":
                self.species_fastchem_indices[sp] = self.fastchem.getGasSpeciesIndex(self.species_name_fastchem[sp])
            ## "h2" is usually not in the free abundances so get its index as well separately.
            self.species_fastchem_indices["h2"] = self.fastchem.getGasSpeciesIndex("H2")

        elif self.chemistry == 'free_chem_with_dissoc':
            self.sp_dissoc_list  = self.config['model']['sp_dissoc_list']
            for sp in self.sp_dissoc_list:
                setattr(self, 'alpha_'+ sp , self.config['model']['sp_dissoc_params']['alpha_'+sp])
                setattr(self, 'log10_beta_'+ sp , self.config['model']['sp_dissoc_params']['log10_beta_'+sp])
                setattr(self, 'gamma_'+ sp , self.config['model']['sp_dissoc_params']['gamma_'+sp])
                
        
        if self.TP_type in ['Linear', 'Linear_force_inverted', 'Linear_force_non_inverted']:
            self.P2 = 10.**self.config['model']['P2']
            self.T2 = self.config['model']['T2']
            self.P1 = 10.**self.config['model']['P1']
            self.T1 = self.config['model']['T1']
        elif self.TP_type == 'Linear_3_point':
            self.P2 = 10.**self.config['model']['P2']
            self.T2 = self.config['model']['T2']
            self.P1 = 10.**self.config['model']['P1']
            self.T1 = self.config['model']['T1']
            self.P0 = 10.**self.config['model']['P0']
            self.T0 = self.config['model']['T0']
            
            
        elif self.TP_type == 'Guillot':
            self.T_int= self.config['model']['T_int']
            self.T_irr= self.config['model']['T_irr']
            self.log_gamma= self.config['model']['log_gamma']
            self.log_kappa_IR= self.config['model']['log_kappa_IR']
            self.f_global= self.config['model']['f_global'] # Setting this to one let's it be folded into T_eq
            
        elif self.TP_type == 'custom_fixed':
            # self.TP_data = ascii_reader.read(self.config['model']['TP_path'])
            self.TP_data = np.loadtxt(self.config['model']['TP_path'], skiprows=0, unpack=True)
            
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
        
        elif self.TP_type == 'Bezier_5_nodes': ## In order of increasing pressures (or going from top to down in altitude) : P3,P2,P1,P0 
            self.T0 = self.config['model']['T0']
            self.log_P0 = self.config['model']['log_P0']
            self.T1 = self.config['model']['T1']
            self.log_P1 = self.config['model']['log_P1']
            self.T2 = self.config['model']['T2']
            self.log_P2 = self.config['model']['log_P2']
            self.T3 = self.config['model']['T3']
            self.log_P3 = self.config['model']['log_P3']
            self.T4 = self.config['model']['T4']
            self.log_P4 = self.config['model']['log_P4']
            
        elif self.TP_type == 'Bezier_6_nodes':
            self.T0 = self.config['model']['T0']
            self.log_P0 = self.config['model']['log_P0']
            self.T1 = self.config['model']['T1']
            self.log_P1 = self.config['model']['log_P1']
            self.T2 = self.config['model']['T2']
            self.log_P2 = self.config['model']['log_P2']
            self.T3 = self.config['model']['T3']
            self.log_P3 = self.config['model']['log_P3']
            self.T4 = self.config['model']['T4']
            self.log_P4 = self.config['model']['log_P4']
            self.T5 = self.config['model']['T5']
            self.log_P5 = self.config['model']['log_P5']
            
        
        # Planet properties 
        self.R_planet= self.config['model']['R_planet'] # Radius of the planet, in terms of R_Jup
        self.vsini_planet = self.config['model']['vsini_planet']
        self.log_g= self.config['model']['log_g'] # Surface gravity, log_g [cgs]
        self.P_ref= self.config['model']['P_ref'] # Reference pressure, in log10 bars
        self.cl_P = self.config['model']['cl_P'] # log10(cloud_pressure) in bars 
        self.log_fs = self.config['model']['log_fs'] # model scale factor 
        self.phase_offset = self.config['model']['phase_offset']
        self.Kp = self.config['model']['Kp'] # Current value of Kp, in km/s
        self.Vsys = self.config['model']['Vsys'] # Current value of Vsys, in km/s
        
        
        self.Kp_pred = self.Kp # Expected value of Kp, in km/s
        self.Vsys_pred = self.Vsys # Expected value of Vsys, in km/s

        self.free_params_dict = self.config['model']['free_params'] 
        
        
        # Set the initial abundances for each species as specified under 'abundances' in croc_config.yaml 
        for sp in self.species:
            setattr(self, sp, 10.**self.config['model']['abundances'][sp])
            
        # Set hydrogen as everything besides other species 
        setattr(self, 'h2', 1. - sum([10.**self.config['model']['abundances'][x] for x in self.species]) )
        
        # Instantiate GENESIS only once based on the given model properties
        self.Genesis_instance = genesis.Genesis(self.P_min, self.P_max, self.N_layers, self.lam_min, self.lam_max, self.resolving_power, self.spacing, method = self.method)
        
        self.use_stellar_phoenix = self.config['model']['use_stellar_phoenix']
        if self.use_stellar_phoenix:
            phoenix_model_wave_inp = fits.getdata(self.config['model']['phoenix_model_wave_path']) * 1e-10 ## dividing by 1e10 to convert Ang to m 
            phoenix_model_flux_inp = fits.getdata(self.config['model']['phoenix_model_flux_path']) * 1e-7 * 1e4 * 1e2 # * (phoenix_model_wave_inp**2./self.vel_c_SI) ## converting from 'erg/s/cm^2/cm' to J/m2/s
            stellar_spectrum_smooth_length = self.config['model']['stellar_spectrum_smooth_length']
            
            start_ind, stop_ind = np.argmin(abs(1e6*phoenix_model_wave_inp - 0.9)), np.argmin(abs(1e6*phoenix_model_wave_inp - 3.)) ## Splice the PHOENIX model between 0.9 to 3 micron 
            
            #### FIRST rotationally broaden the PHOENIX Stellar flux, then resample to wavelength solution of the model. Convolving to the instrument resolution and instrument wavelength grid happens later.
            self.phoenix_model_flux_orig = phoenix_model_flux_inp[start_ind:stop_ind]
            self.phoenix_model_wave_orig = phoenix_model_wave_inp[start_ind:stop_ind]
            self.phoenix_model_wave_orig_nm = phoenix_model_wave_inp[start_ind:stop_ind] * 1e9 ## convert to nm for plotting 
            
            self.phoenix_model_flux_broaden, self.phoenix_model_wave_broaden = self.rotation(vsini = self.vsini, 
                                                                                             model_wav = phoenix_model_wave_inp[start_ind:stop_ind], 
                                                       model_spec = phoenix_model_flux_inp[start_ind:stop_ind])
            ### Checked that rotation kernel convolution isn't causign any boundary effects.
            
            ### Smooth the broadened spectrum
            self.phoenix_model_flux_broaden_smooth = self.convolve_spectra_to_given_std(model_orig = self.phoenix_model_flux_broaden, 
                                                                                                                   std = stellar_spectrum_smooth_length)
            
            ###### Resample broadened and smoothed model to model wavelength grid in nm
            phoenix_spl = interpolate.make_interp_spline((10**9)*self.phoenix_model_wave_broaden, self.phoenix_model_flux_broaden_smooth, 
                                                         bc_type='natural')
            
            self.phoenix_model_flux = self.R_star*self.R_star*6.957e8*6.957e8*phoenix_spl((10**9)*self.gen.lam) 
            self.phoenix_model_wav = (10**9)*self.gen.lam
            
            ###### Resample broadened model to model wavelength grid in nm
            # phoenix_spl = interpolate.make_interp_spline((10**9)*self.phoenix_model_wave_broaden, self.phoenix_model_flux_broaden, bc_type='natural')
            # self.phoenix_model_flux_fine = phoenix_spl((10**9)*self.gen.lam) 
            
            # #### Smooth the PHOENIX model by a Gaussian kernel of std = 250 as done by Smith et al. 2024, scale by the radius of the star 
            # self.phoenix_model_flux = self.R_star*self.R_star*6.957e8*6.957e8 * self.convolve_spectra_to_given_std(model_orig = self.phoenix_model_flux_fine, 
            #                                                                                                        std = 200)
            # self.phoenix_model_wav = (10**9)*self.gen.lam     
            
            # plt.figure(figsize = (10,10))
            # plt.plot(self.phoenix_model_wav, self.phoenix_model_flux, 'k.-',label ='Using')
            # # plt.plot(self.phoenix_model_wave_orig_nm, self.R_star*self.R_star*6.957e8*6.957e8*self.phoenix_model_flux_orig, label = 'Orig')
            # # plt.xlim(xmax = 1400)
            # plt.legend()
            # plt.savefig('/home/astro/phsprd/code/crocodel/scripts/px.png')
            
            #### Broaden the PHOENIX model flux by the rotational velocity 
            # self.phoenix_model_flux, _ = self.rotation(vsini = self.vsini, model_wav = self.gen.lam, model_spec = self.phoenix_model_flux_unbroadened)
            
            # import pdb
            # pdb.set_trace()
            ## Loaded PHOENIX model, the BUNIT is 'erg/s/cm^2/cm'	Unit of flux
            ## Radiative flux unit is J/m2/s
            ## 1 erg is 1e-7 J
            ## PHOENIX model BUNIT needs to be converted from erg/s/cm^2/cm to  J/m2/s : convert to Joules and multiply by wavelength (to get rid of the cm factor in denominator)
    
    
    def get_phoenix_modelcube(self, datadetrend_dd = None, model_phoenix_flux = None, model_phoenix_wav = None):
        
        ### Convolve spectra to instrument resolution 
        model_phoenix_flux_lsf = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_phoenix_flux)
        
        phoenix_model_spl = interpolate.make_interp_spline(model_phoenix_wav, model_phoenix_flux_lsf, bc_type='natural')
        phoenix_modelcube = {}

        datelist = list(datadetrend_dd.keys()) 
        for dt, date in enumerate(datelist):
            
            berv = datadetrend_dd[date]['berv']
            data_wavsoln = datadetrend_dd[date]['data_wavsoln']
            nspec = len(berv)
            norder = data_wavsoln.shape[0]
            nwav = data_wavsoln.shape[1]
            
            phoenix_modelcube[date] = np.ones((norder, nspec, nwav))

            for ind in range(norder):
                for it in range(len(berv)):
                    RV_star = self.Vsys + berv[it]
                    wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[ind,:], 
                                                                 velocity = -1.*RV_star)
                    phoenix_modelcube[date][ind,it,:] = phoenix_model_spl(wavsoln_shift)

        return phoenix_modelcube
    
    ####### Rotation kernel #######
    def rotation(self, vsini = None, model_wav = None, model_spec = None):
        # assert(vsini >= 1.0)
        if vsini < 1.0:
            # assert(atm.params["rot"]>=1.0)
            rker = self.get_rotation_kernel(vsini, model_wav)
            # hlen = int((len(rker)-1)/2)
            spec_conv = fftconvolve(model_spec, rker, mode="same")
        else:
            spec_conv = model_spec
        # import pdb
        # pdb.set_trace()
        # return spec_conv[hlen:-hlen], model_wav[hlen:-hlen]
        
        return spec_conv, model_wav
    
    def get_rotation_kernel(self, vsini, model_wav):
        nx, = model_wav.shape
        dRV = np.mean(2.0*(model_wav[1:]-model_wav[0:-1])/(model_wav[1:]+model_wav[0:-1]))*(self.vel_c_SI) ## Speed of light is in m/s
        vsini = vsini * 1e3 ### Convert vsini to m/s
        
        nker = 801
        hnker = (nker-1)//2
        rker = np.zeros(nker)
        
        for ii in range(nker):
            ik = ii - hnker
            x = ik*dRV / vsini
            if np.abs(x) < 1.0:
                y = np.sqrt(1-x**2)
                rker[ii] = y

        rker /= rker.sum() # Normalize the kernel
        assert(rker[0]==0.0 and rker[-1]==0.0)
        rker = rker[rker>0]
        
        return rker
    
    def get_gaussian_kernel(self, size = None, sigma = None):
        x = np.arange(-size // 2 + 1, size // 2 + 1)
        # x = np.linspace(-int(span/2), int(span/2)+1, num = 200)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        return kernel / np.sum(kernel)  # Normalize the kernel
    
    @property
    def mol_mass_dict(self):
        
        mol_mass = {
            'co':28.01,
            'co2':44.01,
            'h2o':18.01528,
            'ch4': 16.04,
            'h2':2.016,
            'he':4.0026,
            'hcn':27.0253,
            'oh':17.00734,
            'h_minus':1.009
        }
        return mol_mass  
    
    def get_MMW(self):
        if self.fix_MMW:
            MMW = self.MMW_value
        else:
            _, press =  self.get_TP_profile() 
            
            X_dict = self.abundances_dict
            ## Use the abundance profile to calculate the MMW 
            mol_mass = self.mol_mass_dict
            MMW = np.ones((len(press), ))
            for sp in X_dict.keys():
                MMW+= X_dict[sp] * mol_mass[sp]
                
            # plt.figure()
            # plt.plot(MMW, press)
            # plt.ylim(press.max(), press.min())
            # plt.yscale('log')
            # plt.xlabel('MMW')
            # plt.ylabel('Pressure [bar]')
            # plt.show()
        return MMW
    
    def get_TP_profile(self):
        """
        Return TP profile, as arrays of T [K] and P [bars]. This function is useful for manipulating the original Genesis instance (and NOT for retrieval, that is done already as part of gen function above which 
        is a property of the class, so just use that elsewhere for example for calculating the equilibrium chemistry abundances.)
        """
        gen_ = self.Genesis_instance
        if self.TP_type in ['Linear', 'Linear_force_inverted', 'Linear_force_non_inverted']:
            # From Sid'e email and looking at set_T function, 
            # Order is (P1,T1),(P2,T2),[P0=,T0=], i.e. down to top. P1 must be greater than P2!
            gen_.set_T(self.P1, self.T1, self.P2, self.T2, type = self.TP_type) # This part should have options to choose different kinds of TP profile.
        
        elif self.TP_type == 'Linear_3_point':
            # From Sid'e email and looking at set_T function, 
            # Order is (P1,T1),(P2,T2),[P0=,T0=], i.e. down to top. P1 must be greater than P2!
            gen_.set_T(self.P1, self.T1, self.P2, self.T2, P0= self.P0, T0= self.T0, type = self.TP_type) # This part should have options to choose different kinds of TP profile.
 
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
            # tempS, presS = self.TP_data['T[K]'], self.TP_data['P[bar]']
            tempS, presS = self.TP_data[0], self.TP_data[1]
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
        
        elif self.TP_type == 'Bezier_6_nodes':
            gen_.T = aut.PTbez(logParr = np.log10(gen_.P.copy()/1e5),
                                Ps = [self.log_P5, self.log_P4, self.log_P3, self.log_P2, self.log_P1, self.log_P0],
                                Ts = [self.T5, self.T4, self.T3, self.T2, self.T1, self.T0] )  
            
        return gen_.T, gen_.P.copy() / 1E5 
    
    
    @property
    def gen(self):
        """Get the Genesis object based on the latest value of parameters.

        :return: Updated genesis object with model parameters set to the latest values.
        :rtype: genesis.Genesis
        """

        gen_ = self.Genesis_instance
        ### Set the TP profile 
        temp, _ =  self.get_TP_profile() 
        gen_.T = temp
        ### Get the MMW 
        MMW = self.get_MMW()
        # print(MMW)
        # MMW_mean = np.mean(MMW)
        
        gen_.profile(self.R_planet, self.log_g, self.P_ref, mu = MMW) #Rp (Rj), log(g) cgs, Pref (log(bar))
        
        return gen_    
    
    
    def get_eqchem_abundances(self):
        """Given TP profile, and the C/O and metallicity, compute the equilibrium chemistry abundances of the species included in the retrieval. 
        The outputs from this can be used when constructing the abundances dictionary for GENESIS.

        :return: _description_
        :rtype: _type_
        """
        
        element_abundances = np.copy(self.solar_abundances)
  
        #scale the element abundances, except those of H and He
        for j in range(0, self.fastchem.getElementNumber()):
            if self.fastchem.getElementSymbol(j) != 'H' and self.fastchem.getElementSymbol(j) != 'He':
                element_abundances[j] *= 10.**self.logZ_planet
                
        # Set the abundance of C with respect to O according to the C/O ratio ; 
        # only do this if use_C_to_O flag is set to True in the config file 
        if self.use_C_to_O:
            element_abundances[self.index_C] = element_abundances[self.index_O] * self.C_to_O
        
        # Set the abundance of C with respect to O according to the C/O ratio
        # element_abundances[self.index_C] = element_abundances[self.index_O] * self.C_to_O ## Was not commented before 16-06-2025

        self.fastchem.setElementAbundances(element_abundances)
        
        temp, press = self.get_TP_profile()
        
        self.input_data.temperature = temp
        self.input_data.pressure = press ## pressure is already in bar as calculated by get_TP_profile
        
        fastchem_flag = self.fastchem.calcDensities(self.input_data, self.output_data)
        
        #convert the output into a numpy array
        number_densities = np.array(self.output_data.number_densities)
        
        return number_densities
    
    def get_eqchem_ER_abundances(self):
        """Given TP profile, and elemental ratios, compute the equilibrium chemistry abundances of the species included in the retrieval. 
        The outputs from this can be used when constructing the abundances dictionary for GENESIS.

        :return: _description_
        :rtype: _type_
        """
        
        # Get the solar abundances 
        element_abundances = np.copy(self.solar_abundances) 
        
        # Calculate the value of (C_to_H/C_to_H_solar)
        C_to_H_by_C_to_H_solar = 10.**(self.log10_C_to_H_by_C_to_H_solar)
        O_to_H_by_O_to_H_solar = 10.**(self.log10_O_to_H_by_O_to_H_solar)

        # Scale the C and O by their ratios to H
        element_abundances[self.index_C] *= C_to_H_by_C_to_H_solar
        element_abundances[self.index_O] *= O_to_H_by_O_to_H_solar
  
        # Scale the refractories by R/H ratio
        R_to_H_by_R_to_H_solar = 10.**(self.log10_R_to_H_by_R_to_H_solar)
        for refrac in self.refractories:
            ind_refrac = getattr(self, 'index_' + refrac)
            element_abundances[ind_refrac] *= R_to_H_by_R_to_H_solar
        self.fastchem.setElementAbundances(element_abundances)
        
        temp, press = self.get_TP_profile()
        
        self.input_data.temperature = temp
        self.input_data.pressure = press ## pressure is already in bar as calculated by get_TP_profile
        
        fastchem_flag = self.fastchem.calcDensities(self.input_data, self.output_data)
        
        #convert the output into a numpy array
        number_densities = np.array(self.output_data.number_densities)
        
        return number_densities
        
    def dissociation(self, sp_name = 'h2o'):
        X_i = getattr(self, sp_name)
        
        alpha = getattr(self, 'alpha_' + sp_name)
        log10_beta = getattr(self, 'log10_beta_' + sp_name)
        beta = 10.** log10_beta
        gamma = getattr(self, 'gamma_' + sp_name)
        
        # T = np.copy(self.gen.T)
        T, P = self.get_TP_profile()
        
        T[T<1700.0] = 1700.0
        T[T>4000.0] = 4000.0
        
        # P = self.gen.P.copy()/1E5#convert to bar
        
        P[P<1e-6] = 1e-6
        P[P>200.0] = 200.0
        Ad = np.power(10.0, beta/T - gamma)*np.power(P,alpha)
        A = 1.0/np.sqrt(X_i) + 1.0/np.sqrt(Ad)
        #A = 1.0/np.sqrt(np.power(10.0,self.A0[key])) + 1.0/np.sqrt(Ad)
        A = 1.0/(A*A)
        
        return A
    
    @property
    def abundances_dict(self):
        """Setup the dictionary of abundances based on the latest set of parameters.

        :return: Abundance dictionary.
        :rtype: dict
        """
        
        X = {}
        temp, press = self.get_TP_profile()
        
        if self.chemistry == 'free_chem':
            for sp in self.species:
                if sp not in ["h2", "he"]:
                    X[sp] = np.full(len(press), getattr(self, sp))
            X["he"] = np.full(len(press), self.he)
            
            metals = np.full(len(press), 0.)
            for sp in self.species:
                if sp != "h2":
                    metals+=X[sp]
                
                X["h2"] = 1.0 - metals
                
        elif self.chemistry == 'free_chem_with_dissoc': # free chemistry with dissociation
            
            for sp in self.species:
                if sp not in ["h2", "he"] + list(self.sp_dissoc_list):
                    X[sp] = np.full(len(press), getattr(self, sp))
                elif sp in list(self.sp_dissoc_list):
                    X[sp] = self.dissociation(sp_name = sp)
            X["he"] = np.full(len(press), self.he)

            metals = np.full(len(press), 0.)
            for sp in self.species:
                if sp != "h2":
                    metals+=X[sp]
                
                X["h2"] = 1.0 - metals
        
        elif self.chemistry == 'eq_chem':
            number_densities = self.get_eqchem_abundances()
            #total gas particle number density from the ideal gas law 
            #Needed to convert the number densities output from FastChem to mixing ratios
            gas_number_density = ( ( press ) *1e6 ) / ( self.k_B_cgs * temp )
            
            # for sp in self.species:
            #     if sp != 'h_minus':
            #         vmr = number_densities[:, self.species_fastchem_indices[sp]]/gas_number_density
            #         # X[sp] = vmr.value
            #         X[sp] = vmr 
            # vmr_h2 = number_densities[:, self.species_fastchem_indices["h2"]]/gas_number_density
            # X["h2"] = vmr_h2
            
            ####### Extracting the h_minus from the Fastchem itself.
            for sp in self.species:
                # print(sp)
                vmr = number_densities[:, self.species_fastchem_indices[sp]]/gas_number_density
                X[sp] = vmr 
            vmr_h2 = number_densities[:, self.species_fastchem_indices["h2"]]/gas_number_density
            X["h2"] = vmr_h2
        
        elif self.chemistry == 'eq_chem_ER':
            number_densities = self.get_eqchem_ER_abundances()
            gas_number_density = ( ( press ) *1e6 ) / ( self.k_B_cgs * temp )
            ####### Extracting the h_minus from the Fastchem itself.
            
            for sp in self.species:
                vmr = number_densities[:, self.species_fastchem_indices[sp]]/gas_number_density
                X[sp] = vmr 
            vmr_h2 = number_densities[:, self.species_fastchem_indices["h2"]]/gas_number_density
            X["h2"] = vmr_h2
            
        assert all(X["h2"] >= 0.) # make sure that the hydrogen abundance is not going negative!   
        return X 
    
    def get_spectra(self, exclude_species = None):
        """Compute the transmission or emission spectrum.

        :return: Wavelength (in nm) and transmission or emission spectrum arrays.
        :rtype: array_like
        """
        if exclude_species is not None:
            abund_dict = copy.deepcopy(self.abundances_dict)
            for spnm in exclude_species:
                abund_dict[spnm] = abund_dict[spnm] * 1e-30
        else:
            abund_dict = copy.deepcopy(self.abundances_dict)
            
        if self.method == "transmission":
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict, cl_P = self.cl_P)
            spec = self.gen.genesis(abund_dict, cl_P = self.cl_P, include_cia = self.include_cia)
            spec /= ((self.R_star*6.957e8)**2.0)
            # spec = 1.-spec
            spec = -spec
        elif self.method == 'emission':
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict)
            spec = self.gen.genesis(abund_dict, include_cia = self.include_cia)
            spec /= self.stellar_flux_BB(self.R_star, self.gen.lam, self.T_eff)
        
        return (10**9) * self.gen.lam, 10**self.log_fs * spec 
    
    def get_Fp_spectra(self, exclude_species = None):
        """Compute only the Fp in case of emission.

        :return: Wavelength (in nm) and transmission or emission spectrum arrays.
        :rtype: array_like
        """
        if exclude_species is not None:
            abund_dict = copy.deepcopy(self.abundances_dict)
            for spnm in exclude_species:
                abund_dict[spnm] = abund_dict[spnm] * 1e-30
        else:
            abund_dict = copy.deepcopy(self.abundances_dict)
            
        if self.method == "transmission":
            # # spec = self.gen.genesis_without_opac_check(self.abundances_dict, cl_P = self.cl_P)
            # spec = self.gen.genesis(self.abundances_dict, cl_P = self.cl_P)
            # spec /= ((self.R_star*6.957e8)**2.0)
            # # spec = 1.-spec
            # spec = -spec
            sys.exit('Method only applicable to emission.')
        elif self.method == 'emission':
            # spec = self.gen.genesis_without_opac_check(self.abundances_dict)
            spec = self.gen.genesis(abund_dict, include_cia = self.include_cia)
            # spec /= self.stellar_flux(self.R_star, self.gen.lam, self.T_eff)
            # return spec
        return (10**9) * self.gen.lam, 10**self.log_fs * spec     
    
    def stellar_flux_BB(self, Rs, lam, T):
        """Compute the flux from the star, either as blackbody or from a PHOENIX model.

        :param Rs: Stellar radius, in terms of solar radius.
        :type Rs: float
        :param lam: Wavelength range, in metres!
        :type lam: array
        :param T: Effective stellar temperature.
        :type T: float
        :return: Blackbody flux of the star.
        :rtype: array
        """
        lam_5 = lam*lam*lam*lam*lam
        Bs = (2.0*sc.h*self.vel_c_SI*self.vel_c_SI)/(lam_5*(np.expm1((sc.h*self.vel_c_SI)/(lam*sc.k*T))))
        # import pdb
        # pdb.set_trace()
        return Bs*np.pi*Rs*Rs*6.957e8*6.957e8
    
    # def get_stellar_flux_PHOENIX_cube(self, Rs):

    #     # phoenix_flux = self.phoenix_model_flux * (self.vel_c_SI/(lam*lam)) * lam * np.pi*Rs*Rs*6.957e8*6.957e8
    #     phoenix_flux = self.phoenix_model_flux *Rs*Rs*6.957e8*6.957e8
        
    #     phoenix_flux_
    
    def get_contribution_function(self):
        X_dict = self.abundances_dict
        contribution_func, tau, P_array, P_tau = self.gen.contribution_function(X_dict, tau_val = 2./3., include_cia = self.include_cia)    

        return contribution_func, tau, P_array, P_tau
    
    def convolve_spectra_to_instrument_resolution(self, model_spec_orig=None):
        """Convolve the given input model spectrum to the instrument resolution. 
        Assumes that the instrument resolution is constant with wavelength.

        :param model_spec_orig: Given 1D model spectrum flux, defaults to None
        :type model_spec_orig: array_like
        :return: Model spectrum convolved to the instrument resolution.
        :rtype: array_like
        """
        
        delwav_by_wav = 1/self.config['data'][self.inst]['resolution'] # for the instrument (value is 1/100000 for crires and 1/45000 for igrins) 
        delwav_by_wav_model = 1./self.config['model']['R_power']   ### np.diff(model_wav)/model_wav[1:]
        
        ############ Convolve to instrument resolution 
        # FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        # sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
        # model_spec = convolve(model_spec_orig, Gaussian1DKernel(stddev=sig), boundary='extend')
        # return model_spec
        
        FWHM = np.mean(delwav_by_wav/delwav_by_wav_model)
        sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )
        gauss_kernel = self.get_gaussian_kernel(size = sig*10, sigma = sig)
        # model_spec = np.convolve(model_spec_orig)           
        model_spec = np.convolve(model_spec_orig, gauss_kernel, mode = 'same')
        return model_spec
    
    def convolve_spectra_to_given_std(self, model_orig=None, std = None):
        """Convolve the given input model spectrum to the instrument resolution. 
        Assumes that the instrument resolution is constant with wavelength.

        :param model_spec_orig: Given 1D model spectrum flux, defaults to None
        :type model_spec_orig: array_like
        :return: Model spectrum convolved to the instrument resolution.
        :rtype: array_like
        """
        gauss_kernel = self.get_gaussian_kernel(size = std*10, sigma = std)     
        model_spec = np.convolve(model_orig, gauss_kernel, mode = 'same')
        
        # Pad the signal symmetrically
        # model_spec_padded = np.pad(model_orig, (std*10 // 2, std*10 // 2), mode='symmetric')

        # Perform convolution
        # model_spec = np.convolve(model_spec_padded, gauss_kernel, mode='valid')
        return model_spec
    
    def get_reprocessed_modelcube(self, model_spec = None, model_wav = None, 
                                  
                                  model_Fp = None, phoenix_modelcube = None,
                                  
                                  datacube = None, 
                                  datacube_detrended = None, data_wavsoln = None,
                                  pca_eigenvectors = None, colmask = None, post_pca_mask = None,
                                  phases = None, berv = None):
        """Given the datacubes (original and detrended) for a single data and detector/order, 
        and the pca_eigenvectors, for a single value of Kp and Vsys , 
        compute the reprocessed modelcube corresponding to each exposure as those in the observations.

        :param model_spec: Given 1D model spectrum flux, MUST already be convolved to instrument resolution, defaults to None
        :type model_spec: array_like
        
        :param model_wav: Wavelength, in nm, defaults to None
        :type model_wav: array_like, optional
        
        :param datacube: Original datacube, defaults to None
        :type datacube: array_like
        
        :param datacube_detrended: PCA detrended datacube, defaults to None
        :type datacube_detrended: array_like
        
        :param data_wavsoln: Data wavelength solution, 1D array assuming the wavelength solution is same for all exposures in a given order/detector.
        :type data_wavsoln: array_like
        
        :param pca_eigenvectors: Set of PCA eigenvectors used for detrending the datacube.
        :type pca_eigenvectors: array_like
        
        :param colmask: Mask for spectral channels to be masked either because of bad spectral channel or anything that is masked pre-PCA, defaults to None
        :type colmask: array_like, 1D array of bool
        
        :param post_pca_mask: 1D array of post PCA mask, defaults to None
        :type post_pca_mask: array_like, 1D array of bools
        
        :param phases: 1D array of phase values, defaults to None
        :type phases: array_like
        
        :param berv: 1D array of BERV, defaults to None
        :type berv: array_like
        
        :return: Reprocessed model cube and combined mask for spectral channels to be 'avoided' for future calculations.
        :rtype: _type_
        """
        
        nspec, nwav = datacube.shape[0], datacube.shape[1]
        # Initialize the reprocessed modelcube
        model_reprocess = np.empty((nspec, nwav))
        
        datamodel = np.empty((nspec, nwav))
        datamodel_fit = np.empty((nspec, nwav))
        datamodel_detrended = np.empty((nspec, nwav))
        
        # model_spl = splrep(model_wav, model_spec)
        if self.use_stellar_phoenix:
            model_spl = interpolate.make_interp_spline(model_wav, model_Fp, bc_type='natural')
        else:
            model_spl = interpolate.make_interp_spline(model_wav, model_spec, bc_type='natural')
        
        model_spec_shift_cube = np.empty((nspec, nwav))
        
        # Based on given value of Kp, shift the 1D model by the expected total velocity of the planet for each exposure
        # print(self.Kp, self.Vsys)
        for it in range(nspec):
            RV = self.Kp * np.sin(2. * np.pi * (phases[it] + self.phase_offset)) + self.Vsys + berv[it]
            
            data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * RV)
            # model_spec_shift_exp = splev(data_wavsoln_shift, model_spl)
            model_spec_shift_exp = model_spl(data_wavsoln_shift)
            model_spec_shift_cube[it, :] = model_spec_shift_exp

        ######## Check if you are using phoenix models, 
        ##### if yes then model_spec_shift_cube is only Fp, and you need to normalize it by phoenix_modelcube,
        ##### else just keep using it as is.
        if self.use_stellar_phoenix:
            model_spec_shift_cube = model_spec_shift_cube/phoenix_modelcube
        
        
        # Inject the model into the data (should work for both transmission and emission as for transmission the model_spec has -ve sign)
        if self.method == 'transmission':
            datamodel = datacube * model_spec_shift_cube
        elif self.method == 'emission':
            datamodel = datacube * (1. + model_spec_shift_cube)

        # Perform the linear regression fit to data+model
        datamodel_fit = stc.linear_regression(X=pca_eigenvectors,
                                                Y=datamodel)
        # Detrend data+model
        datamodel_detrended = datamodel/(datamodel_fit+1e-100) - 1.

        # Zero out the same channels/columns as done for the datacube (including the ones post PCA)
        datamodel_detrended[:, post_pca_mask] = 0.
        # Also zero out the bad columns 
        datamodel_detrended[:, colmask] = 0.
        
        # Calculate the reprocessed modelcube
        model_reprocess = datamodel_detrended - datacube_detrended
        
        # Zero out the model for channels you want to mask 
        model_reprocess[:, post_pca_mask] = 0.
        model_reprocess[:, colmask] = 0.

        avoid_mask = np.logical_or(post_pca_mask, colmask) 
        
        return model_reprocess, avoid_mask # see line 684 in croc on how to use this output further 
        
    

            
    
    def logL_fast(self, theta, datadetrend_dd = None, order_inds = None):
        """Function to calculate the total logL for data from a single instrument and all dates combined as per the 
        initial specifications of the Model class. The separation of date and instruments here has been done to 
        allow flexibility in the dates and instruments you might want to include in a retrieval. 
        The order of free params in the theta vector should be EXACTLY the same 
        as that defined in the yaml file. Note that in python version >=3.7, dictionaries are by default always ordered.

        :param theta: Array of values of free parameters for which the total log-likelihood is to be computed. 
        :type theta: array
        
        :param datadetrend_dd: Detrended data dictionary.
        :type datadetrended_dd: dict
        
        :param order_inds: Index of orders/detectors for which you want to compute the log-likelihood.
        :type order_inds: array of int
        
        :return: Scalar log-likelihood value 
        :rtype: float64
        
        """
        
        # First set the parameters of the model based on the list of free parameters in the config file. 
        #######################################################################
        ################# SET THE SAMPLED FREE PARAMETERS FOR THE MODEL #######
        for i, pname in enumerate(self.free_params_dict.keys()):
            if pname in self.species or pname in ['P0', 'P1','P2']:
                setattr(self, pname, 10.**theta[i])
                # print('Setting ', pname, ': ', 10.**theta[i])
            else:
                setattr(self, pname, theta[i])
                # print('Setting ', pname, ': ', theta[i])
        
        
        datelist = list(datadetrend_dd.keys()) 
        
        #######################################################################
        #######################################################################
                
        # Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        if self.use_stellar_phoenix:
            model_wav, model_Fp_orig = self.get_Fp_spectra()
            
            phoenix_modelcube = self.get_phoenix_modelcube(datadetrend_dd = datadetrend_dd, 
                                                           model_phoenix_flux = self.phoenix_model_flux, 
                                                           model_phoenix_wav = self.phoenix_model_wav)
            
            ### Rotationally broaden the planetary spectrum
             
            model_Fp_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                                       model_wav = model_wav, model_spec = model_Fp_orig)
            
            model_Fp = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
            model_spec = None
        else:
            model_wav, model_spec_orig = self.get_spectra()        
            # Rotationally broaden the spectrum 
            model_spec_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                            model_wav = model_wav, model_spec = model_spec_orig)
            
            # Convolve the model to the instrument resolution already
            model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
            model_Fp = None
            phoenix_modelcube = {}
            for date in datelist:
                phoenix_modelcube[date] = None
            

        logL_per_date = np.empty(len(datelist))
         
        # Loop over all dates in the datadetrended_dd dictionary
        for dt, date in enumerate(datelist):
            
            # Instantiate the array to store the logL
            logL_per_ord = np.empty(len(order_inds))
            
            # Loop over the specified orders
            for num_ind, ind in zip(range(len(order_inds)),order_inds):
                
                nspec = datadetrend_dd[date]['datacube'][ind, :, :].shape[0]
                
                # Calculate the reprocessed modelcube for the given values of Kp and Vsys
                if self.use_stellar_phoenix:
                    phoenix_modelcube_inp = phoenix_modelcube[date][ind,:,:]
                else:
                    phoenix_modelcube_inp = None
                model_reprocess, avoid_mask = self.get_reprocessed_modelcube(
                                                model_spec = model_spec, model_wav = model_wav, 
                                                
                                                model_Fp = model_Fp, phoenix_modelcube = phoenix_modelcube_inp,
                                                
                                                datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                                datacube_detrended = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                                data_wavsoln = datadetrend_dd[date]['data_wavsoln'][ind, :],
                                        pca_eigenvectors = datadetrend_dd[date]['pca_eigenvectors'][ind][:], 
                                        colmask = datadetrend_dd[date]['colmask'][ind, :],
                                        post_pca_mask = datadetrend_dd[date]['post_pca_mask'][ind, :],
                                        phases = datadetrend_dd[date]['phases'], berv = datadetrend_dd[date]['berv'], 
                                        )
                
                
                # Instantiate arrays to store cross-covariance, cross-correlation, and log-likelihood values per exposure 
                R_per_spec, C_per_spec, logL_per_spec = np.empty(nspec), np.empty(nspec), np.empty(nspec)

                # Loop over all exposures for this date and order 
                for it in range(nspec):
                    model_spec_flux_shift = model_reprocess[it, :]
                    
                    # Mean subtract the model excluding the values which have been set to zero 
                    model_spec_flux_shift = crocut.sub_mask_1D(model_spec_flux_shift, avoid_mask)
                                        
                    # Compute R, C, logL accounting for the correct number of non-zero data points (only those that actually contribute to the CCF finitely, so all channels besides avoid_mask)
                    R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, ~avoid_mask],
                                                                                        model=model_spec_flux_shift[~avoid_mask])

                # Sum the log-likelihood over all exposures to compute the total log-likelihood for this order/detector
                logL_per_ord[num_ind] = np.sum(logL_per_spec) 
            
            # Sum the log-likelihood over all orders to compute the total log-likelihood for this date
            logL_per_date[dt] = np.sum(logL_per_ord) # np.dot(logL_per_ord, np.ones(len(order_inds)))

        # Sum the log-likelihood over all dates
        logL_total = np.sum(logL_per_date)
                
        return logL_total
    
    def get_ccf_trail_matrix(self, datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, plot = False, savedir = None, 
                             fixed_model_wav = None, fixed_model_spec = None):
        
        """For the initial set of model parameters - for each date and each detector, 
        take a model spectrum, and cross correlate with each exposure, 
        construct the CCF trail matrix and return. 

        :param datadetrend_dd: Detrended data dictionary, defaults to None
        :type datadetrend_dd: dict
        
        :param order_inds: Index of orders/detectors for which you want to compute the log-likelihood.
        :type order_inds: array of int
        
        :param Vsys_range: Range of Vsys, defaults to None
        :type Vsys_range: array_like
        
        :param plot: Set True if you want to plot the trail matrix., defaults to False
        :type plot: bool, optional
        
        :param savedir: path to the directory where you want to save the plot, defaults to None
        :type savedir: str, optional
        
        :return: CCF trail matrix dictionary
        :rtype: dict
        """
        # # Calculate the model_spec and model_wav which should be the same for all dates for this instrument 
        # # (all taken in same mode : transmission or emission)
        # model_wav, model_spec_orig = self.get_spectra()
        
        # # Convolve model to instrument resolution
        # model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig, model_wav_orig=model_wav)
        # # model_spl = splrep(model_wav, model_spec)
        # model_spl = interpolate.make_interp_spline(model_wav, model_spec, bc_type='natural')
        
        datelist = list(datadetrend_dd.keys())
        
        if self.use_stellar_phoenix:
            if fixed_model_spec is None:
                model_wav, model_Fp_orig = self.get_Fp_spectra()
            else:
                model_wav, model_Fp_orig = fixed_model_wav, fixed_model_spec
                
            phoenix_modelcube = self.get_phoenix_modelcube(datadetrend_dd = datadetrend_dd, 
                                                        model_phoenix_flux = self.phoenix_model_flux, 
                                                        model_phoenix_wav = self.phoenix_model_wav)
            
            ### Rotationally broaden the planetary spectrum 
            model_Fp_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                                       model_wav = model_wav, model_spec = model_Fp_orig)
            
            
            model_Fp = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
            model_spl = interpolate.make_interp_spline(model_wav, model_Fp, bc_type='natural')  
            plt.figure()
            plt.plot(model_wav, model_Fp/self.phoenix_model_flux)
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Fp/Fs')
            plt.savefig(savedir + 'FpFs_actual.png', 
                        format='png', dpi=300, bbox_inches='tight')
            
        else:
            if fixed_model_spec is None:
                model_wav, model_spec_orig = self.get_spectra()
            else:
                model_wav, model_spec_orig = fixed_model_wav, fixed_model_spec
            
            # Rotationally broaden the spectrum 
            model_spec_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                            model_wav = model_wav, model_spec = model_spec_orig)   
            # Convolve the model to the instrument resolution already
            model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
            model_Fp = None
            phoenix_modelcube = {}
            for date in datelist:
                phoenix_modelcube[date] = None
            model_spl = interpolate.make_interp_spline(model_wav, model_spec, bc_type='natural')   
            
            plt.figure()
            plt.plot(model_wav, model_spec)
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Fp/Fs')
            plt.savefig(savedir + 'FpFs_actual.png', 
                        format='png', dpi=300, bbox_inches='tight')
        
        
        # Dictionary to store the CCF trail matrix
        ccf_trail_matrix_dd = {}
        
        # Loop over all dates 
        for dt, date in tqdm(enumerate(datadetrend_dd.keys())):
            
            # Array to store the total logL for each order 
            logL_per_ord = np.empty(len(order_inds))
            
            # Dictionary to store the CCF trail matrix for each order for the given date 
            ccf_trail_matrix_dd[date] = {}
            if self.use_stellar_phoenix:
                phoenix_modelcube_this_date = phoenix_modelcube[date]
                
            # Loop over all orders 
            for ind in tqdm(order_inds):
                
                nspec, nvel  = datadetrend_dd[date]['datacube'][ind, :, :].shape[0], len(Vsys_range)
                
                # Empty array to fill the CCF matrix for this order and date
                ccf_trail_matrix_dd[date][ind] = np.empty((nspec, nvel))
                
                # Loop over all velocities 
                for iVsys, Vsys in enumerate(Vsys_range):
                    for it in range(nspec):
                        RV = Vsys + datadetrend_dd[date]['berv'][it]
                        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=datadetrend_dd[date]['data_wavsoln'][ind, :], 
                                                                        velocity=-1. * RV)
                        # model_spec_shift_exp = splev(data_wavsoln_shift, model_spl)
                        model_spec_shift_exp = model_spl(data_wavsoln_shift)
                        model_spec_shift_exp = model_spec_shift_exp - crocut.fast_mean(model_spec_shift_exp)
                        if self.use_stellar_phoenix:
                            model_spec_shift_exp = model_spec_shift_exp/phoenix_modelcube_this_date[ind,it, :]
                        
                        ccf_trail_matrix_dd[date][ind][it, iVsys], _, _ = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, :],
                                                                                        model=model_spec_shift_exp)
                        
                        if iVsys == it == 0:
                            fig = plt.figure()
                            plt.plot(datadetrend_dd[date]['data_wavsoln'][ind, :], datadetrend_dd[date]['datacube_mean_sub'][ind, it, :], label = 'Data')
                            plt.plot(data_wavsoln_shift, model_spec_shift_exp, label = 'Model')
                            plt.legend()
                            plt.savefig(savedir + 'ccf_trail_model_data_comparison_'+date+ '_' + str(ind) + '_'+'.png', 
                                        format='png', dpi=300, bbox_inches='tight')
                            plt.close(fig = fig)
                        
        # Sum the CCF across all orders for each date 
        for date in datadetrend_dd.keys():
            ccf_trail_total = np.zeros((nspec,nvel))
            for ind in order_inds:
                ccf_trail_total+=ccf_trail_matrix_dd[date][ind]
            ccf_trail_matrix_dd[date]['total'] = ccf_trail_total
            ccf_trail_matrix_dd[date]['total'] = ccf_trail_total
            ccf_trail_matrix_dd[date]['phases'] = datadetrend_dd[date]['phases']
            ccf_trail_matrix_dd[date]['berv'] = datadetrend_dd[date]['berv']
        ccf_trail_matrix_dd['Vsys_range'] = Vsys_range
        np.save(savedir + 'ccf_trail_matrix_NO_model_reprocess.npy', ccf_trail_matrix_dd)
        
        if plot:
            # First the CCF trail matrix for plot individual dates and orders 
            for date in ccf_trail_matrix_dd.keys():
                plot_type = 'pcolormesh'
                # subplot_num = len(ccf_trail_matrix_dd[date].keys()) ## Make a subplot for each detector
                
                # fig, axx = plt.subplots(subplot_num, 1, figsize=(5, 5*subplot_num))

                # plt.subplots_adjust(hspace=0.8)

                # for axis_ind, ind in zip(range(subplot_num), order_inds):
                #     hnd1 = crocut.subplot_cc_matrix(axis=axx[axis_ind],
                #                                 cc_matrix=ccf_trail_matrix_dd[date][ind],
                #                                 phases=datadetrend_dd[date]['phases'],
                #                                 velocity_shifts=Vsys_range,
                #                                 ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                #                                 title= 'Date: '+ date +'; Detector: ' + str(ind) ,
                #                                 setxlabel=True, plot_type = plot_type)
                #     fig.colorbar(hnd1, ax=axx[axis_ind])

                #     axx[axis_ind].set_ylabel(r'$\phi$')
                #     axx[axis_ind].set_xlabel(r'V$_{rest}$ [km/s]')

                # plt.savefig(savedir + 'ccf_trail_date-' + date + '.png', format='png', dpi=300, bbox_inches='tight')
                # plt.close()
            
                # Plot the total trail matrix across all dates and detectors 
                plt.figure(figsize = (15,5))
                ax = plt.gca()
                hnd1 = crocut.subplot_cc_matrix(axis=ax,
                                cc_matrix=ccf_trail_matrix_dd[date]['total'],
                                phases=datadetrend_dd[date]['phases'],
                                velocity_shifts=Vsys_range,
                                ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                title= date+', Total' ,
                                setxlabel=True, plot_type = plot_type)
                
                ax.set_ylabel(r'$\phi$')
                ax.set_xlabel(r'V$_{rest}$ [km/s]')
                plt.savefig(savedir + 'ccf_trail_total_'+date+'.png', format='png', dpi=300, bbox_inches='tight')
            
            return ccf_trail_matrix_dd
        
        else:
            return ccf_trail_matrix_dd
        
    def get_ccf_trail_matrix_with_model_reprocess(self, datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, Kp_range = None, savedir = None,
                             fixed_model_spec = None, fixed_model_wav = None):
        
        """For the initial set of model parameters - for each date and each detector, 
        take a model spectrum, and cross correlate with each exposure, 
        construct the CCF trail matrix and return. 

        :param datadetrend_dd: Detrended data dictionary, defaults to None
        :type datadetrend_dd: dict
        
        :param order_inds: Index of orders/detectors for which you want to compute the log-likelihood.
        :type order_inds: array of int
        
        :param Vsys_range: Range of Vsys, defaults to None
        :type Vsys_range: array_like
        
        :param plot: Set True if you want to plot the trail matrix., defaults to False
        :type plot: bool, optional
        
        :param savedir: path to the directory where you want to save the plot, defaults to None
        :type savedir: str, optional
        
        :return: CCF trail matrix dictionary
        :rtype: dict
        """
        
        datelist = list(datadetrend_dd.keys()) 

        if self.use_stellar_phoenix:
            if fixed_model_spec is None:
                model_wav, model_Fp_orig = self.get_Fp_spectra()
            else:
                model_wav, model_Fp_orig = fixed_model_wav, fixed_model_spec
                
            phoenix_modelcube = self.get_phoenix_modelcube(datadetrend_dd = datadetrend_dd, 
                                                        model_phoenix_flux = self.phoenix_model_flux, 
                                                        model_phoenix_wav = self.phoenix_model_wav)
            
            ### Rotationally broaden the planetary spectrum 
            model_Fp_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                                       model_wav = model_wav, model_spec = model_Fp_orig)
            
            
            model_Fp = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
            model_spec = None
            
        else:
            if fixed_model_spec is None:
                model_wav, model_spec_orig = self.get_spectra()
            else:
                model_wav, model_spec_orig = fixed_model_wav, fixed_model_spec
            
            # Rotationally broaden the spectrum 
            model_spec_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                            model_wav = model_wav, model_spec = model_spec_orig)   
            # Convolve the model to the instrument resolution already
            model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
            model_Fp, phoenix_modelcube = None, None
        
        
        ccf_trail_matrix_dd = {}
        ccf_trail_matrix_interp_dd = {}
        
        # Loop over all dates 
        for date in tqdm(datelist):
            
            # Dictionary to store the CCF trail matrix for each order for the given date 
            ccf_trail_matrix_dd[date] = {}
            ccf_trail_matrix_interp_dd[date] = {}

            # Loop over all orders 
            for ind in tqdm(order_inds):
                
                nspec = datadetrend_dd[date]['datacube'][ind, :, :].shape[0]

                V_pair = []
                for iKp, Kp_val in enumerate(Kp_range):
                    for iVsys, Vsys_val in enumerate(Vsys_range):
                        V_pair.append((Kp_val, Vsys_val))
                
                V_pair = np.array(V_pair)
                nvel = V_pair.shape[0]
                V_total = np.zeros((nspec, nvel))
                

                
                # Empty array to fill the CCF matrix for this order and date
                ccf_trail_matrix_dd[date][ind] = np.empty((nspec, nvel))
                # Loop over all velocities 
                # Kp_orig, Vsys_orig = copy.deepcopy(self.Kp), copy.deepcopy(self.Vsys)
                # for iKp, Kp_val in enumerate(Kp_range): 
                #     for iVsys, Vsys_val in enumerate(Vsys_range):
                        
                for iV, V_val in enumerate(V_pair):
                    Kp_val, Vsys_val = V_val[0], V_val[1]
                    setattr(self, 'Vsys', Vsys_val)
                    setattr(self, 'Kp', Kp_val)
                    
                    
                    if self.use_stellar_phoenix:
                        phoenix_modelcube_inp = phoenix_modelcube[date][ind,:,:]
                    else:
                        phoenix_modelcube_inp = None
                    # Compute the reprocessed model 
                    model_reprocess, avoid_mask = self.get_reprocessed_modelcube(model_spec = model_spec, model_wav = model_wav, 
                                                                                    
                                                                                    model_Fp = model_Fp, phoenix_modelcube = phoenix_modelcube_inp,
                                                                                    
                                                    datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                                    datacube_detrended = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                                    data_wavsoln = datadetrend_dd[date]['data_wavsoln'][ind, :],
                                            pca_eigenvectors = datadetrend_dd[date]['pca_eigenvectors'][ind][:], 
                                            colmask = datadetrend_dd[date]['colmask'][ind, :],
                                            post_pca_mask = datadetrend_dd[date]['post_pca_mask'][ind, :],
                                            phases = datadetrend_dd[date]['phases'], berv = datadetrend_dd[date]['berv'])
                    
                    for it in range(nspec):
                        model_spec_flux_shift = model_reprocess[it, :]
                        # Mean subtract the model with the zero values ignored
                        model_spec_flux_shift = crocut.sub_mask_1D(model_spec_flux_shift, avoid_mask)
                        _, ccf_trail_matrix_dd[date][ind][it, iV], _ = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, ~avoid_mask],
                                                                model=model_spec_flux_shift[~avoid_mask])

                        V_total[it, iV] = Kp_val * np.sin(2. * np.pi * datadetrend_dd[date]['phases'][it] + self.phase_offset) + Vsys_val + datadetrend_dd[date]['berv'][it]
                
                ##### After the velocity loop is done, interpolate the CCF trail matrix to a common velocity grid
                V_range_common = np.arange(-100., 100., 1.)
                ccf_trail_matrix_interp_dd[date][ind] = np.zeros((nspec, len(V_range_common)))
                for it in range(nspec):
                    V_total_sorted = np.sort(V_total[it, :])
                    # Sort ccf_trail_matrix_dd[date][ind][it, :] in the same order as V_total_sorted
                    sort_indices = np.argsort(V_total[it, :])
                    ccf_sorted = ccf_trail_matrix_dd[date][ind][it, :][sort_indices]
                    spl = interpolate.make_interp_spline(V_total_sorted, ccf_sorted, bc_type='natural')
                    # Interpolate the CCF to a common velocity range
                    ccf_trail_matrix_interp_dd[date][ind][it, :] = spl(V_range_common)
                            
                        
                    #### Normalize by the mean 
                    # ccf_trail_matrix_dd[date][ind][it, iVsys] = ccf_trail_matrix_dd[date][ind][it, iVsys] - np.mean(ccf_trail_matrix_dd[date][ind][it, iVsys])
                    # if iVsys == it == 0:
                    #     plt.figure()
                    #     plt.plot(datadetrend_dd[date]['data_wavsoln'][ind, :], datadetrend_dd[date]['datacube_mean_sub'][ind, it, :], label = 'Data')
                    #     plt.plot(datadetrend_dd[date]['data_wavsoln'][ind, :], model_spec_flux_shift, label = 'Model')
                    #     plt.legend()
                    #     plt.savefig(savedir + 'ccf_trail_model_data_comparison_'+date+ '_' + str(ind) + '_'+'with_reprocess.png', 
                    #                 format='png', dpi=300, bbox_inches='tight')
                        
                        
        # Sum the CCF across all orders for each date 
        for date in datadetrend_dd.keys():
            ccf_trail_total = np.zeros((nspec,nvel))
            for ind in order_inds:
                ccf_trail_total+=ccf_trail_matrix_dd[date][ind]
            ccf_trail_matrix_dd[date]['total'] = ccf_trail_total
            ccf_trail_matrix_dd[date]['phases'] = datadetrend_dd[date]['phases']
            ccf_trail_matrix_dd[date]['berv'] = datadetrend_dd[date]['berv']
        # ccf_trail_matrix_dd['Vsys_range'] = Vsys_range
        ccf_trail_matrix_dd['V_total'] = V_total # Vsys_range
        
        np.save(savedir + 'ccf_trail_matrix_with_model_reprocess.npy', ccf_trail_matrix_dd)
        
        # # Sum the CCF across all orders for each date ; for the one interp to common velocity range
        # for date in datadetrend_dd.keys():
        #     ccf_trail_interp_total = np.zeros((nspec,len(V_range_common)))
        #     for ind in order_inds:
        #         ccf_trail_interp_total+=ccf_trail_matrix_interp_dd[date][ind]
        #     ccf_trail_matrix_interp_dd[date]['total'] = ccf_trail_interp_total
        #     ccf_trail_matrix_interp_dd[date]['phases'] = datadetrend_dd[date]['phases']
        #     ccf_trail_matrix_interp_dd[date]['berv'] = datadetrend_dd[date]['berv']
        # # ccf_trail_matrix_dd['Vsys_range'] = Vsys_range
        # ccf_trail_matrix_interp_dd['V_range_common'] = V_range_common # Vsys_range
        
        # np.save(savedir + 'ccf_trail_matrix_with_model_reprocess_interp_common_velocity.npy', ccf_trail_matrix_interp_dd)
        
        
        ###### Plot the CC trail matrix to test 
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (15,5))
            
            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=ccf_trail_matrix_dd[date]['total'],
                                        phases=datadetrend_dd[date]['phases'],
                                        velocity_shifts=ccf_trail_matrix_dd['V_total'],
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total ; Date: '+ date ,
                                        setxlabel=True, plot_type = 'pcolormesh')
            fig.colorbar(hnd1, ax=axx)
            velocity_trail = []
            for it in range(len(datadetrend_dd[date]['phases'])):
                V_planet =  self.Kp_pred * np.sin(2. * np.pi * datadetrend_dd[date]['phases'][it]) + self.Vsys_pred + datadetrend_dd[date]['berv'][it]
                velocity_trail.append(V_planet)
            velocity_trail = np.array(velocity_trail)
            plt.plot(velocity_trail, datadetrend_dd[date]['phases'], color = 'w', lw = 1, linestyle = 'dashed') 
            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'$\phi$')
            axx.set_xlabel(r'V$_{sys}$ [km/s]')
            plt.savefig(savedir + 'ccf_total_trail_matrix_with_model_reprocess_date-' + date + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
        ###### Plot the CC trail matrix to test ; interpolated to common wavelength grid
        # for dt, date in enumerate(datelist):
        #     fig, axx = plt.subplots(figsize = (15,5))
            
        #     hnd1 = crocut.subplot_cc_matrix(axis=axx,
        #                                 cc_matrix=ccf_trail_matrix_interp_dd[date]['total'],
        #                                 phases=datadetrend_dd[date]['phases'],
        #                                 velocity_shifts=ccf_trail_matrix_interp_dd['V_range_common'],
        #                                 ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
        #                                 title= 'Total ; Date: '+ date ,
        #                                 setxlabel=True, plot_type = 'pcolormesh')
        #     fig.colorbar(hnd1, ax=axx)
        #     velocity_trail = []
        #     for it in range(len(datadetrend_dd[date]['phases'])):
        #         V_planet =  self.Kp_pred * np.sin(2. * np.pi * datadetrend_dd[date]['phases'][it]) + self.Vsys_pred + datadetrend_dd[date]['berv'][it]
        #         velocity_trail.append(V_planet)
        #     velocity_trail = np.array(velocity_trail)
        #     plt.plot(velocity_trail, datadetrend_dd[date]['phases'], color = 'w', lw = 1, linestyle = 'dashed') 
        #     # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
        #     axx.set_ylabel(r'$\phi$')
        #     axx.set_xlabel(r'V$_{sys}$ [km/s]')
        #     plt.savefig(savedir + 'ccf_total_trail_matrix_interp_with_model_reprocess_date-' + date + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #     plt.close()
            
        # return ccf_trail_matrix_dd
        
    
    def compute_2D_KpVsys_map(self, theta_fit_dd = None, posterior = 'median', datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, Kp_range = None, savedir = None, exclude_species = None, species_info = None, 
                             fixed_model_spec = None, fixed_model_wav = None, phase_range = None
                             ):
        """
        
        
        For a set of parameters inferred from the retrieval posteriors (stored in the dictionary theta_fit_dd), 
        compute the 2D cross-correlation map for a range of Kp and Vsys.
        

        :param theta_fit_dd: Dictionary of parameter values inferred from the posteriors, defaults to None
        :type theta_fit_dd: dict, optional
        
        :param posterior: Specify the type of parameters in theta_fit_dd with respect to the posterior; 
        'median', '+1sigma', '-1sigma', defaults to 'median'
        :type posterior: str, optional
        
        :param datadetrend_dd: Detrended data dictionary, defaults to None
        :type datadetrend_dd: dict, optional
        
        :param order_inds: Index of orders/detectors for which you want to compute the log-likelihood.
        :type order_inds: array of int
        
        :param Vsys_range: Range of Vsys, defaults to None
        :type Vsys_range: array_like
        
        :param Kp_range: Range of Vsys, defaults to None
        :type Kp_range: array_like
        
        :param savedir: path to the directory where you want to save the plot, defaults to None
        :type savedir: str, optional
        
        :param exclude_species: List of species you want to exclude the contribution from in the model, defaults to None
        :type exclude_species: array_like of str, optional
        
        :param species_info: Name of the species for which the model is being calculated, often just one species, defaults to None
        :type species_info: str, optional
        
        :return: dictionary storing all the Kp-Vsys maps
        :rtype: dict
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
                if pname in self.species or pname in ['P0', 'P1','P2']:
                    setattr(self, pname, 10.**theta_fit_dd[pname][postind])
                else:
                    setattr(self, pname, theta_fit_dd[pname][postind])
            #######################################################################
            #######################################################################
            ################# Due you want to zero out certain species to get the contribution of others? ########
            ########## For fix param condition (when theta_fit_dd is set to None), this happens outside. 
            # if exclude_species is not None:
            #     for spnm in exclude_species:
            #         abund_dict = copy.deepcopy(self.abundances_dict)
            #         abund_dict[spnm] = abund_dict[spnm] * 1e-30
            #         self.abundances_dict = abund_dict
        
        nKp, nVsys = len(Kp_range), len(Vsys_range)

        datelist = list(datadetrend_dd.keys()) 

        if self.use_stellar_phoenix:
            if fixed_model_spec is None:
                model_wav, model_Fp_orig = self.get_Fp_spectra(exclude_species=exclude_species)
            else:
                model_wav, model_Fp_orig = fixed_model_wav, fixed_model_spec
                
            phoenix_modelcube = self.get_phoenix_modelcube(datadetrend_dd = datadetrend_dd, 
                                                        model_phoenix_flux = self.phoenix_model_flux, 
                                                        model_phoenix_wav = self.phoenix_model_wav)
            ### Rotationally broaden the planetary spectrum 
            model_Fp_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                                       model_wav = model_wav, model_spec = model_Fp_orig)
            model_Fp = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
            model_spec = None
            model_FpFs_save = model_Fp/self.phoenix_model_flux
            
        else:
            if fixed_model_spec is None:
                model_wav, model_spec_orig = self.get_spectra(exclude_species=exclude_species)
            else:
                model_wav, model_spec_orig = fixed_model_wav, fixed_model_spec  
            
            # Rotationally broaden the spectrum 
            model_spec_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                            model_wav = model_wav, model_spec = model_spec_orig)
                 
            # Convolve the model to the instrument resolution already
            model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
            model_Fp = None
            phoenix_modelcube = {}
            for date in datelist:
                phoenix_modelcube[date] = None

            model_FpFs_save = model_spec
            

          
        # Dictionaries to store the R, C, and logL maps 
        R_dd, C_dd, logL_dd = {}, {}, {}
        
        # Loop over dates
        for dt, date in enumerate(datelist):
            print('Date: ', date)
            phases = datadetrend_dd[date]['phases']
            phase_mask = np.ones(len(phases), dtype = bool)
            if phase_range is not None:
                phase_mask = np.logical_and(phases > phase_range[0], phases < phase_range[1])
            print('Total phases: ', len(phases))
            print('Summing signal across: ', np.sum(phase_mask))
            
            # Dictionaries to store the R, C, and logL maps for this date
            R_dd[date], C_dd[date], logL_dd[date] = {}, {}, {}
            
            # Loop over all orders
            for ind in tqdm(order_inds):
                
                # Arrays to store the R, C, and logL maps for this date and order/detector
                R_dd[date][ind], C_dd[date][ind], logL_dd[date][ind] = np.empty((nKp, nVsys)), np.empty((nKp, nVsys)), np.empty((nKp, nVsys))
                
                # Loop over Kp and Vsys values
                for iKp, Kp_val in enumerate(Kp_range): 
                    for iVsys, Vsys_val in enumerate(Vsys_range):
                        
                        ## Set the Kp and Vsys value 
                        setattr(self, 'Kp', Kp_val)
                        setattr(self, 'Vsys', Vsys_val) 
                        nspec = datadetrend_dd[date]['datacube'][ind, :, :].shape[0]
                        # Compute the reprocessed model 
                        if self.use_stellar_phoenix:
                            phoenix_modelcube_inp = phoenix_modelcube[date][ind,:,:]
                        else:
                            phoenix_modelcube_inp = None
                        model_reprocess, avoid_mask = self.get_reprocessed_modelcube(model_spec = model_spec, model_wav = model_wav, 
                                                                                     
                                                                                     model_Fp = model_Fp, phoenix_modelcube = phoenix_modelcube_inp,
                                                                                     
                                                        datacube = datadetrend_dd[date]['datacube'][ind, :, :], 
                                                        datacube_detrended = datadetrend_dd[date]['datacube_detrended'][ind, :, :], 
                                                        data_wavsoln = datadetrend_dd[date]['data_wavsoln'][ind, :],
                                                pca_eigenvectors = datadetrend_dd[date]['pca_eigenvectors'][ind][:], 
                                                colmask = datadetrend_dd[date]['colmask'][ind, :],
                                                post_pca_mask = datadetrend_dd[date]['post_pca_mask'][ind, :],
                                                phases = datadetrend_dd[date]['phases'], berv = datadetrend_dd[date]['berv'])
                        
                        R_per_spec, C_per_spec, logL_per_spec = np.empty(nspec), np.empty(nspec), np.empty(nspec)

                        # Loop over time
                        for it in range(nspec):
                            # if phase_mask[it] == True:
                            model_spec_flux_shift = model_reprocess[it, :]
                            # Mean subtract the model with the zero values ignored
                            model_spec_flux_shift = crocut.sub_mask_1D(model_spec_flux_shift, avoid_mask)

                            # Compute R, C, logL accounting for the correct number of non-zero data points (only those that actually contribute to the CCF finitely, so all channels besides avoid_mask)
                            R_per_spec[it], C_per_spec[it], logL_per_spec[it] = crocut.fast_cross_corr(data=datadetrend_dd[date]['datacube_mean_sub'][ind, it, ~avoid_mask],
                                                                                                model=model_spec_flux_shift[~avoid_mask])
                            # else:
                            #     R_per_spec[it], C_per_spec[it], logL_per_spec[it] = 0., 0., 0.
                            
                        # Sum over all exposures
                        R_dd[date][ind][iKp, iVsys], C_dd[date][ind][iKp, iVsys], logL_dd[date][ind][iKp, iVsys] = np.sum(R_per_spec[phase_mask]), np.sum(C_per_spec[phase_mask]), np.sum(logL_per_spec[phase_mask]) 

                    
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
        KpVsys_save['model_spec'] = model_FpFs_save
        KpVsys_save['model_wav'] = model_wav
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
            np.save(savedir + 'KpVsys_dict_' + posterior + '_without_' + species_info, KpVsys_save)
        
        return KpVsys_save
    
    def compute_2D_KpVsys_map_fast_without_model_reprocess(self, theta_fit_dd = None, posterior = None, 
                                                           datadetrend_dd = None, order_inds = None, 
                             Vsys_range = None, Kp_range = None, savedir = None, exclude_species = None, 
                             species_info = None, vel_window = None, 
                            fixed_model_spec = None, fixed_model_wav = None, phase_range = None):
        """For a set of parameters inferred from the retrieval posteriors (stored in the dictionary theta_fit_dd), 
        compute the 2D cross-correlation map for a range of Kp and Vsys WITHOUT model reprocessing, using the 'fast' method.

        :param theta_fit_dd: Dictionary of parameter values inferred from the posteriors, defaults to None
        :type theta_fit_dd: dict, optional
        
        :param posterior: Specify the type of parameters in theta_fit_dd with respect to the posterior; 
        'median', '+1sigma', '-1sigma', defaults to 'median'
        :type posterior: str, optional
        
        :param datadetrend_dd: Detrended data dictionary, defaults to None
        :type datadetrend_dd: dict, optional
        
        :param order_inds: Index of orders/detectors for which you want to compute the log-likelihood.
        :type order_inds: array of int
        
        :param Vsys_range: Range of Vsys, defaults to None
        :type Vsys_range: array_like
        
        :param Kp_range: Range of Vsys, defaults to None
        :type Kp_range: array_like
        
        :param savedir: path to the directory where you want to save the plot, defaults to None
        :type savedir: str, optional
        
        :param exclude_species: List of species you want to exclude the contribution from in the model, defaults to None
        :type exclude_species: array_like of str, optional
        
        :param species_info: Name of the species for which the model is being calculated, often just one species, defaults to None
        :type species_info: str, optional
        
        :return: dictionary storing all the Kp-Vsys maps
        :rtype: dict
        
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
                if pname in self.species or pname in ['P0', 'P1','P2']:
                    setattr(self, pname, 10.**theta_fit_dd[pname][postind])
                else:
                    setattr(self, pname, theta_fit_dd[pname][postind])
            #######################################################################
            #######################################################################

        datelist = list(datadetrend_dd.keys())

        nKp, nVsys = len(Kp_range), len(Vsys_range)
        ################# Due you want to zero out certain species to get the contribution of others? ######## NOT IMPLMENTED YET : 13-06-2024
        # if exclude_species is not None:
        #     for spnm in exclude_species:
        #         # setattr(self, spnm, 10.**-30.)
        #         abund_dict = copy.deepcopy(self.abundances_dict)
        #         abund_dict[spnm] = abund_dict[spnm] * 1e-30
        #         self.abundances_dict = abund_dict
        
        ### Calculate the model_spec and model_wav which should be the same for all dates for this instrument (all taken in same mode : transmission or emission)
        if self.use_stellar_phoenix:
            if fixed_model_spec is None:
                model_wav, model_Fp_orig = self.get_Fp_spectra(exclude_species=exclude_species)
            else:
                model_wav, model_Fp_orig = fixed_model_wav, fixed_model_spec
            
            phoenix_modelcube = self.get_phoenix_modelcube(datadetrend_dd = datadetrend_dd, 
                                                        model_phoenix_flux = self.phoenix_model_flux, 
                                                        model_phoenix_wav = self.phoenix_model_wav) ## This includes broadening by instrument LSF
            ### Rotationally broaden the planetary spectrum 
            model_Fp_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                                       model_wav = model_wav, model_spec = model_Fp_orig)
            
            model_Fp = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_Fp_orig_broadened)
            model_spl = interpolate.make_interp_spline(model_wav, model_Fp, bc_type='natural')  
            model_FpFs_save = model_Fp/self.phoenix_model_flux

        else:
            if fixed_model_spec is None:
                model_wav, model_spec_orig = self.get_spectra(exclude_species=exclude_species)
            else:
                model_wav, model_spec_orig = fixed_model_wav, fixed_model_spec
                
            # Rotationally broaden the spectrum 
            model_spec_orig_broadened, _ = self.rotation(vsini = self.vsini_planet, 
                                            model_wav = model_wav, model_spec = model_spec_orig)   
            model_spec = self.convolve_spectra_to_instrument_resolution(model_spec_orig=model_spec_orig_broadened)
            model_Fp = None
            phoenix_modelcube = {}
            for date in datelist:
                phoenix_modelcube[date] = None
            model_spl = interpolate.make_interp_spline(model_wav, model_spec, bc_type='natural')   
            model_FpFs_save = model_spec
        
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
            if self.use_stellar_phoenix:
                phoenix_modelcube_this_date = phoenix_modelcube[date]
            
            cc_matrix_all_orders, logL_matrix_all_orders = np.zeros((len(order_inds), nspec, nVsys)), np.zeros((len(order_inds), nspec, nVsys))
            ## Loop over orders
                
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
                        # model_spec_flux_shift = splev(data_wavsoln_shift, model_spl)
                        model_spec_flux_shift = model_spl(data_wavsoln_shift)
                        
                        # if it == 0 and iv ==0:
                            # plt.figure(figsize = (10,8))
                            # plt.plot(data_wavsoln_shift, model_spec_flux_shift/phoenix_modelcube_this_date[ind,it, :], color = 'r', label = 'FpFs')
                            # plt.legend()
                            # plt.savefig(savedir + 'FpFs_'+str(ind)+'.png', format='png', dpi=300, bbox_inches='tight')  
                            
                            # plt.figure(figsize = (10,8))
                            # plt.plot(data_wavsoln_shift, model_spec_flux_shift, color = 'r', label = 'Fp')
                            # plt.legend()
                            # plt.savefig(savedir + 'Fp_'+str(ind)+'.png', format='png', dpi=300, bbox_inches='tight')  
                            
                            # plt.figure(figsize = (10,8))
                            # plt.plot(data_wavsoln_shift, phoenix_modelcube_this_date[ind,it, :], color = 'r', label = 'Fs')
                            # plt.legend()
                            # plt.savefig(savedir + 'Fs_'+str(ind)+'.png', format='png', dpi=300, bbox_inches='tight') 
                        
                        # plt.close('all')
                        
                        if self.use_stellar_phoenix:
                            model_spec_flux_shift = model_spec_flux_shift/phoenix_modelcube_this_date[ind,it, :]
                        
                        # Subtract the mean from the model
                        model_spec_flux_shift = model_spec_flux_shift - crocut.fast_mean(model_spec_flux_shift)
                        #### Compute the cross correlation value between the shifted model and the data
                        
                        if it == 0 and iv == 0:
                            plt.suptitle('Data vs Model; exposure ' + str(it) )
                            plt.figure(figsize=(12, 6))
                            plt.subplot(1, 2, 1)
                            plt.plot(data_wavsoln[ind, ~avoid_mask], datacube_mean_sub[ind, it, ~avoid_mask], color='k', label='data')
                            plt.legend()
                            plt.xlabel('Wavelength [nm]')
                            plt.ylabel('Flux')
                            plt.title('Data')

                            plt.subplot(1, 2, 2)
                            plt.plot(data_wavsoln_shift[~avoid_mask], model_spec_flux_shift[~avoid_mask], color='r', label='model')
                            plt.legend()
                            plt.xlabel('Wavelength [nm]')
                            plt.ylabel('Flux')
                            plt.title('Model')
                            # plt.figure(figsize = (10,8))
                            # plt.plot(data_wavsoln[ind,~avoid_mask], datacube_mean_sub[ind,it,~avoid_mask], color = 'k', label = 'data')
                            # plt.plot(data_wavsoln_shift[~avoid_mask], model_spec_flux_shift[~avoid_mask], color = 'r', label = 'model')
                            plt.legend()
                            plt.savefig(savedir + 'data_model_comp_order_'+str(ind)+'.png', format='png', dpi=300, bbox_inches='tight')  
                        plt.close('all')
                        _, cc_matrix_all_orders[i_ind,it,iv], logL_matrix_all_orders[i_ind,it,iv] = crocut.fast_cross_corr(data=datacube_mean_sub[ind,it,~avoid_mask], 
                                                                                                                     model=model_spec_flux_shift[~avoid_mask])
                        
            CC_matrix_all_dates[dt] , logL_matrix_all_dates[dt] = np.sum(cc_matrix_all_orders, axis = 0), np.sum(logL_matrix_all_orders, axis = 0)              

        
        ###### Plot the CC trail matrix to test 
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (15,5))
            
            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=CC_matrix_all_dates[dt],
                                        phases=datadetrend_dd[date]['phases'],
                                        velocity_shifts=Vsys_range,
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total ; Date: '+ date ,
                                        setxlabel=True, plot_type = 'pcolormesh')
            fig.colorbar(hnd1, ax=axx)
            velocity_trail = []
            for it in range(len(datadetrend_dd[date]['phases'])):
                V_planet =  self.Kp * np.sin(2. * np.pi * datadetrend_dd[date]['phases'][it]) + self.Vsys + datadetrend_dd[date]['berv'][it]
                velocity_trail.append(V_planet)
            velocity_trail = np.array(velocity_trail)
            plt.plot(velocity_trail, datadetrend_dd[date]['phases'], color = 'w', lw = 1, linestyle = 'dashed')
            if self.method == 'transmission':
                plt.axhline(y = phase_range[0], color = 'w', lw = 2, linestyle = 'dotted')
                plt.axhline(y = phase_range[1], color = 'w', lw = 2, linestyle = 'dotted')
            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'$\phi$')
            axx.set_xlabel(r'V$_{sys}$ [km/s]')
            plt.savefig(savedir + 'ccf_total_trail_matrix_fast_date-' + date + '.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
        
        ####### Repeat the CCF trail matrix plotting, but now with each frame normalized by its median
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (15,5))
            cc_matrix = CC_matrix_all_dates[dt]
            # Normalize each row by its median
            cc_matrix_normalized = np.zeros_like(cc_matrix)
            for it in range(cc_matrix.shape[0]):
                cc_matrix_normalized[it, :] = cc_matrix[it, :] - np.median(cc_matrix[it, :])
            # Plot the normalized matrix
            hnd1 = crocut.subplot_cc_matrix(axis=axx,
                                        cc_matrix=cc_matrix_normalized,
                                        phases=datadetrend_dd[date]['phases'],
                                        velocity_shifts=Vsys_range,
                                        ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                        title= 'Total ; Date: '+ date ,
                                        setxlabel=True, plot_type = 'pcolormesh')
            fig.colorbar(hnd1, ax=axx)
            velocity_trail = []
            for it in range(len(datadetrend_dd[date]['phases'])):
                V_planet =  self.Kp * np.sin(2. * np.pi * datadetrend_dd[date]['phases'][it]) + self.Vsys + datadetrend_dd[date]['berv'][it]
                velocity_trail.append(V_planet)
            velocity_trail = np.array(velocity_trail)
            plt.plot(velocity_trail, datadetrend_dd[date]['phases'], color = 'w', lw = 1, linestyle = 'dashed') 
            if self.method == 'transmission':
                plt.axhline(y = phase_range[0], color = 'w', lw = 2, linestyle = 'dotted')
                plt.axhline(y = phase_range[1], color = 'w', lw = 2, linestyle = 'dotted') 
            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'$\phi$')
            axx.set_xlabel(r'V$_{sys}$ [km/s]')
            plt.savefig(savedir + 'ccf_total_trail_matrix_fast_normalized_date-' + date + '.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
        
        ###### Plot the logL trail matrix to test
        for dt, date in enumerate(datelist):
            fig, axx = plt.subplots(figsize = (15,5))
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
            axx.set_xlabel(r'V$_{sys}$ [km/s]')
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
            print('Date', date)
            CC_KpVsys, logL_KpVsys = np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nKp, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
            phases = datadetrend_dd[date]['phases']
            berv = datadetrend_dd[date]['berv']
            nspec = len(phases)
            print('phases:', phases)
            phase_mask = np.ones(len(phases), dtype = bool)
            if phase_range is not None:
                phase_mask = np.logical_and(phases > phase_range[0], phases < phase_range[1])
            print('Total phases: ', len(phases))
            print('Summing signal across: ', np.sum(phase_mask))
        
            for iKp, Kp in enumerate(Kp_range):
                CC_matrix_shifted, logL_matrix_shifted = np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) )), np.zeros((nspec, len(Vsys_range[vel_window[0]:vel_window[1]]) ))
                for it in range(nspec):
                    if phase_mask[it] == True:
                        Vp = Kp * np.sin(2. * np.pi * phases[it])
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
        KpVsys_save['model_spec'] = model_FpFs_save
        KpVsys_save['model_wav'] = model_wav
        KpVsys_save['model_wav'] = model_wav
        KpVsys_save['logL'] = logL_KpVsys_total
        KpVsys_save['cc'] = CC_KpVsys_total
        KpVsys_save['Kp_range'] = Kp_range
        KpVsys_save['Vsys_range'] = Vsys_range
        KpVsys_save['vel_window'] = vel_window
        KpVsys_save['Vsys_range_windowed'] = Vsys_range[vel_window[0]:vel_window[1]]
        
        
        if species_info is None:
            np.save(savedir + 'KpVsys_fast_no_model_reprocess_dict.npy', KpVsys_save)
        else:
            plt.figure()
            plt.plot(KpVsys_save['model_wav'], KpVsys_save['model_spec'])
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Fp/Fs')
            plt.savefig(savedir + 'model_without_' + species_info + '.png', format = 'png', dpi = 300)
            np.save(savedir + 'KpVsys_fast_no_model_reprocess_dict' + '_without_' + species_info + '.npy', KpVsys_save)
        
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
                # axx[ip].vlines(x=theta_fit_dd['Vsys'][postind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][postind]-5., color='k', linestyle='dashed')
                # axx[ip].vlines(x=theta_fit_dd['Vsys'][postind], ymin=theta_fit_dd['Kp'][postind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                # axx[ip].hlines(y=theta_fit_dd['Kp'][postind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][postind]-5., color='k', linestyle='dashed')
                # axx[ip].hlines(y=theta_fit_dd['Kp'][postind], xmin=theta_fit_dd['Vsys'][postind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')
                axx[ip].axvline(x=theta_fit_dd['Vsys'][postind], color='w', linestyle='dashed')
                axx[ip].axhline(y=theta_fit_dd['Kp'][postind], color='w', linestyle='dashed')
        else:
            for ip in [0,1]:
                axx[ip].axvline(x=self.Vsys_pred, color='w', linestyle='dashed')
                axx[ip].axhline(y=self.Kp_pred, color='w', linestyle='dashed')
                # axx[ip].vlines(x=self.Vsys_pred, ymin=KpVsys_save['Kp_range'][0], ymax=self.Kp_pred-5., color='w', linestyle='dashed')
                # axx[ip].vlines(x=self.Vsys_pred, ymin=self.Kp_pred+5., ymax=KpVsys_save['Kp_range'][-1], color='w', linestyle='dashed')

                # axx[ip].hlines(y=self.Kp_pred, xmin=KpVsys_save['Vsys_range_windowed'][0], xmax=self.Vsys_pred-5., color='w', linestyle='dashed')
                # axx[ip].hlines(y=self.Kp_pred, xmin=self.Vsys_pred+5., xmax=KpVsys_save['Vsys_range_windowed'][-1], color='w', linestyle='dashed')
                

            

        axx[0].set_ylabel(r'K$_{P}$ [km/s]')
        axx[0].set_xlabel(r'V$_{sys}$ [km/s]')
        axx[1].set_ylabel(r'K$_{P}$ [km/s]')
        axx[1].set_xlabel(r'V$_{sys}$ [km/s]')
        
        if species_info is None:
            plt.savefig(savedir + 'KpVsys_fast_no_model_reprocess.png', format='png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(savedir + 'KpVsys_fast_no_model_reprocess' + '_without_' + species_info + '.png', format='png', dpi=300, bbox_inches='tight')

                    
    ##############################################################################################################
    ##############################################################################################################
    ## Function to plot KpVsys maps ##############################################################################
    ##############################################################################################################
    ##############################################################################################################
    def plot_KpVsys_maps(self, KpVsys_save = None, posterior = 'median', theta_fit_dd = None, savedir = None, species_info = None):
        """Plot all the maps given the KpVsys_save dictionary returned by the compute_2D_KpVsys_map_ above.

        :param KpVsys_save: KpVsys save dictionary, defaults to None
        :type KpVsys_save: dict
        
        :param posterior: Specify the type of parameters in theta_fit_dd with respect to the posterior; 
        'median', '+1sigma', '-1sigma', defaults to 'median'
        :type posterior: str, optional
        
        :param theta_fit_dd: Dictionary of parameter values inferred from the posteriors, defaults to None
        :type theta_fit_dd: dict, optional
        
        :param savedir: path to the directory where you want to save the plot, defaults to None
        :type savedir: str, optional
        
        :param species_info: Name of the species for which the model is being calculated, often just one species, defaults to None
        :type species_info: str, optional
        
        """
        
        if KpVsys_save is None:
            if species_info is None:
                KpVsys_save = np.load(savedir + 'KpVsys_dict_' + posterior + '.npy', allow_pickle = True).item()
            else:
                KpVsys_save = np.load(savedir + 'KpVsys_dict_' + posterior + '_without_' + species_info + '.npy', allow_pickle = True).item()
        
        datelist = list(KpVsys_save['all_dates']['logL'].keys())
        print(datelist)
        
        if theta_fit_dd is not None:
        ## Define the index of the theta_fit_dd to use depending on if you are doing the computation for median, +1sigma, or -1sigma values of the posterior. 
            if posterior == 'median':
                postind = 0
            elif posterior == '-1sigma':
                postind = 1
            elif posterior == '+1sigma':
                postind = 2
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
                    axx[dd].vlines(x=theta_fit_dd['Vsys'][postind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][postind]-5., color='k', linestyle='dashed')
                    axx[dd].vlines(x=theta_fit_dd['Vsys'][postind], ymin=theta_fit_dd['Kp'][postind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                    axx[dd].hlines(y=theta_fit_dd['Kp'][postind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][postind]-5., color='k', linestyle='dashed')
                    axx[dd].hlines(y=theta_fit_dd['Kp'][postind], xmin=theta_fit_dd['Vsys'][postind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')

                # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
                axx[dd].set_ylabel(r'K$_{P}$ [km/s]')
                axx[dd].set_xlabel(r'V$_{sys}$ [km/s]')

                # axx[dd].set_xlim(xmin = velocity_shifts_win[0], xmax = velocity_shifts_win[-1])
                # axx[dd].set_ylim(ymin=Kp_range[0], ymax=Kp_range[-1])

            if species_info is None:
                plt.savefig(savedir + 'all_dates_' + mk + '_'+posterior+'_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savedir + 'all_dates_' + mk + '_'+posterior+'_without_'+species_info+'_.pdf', format='pdf', dpi=300, bbox_inches='tight')

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
            
                axx.vlines(x=theta_fit_dd['Vsys'][postind], ymin=KpVsys_save['Kp_range'][0], ymax=theta_fit_dd['Kp'][postind]-5., color='k', linestyle='dashed')
                axx.vlines(x=theta_fit_dd['Vsys'][postind], ymin=theta_fit_dd['Kp'][postind]+5., ymax=KpVsys_save['Kp_range'][-1], color='k', linestyle='dashed')

                axx.hlines(y=theta_fit_dd['Kp'][postind], xmin=KpVsys_save['Vsys_range'][0], xmax=theta_fit_dd['Vsys'][postind]-5., color='k', linestyle='dashed')
                axx.hlines(y=theta_fit_dd['Kp'][postind], xmin=theta_fit_dd['Vsys'][postind]+5., xmax=KpVsys_save['Vsys_range'][-1], color='k', linestyle='dashed')


            # axx[1].plot(velocity_shifts, cc_matrix_sum[:])
            axx.set_ylabel(r'K$_{P}$ [km/s]')
            axx.set_xlabel(r'V$_{sys}$ [km/s]')

            # axx.set_xlim(xmin=velocity_shifts_win[0], xmax=velocity_shifts_win[-1])
            # axx.set_ylim(ymin=Kp_range[0], ymax=Kp_range[-1])
            
            if species_info is None:
                plt.savefig(savedir + 'total_'+ mk + '_'+ posterior+ '_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(savedir + 'total_'+ mk + '_'+ posterior+ '_without_' + species_info + '_.pdf', format='pdf', dpi=300, bbox_inches='tight')
            
        plt.close('all')