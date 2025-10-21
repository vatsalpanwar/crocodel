"""
Compute the spectrum from the planet alone, without the contribution from the star. 
Defines the Planet class, which takes in the planet properties from an input config file,
and generates either the transmission radius (Rp(wavelength)) or the emission flux (Fp(wavelength)). 
Both of these outputs can be used by StellarGrid object to compute the 
observed transmission spectrum (Rp/Rs)**2 (wavelength), or emission spectrum Fp/Fs (wavelength). 
 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
import copy
from genesis import genesis
import pyfastchem

FAST_CHEM_DIR = '/home/astro/phsprd/code/crocodel/fastchem_inputs/'

class Model_LR():
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the PlanetAtmosphere class with the provided arguments, 
        to compute the spectrum of the planet (in transmission: Rp, in emission: Fp).
        
        Parameters:
        - *args: Positional arguments.
        - **kwargs: Keyword arguments.
        """
        ##### Define some constants in SI units.
        self.vel_c_SI = 299792458.0
        self.k_B_cgs = 1.380649e-16
        
        self.data = kwargs.pop('data')
        
        self.planet_dict = kwargs.pop('config_dd')
        
        # self.planet_dict = kwargs.pop('planet_dict')
        # self.simulation_dict = kwargs.pop('simulation_dict')
        # self.star_dict = kwargs.pop('star_dict')
        # self.wavelength_solution = kwargs.pop('wavelength_solution')
        # Type of TP profile parametrization
        
        self.free_param_dict = self.planet_dict['free_params']
        
        self.TP_type = self.planet_dict['TP_type']
        self.method = self.planet_dict['method']
        
        self.P_min = self.planet_dict['P_min'] # Minimum pressure level for model calculation, in bars 
        self.P_max = self.planet_dict['P_max'] # Maximum pressure level for model calculation, in bars 
        self.N_layers = self.planet_dict['N_layers'] # Number of pressure layers in the atmosphere 
        self.spacing = self.planet_dict['spacing'] # Wavelength grid spacing, use 'R' for constant resolving power

        self.lam_min = self.planet_dict['lam_min'] # Minimum wavelength for model calculation, in microns
        self.lam_max = self.planet_dict['lam_max'] # Maximum wavelength for model calculation, in microns
        # elif self.wavelength_solution is not None and self.spacing == None:
        #     self.lam_min, self.lam_max = None, None
        
        # self.lam_min, self.lam_max = None, None 
        self.resolving_power = self.planet_dict['model_resolution_working'] # Resolving power for the model calculation (use 250000 which will later be convolved down)
        
        self.fix_MMW = self.planet_dict['fix_MMW']
        self.MMW_value = self.planet_dict['MMW_value']
        self.chemistry = self.planet_dict['chemistry']
        
        # Load the names of all the absorbing species to be included in the model
        self.species = np.array(list(self.planet_dict['abundances'].keys()))
        self.species_name_fastchem = self.planet_dict['species_name_fastchem']
        if self.planet_dict['include_cia'] is not None:
            self.include_cia = self.planet_dict['include_cia']
        else:
            self.include_cia = True
        if self.chemistry == 'eq_chem':
            #create a FastChem object
            
            #it needs the locations of the element abundance and equilibrium constants files
            self.include_condensation = self.planet_dict['include_condensation']
            
            
            if self.include_condensation:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR + 'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR + 'input/logK/logK.dat',
                FAST_CHEM_DIR + 'input/logK/logK_condensates.dat',
                1)
            else:
                self.fastchem = pyfastchem.FastChem(
                FAST_CHEM_DIR + 'input/element_abundances/asplund_2020.dat',
                FAST_CHEM_DIR + 'input/logK/logK.dat',
                1)
            
            # Make a copy of the solar abundances from FastChem
            self.solar_abundances = np.array(self.fastchem.getElementAbundances())
            
            
            self.logZ_planet = self.planet_dict['logZ_planet']
            self.C_to_O = self.planet_dict['C_to_O']
            self.use_C_to_O = self.planet_dict['use_C_to_O']
            
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

        if self.TP_type in ['Linear', 'Linear_force_inverted', 'Linear_force_non_inverted']:
            self.P2 = 10.**self.planet_dict['P2']
            self.T2 = self.planet_dict['T2']
            self.P1 = 10.**self.planet_dict['P1']
            self.T1 = self.planet_dict['T1']
        elif self.TP_type == 'Linear_3_point':
            self.P2 = 10.**self.planet_dict['P2']
            self.T2 = self.planet_dict['T2']
            self.P1 = 10.**self.planet_dict['P1']
            self.T1 = self.planet_dict['T1']
            self.P0 = 10.**self.planet_dict['P0']
            self.T0 = self.planet_dict['T0']
        
        # Planet properties 
        self.R_planet= self.planet_dict['R_planet'] # Radius of the planet, in terms of R_Jup
        self.vsini_planet = self.planet_dict['vsini_planet']
        self.log_g= self.planet_dict['log_g'] # Surface gravity, log_g [cgs]
        self.P_ref= self.planet_dict['P_ref'] # Reference pressure, in log10 bars
        self.cl_P = self.planet_dict['cl_P'] # log10(cloud_pressure) in bars 
        self.log_fs = self.planet_dict['log_fs'] # model scale factor 
        self.phase_offset = self.planet_dict['phase_offset']
        self.R_star = self.planet_dict['R_star']
        # self.Kp = self.planet_dict['Kp'] # Current value of Kp, in km/s
        # self.Vsys = self.star_dict['Vsys'] # Current value of Vsys, in km/s
        
        # self.Kp_pred = self.Kp # Expected value of Kp, in km/s
        # self.Vsys_pred = self.Vsys # Expected value of Vsys, in km/s
        
        # Set the initial abundances for each species as specified under 'abundances' in croc_config.yaml 
        for sp in self.species:
            setattr(self, sp, 10.**self.planet_dict['abundances'][sp])
        
        # Instantiate GENESIS only once based on the given model properties
        print(self.lam_min, self.lam_max)
        self.Genesis_instance = genesis.Genesis(self.P_min, self.P_max, 
                                                self.N_layers, self.lam_min, 
                                                self.lam_max, self.resolving_power, 
                                                self.spacing, 
                                                method = self.method)


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
            'h2s': 34.08,
            'nh3': 17.03052,
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
                
        elif self.chemistry == 'eq_chem':
            number_densities = self.get_eqchem_abundances()
            #total gas particle number density from the ideal gas law 
            #Needed to convert the number densities output from FastChem to mixing ratios
            gas_number_density = ( ( press ) *1e6 ) / ( self.k_B_cgs * temp )
            
            ####### Extracting the h_minus from the Fastchem itself.
            for sp in self.species:
                # print(sp)
                vmr = number_densities[:, self.species_fastchem_indices[sp]]/gas_number_density
                X[sp] = vmr 
            vmr_h2 = number_densities[:, self.species_fastchem_indices["h2"]]/gas_number_density
            X["h2"] = vmr_h2
            
        assert all(X["h2"] >= 0.) # make sure that the hydrogen abundance is not going negative!   
        return X
        
    def get_Fp_or_Rp_by_Rs_2(self, exclude_species = None):
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
            ## Return Rp
            spec = self.gen.genesis(abund_dict, cl_P = self.cl_P, include_cia = self.include_cia)
            spec = spec/((self.R_star*6.957e8)**2.0)
        elif self.method == 'emission':
            ## Return Fp
            spec = self.gen.genesis(abund_dict, include_cia = self.include_cia)

        return (10**9) * self.gen.lam, 10**self.log_fs * spec
    
    def logL(self, theta, inst_list = None):
        
        for i, pname in enumerate(self.free_param_dict.keys()):
            setattr(self, pname, theta[i])
            
        lam_nm, model = self.get_Fp_or_Rp_by_Rs_2()
        ll = 0.
        for key in inst_list:
            ll += self.data.inst[key].loglike( lam_nm, model)
        # print(ll)
        return ll