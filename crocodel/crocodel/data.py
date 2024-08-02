import sys 
import numpy as np
import yaml
import genesis
import scipy.constants as sc
from scipy.interpolate import splev, splrep
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve

from . import stellcorrection_utils as stc
from . import cross_correlation_utils as crocut
from . import astro_utils as aut

from tqdm import tqdm 
import time
import pdb

from astropy.io import ascii as ascii_reader

class Data:
    """Data class to contain and analyse typical high-resolution cross-correlation spectroscopy dataset.
    """
    def __init__(self, *args, **kwargs):
        
        ## Load the information from the croc_config.yaml file taken as input when initiating this class.
        with open(kwargs.pop('config')) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
            
            # Define the dictionary 'N_PCA_dd' which will specify the value of N_PCA desired to be used 
            # for each instrument, date, and order.
            
            # Get the list of instruments.
            inst_list = self.get_instrument_list
            
            # Initiate the dictionary N_PCA_dd
            N_PCA_dd = {}
            
            # Construct the N_PCA_dd
            for inst in inst_list:
                # If it is desired to use common values of N_PCA across all dates and orders, 
                # then populate the same value for each date and order.
                if self.config['data'][inst]['stellcorr_params']['N_PCA_all_order_common']:
                    N_PCA_dd[inst] = {}
                    for date in self.get_dates(inst):
                        N_PCA_dd[inst][date] = {}
                        for iord in range(self.config['data'][inst]['N_order_all']):# in self.get_use_order_inds(inst = inst, date = date):
                            N_PCA_dd[inst][date][iord] = self.config['data'][inst]['stellcorr_params']['N_PCA_all_order_common_value']
                
                # If not, then take the manually defined N_PCA_dd from the config file.
                else:
                    N_PCA_dd[inst] = self.config['data'][inst]['stellcorr_params']['N_PCA_dd']
                            
            self.N_PCA_dd = N_PCA_dd

            # Based on the info in croc_config.yaml file, construct a dictionary per instrument, date, and order of 
            # the spectral channels that are desired to be masked before performing the PCA. This will take care of not masking any channels 
            # if it is specified so in the croc_config.yaml file by setting the 'pre_pca_mask_flux_threshold' keyword to 'none'.
            pre_pca_mask_dd = {}
            for inst in inst_list:
                pre_pca_mask_dd[inst] = {}
                for date in self.get_dates(inst):
                    pre_pca_mask_dd[inst][date] = {}
                    for iord in range(self.config['data'][inst]['N_order_all']):
                        pre_pca_mask_dd[inst][date][iord] = self.create_pre_pca_mask_deep_tellurics_per_order(inst = inst, date = date, order_ind = iord)
       
            self.pre_pca_mask_dd = pre_pca_mask_dd
            
            
            
    
    @property
    def get_instrument_list(self):
        """Get the list of all instruments in the dataset.

        :return: Array of instrument names.
        :rtype: array
        """
        return np.array(list(self.config['data'].keys()))
    
    def get_dates(self, inst):
        """For a given instrument, return the dates of all observations specified in the yaml file.

        :param inst: Name of the instrument. 
        :type inst: str
        :return: Array of the dates observed by the instrument. 
        :rtype: array
        """
        return list(self.config['data'][inst]['dates'].keys()) 
    
    @property  
    def get_spdatacubes_dict(self):
        """Method to construct and return the dictionary containing the spectral datacubes i.e. spdatacubes 
        from all instruments and all dates. 
        
        :return: Dictionary containing all the spdatacubes.
        :rtype: dict
        """
        spdatacubes_dict = {}
        for inst in self.get_instrument_list:
            spdatacubes_dict[inst] = {}
            for date in self.get_dates(inst):
                spdatacubes_dict[inst][date] = np.load(self.config['data'][inst]['dates'][date][0], allow_pickle = True).item()
        return spdatacubes_dict
    
    def get_use_order_inds(self, inst = None, date = None):
        """Get the array of order indices being used, accounting for any that are desired to be skipped.

        :param inst: Name of the instrument, defaults to None
        :type inst: str
        
        :param date: Date, as specified in the croc_config.yaml file, defaults to None
        :type date: str
        
        :return: Array of order indices to be used.
        :rtype: array_like
        """

        spdatacubes_dd = self.get_spdatacubes_dict[inst][date]
        use_orders = []
        for i in range(spdatacubes_dd['spdatacube'].shape[0]):
            if i not in self.config['data'][inst]['skip_order_inds']:
                use_orders.append(i)
        return np.array(use_orders)
    
    def create_pre_pca_mask_deep_tellurics_per_order(self, inst = None, date = None, order_ind = None):
        
        """If pre_pca_mask_flux_threshold in the config file is not 'none', 
        then create a mask for each detector and date based on the flux threshold specified i.e. if flux in a spectral pixel channel 
        goes below this, that pixel will be masked out.
         
        This will be merged with colmask before doing PCA because colmask is also effectively a pre PCA mask.
        
        :param inst: Name of the instrument, defaults to None
        :type inst: str
        
        :param date: Date, as specified in the croc_config.yaml file, defaults to None
        :type date: str
        
        :param order_ind: order_ind
        :type date: int
        
        :return: 1D mask.
        :rtype: array_like of bool
        
        """
        
        spdatacubes_dd = self.get_spdatacubes_dict[inst][date]
        datacube = spdatacubes_dd['spdatacube'][order_ind, :, :]
        if self.config['data'][inst]['stellcorr_params']['data_norm_already']:
            data_mean_norm = np.mean(datacube, axis = 0) 
        else:
            data_mean_norm = np.mean(datacube, axis = 0)/np.mean(np.mean(datacube, axis = 0))
        pre_pca_mask = np.zeros(datacube.shape[1], dtype=bool) ## Just like colmask, True are for the indices you want to mask OUT. 
        
        if self.config['data'][inst]['stellcorr_params']['pre_pca_mask_flux_threshold'] != 'none':
            for ipix in range(len(data_mean_norm)):
                if data_mean_norm[ipix] < self.config['data'][inst]['stellcorr_params']['pre_pca_mask_flux_threshold']:
                    pre_pca_mask[ipix] = True
                        
        return pre_pca_mask
        
          
    def pca_per_order_fast(self, inst = None, date = None, order_ind = None):
        """Perform PCA on a datacube for a single order from a single date.
        
        :param inst: Instrument name.
        :type inst: str
        
        :param date: Date, in YYYY-MM-DD format. Should be the same as the key used to define in the config file, defaults to None
        :type date: str
        
        :param order_num: Order index you want to perform the pca order for.
        :type order_num: int
        
        :return: colmask, post_pca_mask, data_wavsoln, datacube, datacube_fit, datacube_detrended_post_pca_mask, datacube_mean_sub, pca_eigenvectors
        :rtype: array_like
        
        """
        
        # Create the colmask from the colmask info in the croc_config.yaml file. colmask_info is a list of 2 element lists specifying the range 
        # pixels to be masked out.
        colmask_info = self.config['data'][inst]['stellcorr_params']['badcolmask_inds'][date][order_ind]
        colmask_inds = []
        for i in range(len(colmask_info)):
            inds = np.arange(colmask_info[i][0], colmask_info[i][1])
            colmask_inds.extend(inds)
        colmask_inds = np.array(colmask_inds)
        
        # Get the N_PCA for this date and order from the N_PCA_dd dictionary. 
        N_PCA = self.N_PCA_dd[inst][date][order_ind]

        # Get the spdatacube dictioary for this instrument and date.
        spdatacubes_dd = self.get_spdatacubes_dict[inst][date]
        
        # Extract the wavelength solution of the data for this order.
        data_wavsoln = spdatacubes_dd['wavsoln'][order_ind,:]
        
        # Extract the datacube for this order.
        datacube = spdatacubes_dd['spdatacube'][order_ind, :, :]
        
        # Extract the info on number of spectral exposures (nspec) and number of spectral channels (nwav).
        nspec, nwav = datacube.shape[0], datacube.shape[1]
        
        # Standardise the datacube first
        datacube_standard = stc.standardise_data(datacube)
        
        # Create a column mask for bad pixels using the compiled colmask_inds, same for all dates and detectors
        colmask_ = np.zeros(datacube.shape[1], dtype=bool)
        if len(colmask_inds) > 0:
            colmask_[colmask_inds] = True

        ##############################################
        ## Combine the colmask_ with the pre_pca_mask for this date and order.
        ##############################################
        colmask = np.logical_or(colmask_, self.pre_pca_mask_dd[inst][date][order_ind])
        
        # Initiate arrays for storing pca_eigenvectors, multi-linear regression fit to the datacube, and the detrended datacube.
        pca_eigenvectors = np.empty((nspec,N_PCA+1))
        datacube_fit = np.empty((nspec, nwav))
        datacube_detrended = np.empty((nspec, nwav))
        
        # Create a copy of the standardised datacube.
        fStd = datacube_standard.copy()
        # Zero out the spectral channels that you want to skip for the PCA (combined bad pixel and other specified pre_PCA mask)
        fStd[:, colmask] = 0
        
        # Perform the SVD and get the PCA eigenvectors.
        pca_eigenvectors = stc.get_eigenvectors_via_PCA_Matteo(fStd[:, colmask == False], nc=N_PCA)

        # Perform the multilinear regression to the datacube (note that this is not the standardised datacube but the raw datacube)
        datacube_fit = stc.linear_regression(X=pca_eigenvectors, Y=datacube)
        datacube_detrended = datacube/datacube_fit - 1.

        ## Apply post PCA mask to zero out wavelength channels that were not corrected properly. 
        # Automatically detect these channels by checking if the variance of a channel is larger than sigma times 
        # median variance of all channels, where sigma is the threshold probability below which such a deviation
        # could have occured by random chance. You could use stdev here instead of variance (defined using the post_pca_mask_thershold in the croc_config.yaml file), 
        # it will just be less stricter filter in that case. 
        datacube_detrended_post_pca_mask, post_pca_mask =  stc.mask_data_post_pca_per_order(datacube_detrended, maskval = 0., 
                                                                                            threshold = self.config['data'][inst]['stellcorr_params']['post_pca_mask_threshold']) ###  
        
        # Also zero out the pre_pca_mask channels (bad columns and other pre_pca_mask channels using the combined colmask) 
        datacube_detrended_post_pca_mask[:, colmask] = 0
        
        # Mean subtract the data, excluding the zeroed out values from the the mean calculation.
        datacube_mean_sub = np.empty((nspec, nwav))
        
        # Tile the 1D zeroed out masks in the time dimension so that sub_mask function can take it as an input
        zeroMask = np.tile( np.logical_or(post_pca_mask, colmask), (nspec,1) )
        datacube_mean_sub = crocut.sub_mask(datacube_detrended_post_pca_mask, zeroMask)
        
        return colmask, post_pca_mask, data_wavsoln, datacube, datacube_fit, datacube_detrended_post_pca_mask, datacube_mean_sub, pca_eigenvectors
