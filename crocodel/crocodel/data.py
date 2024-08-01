import sys 
# sys.path.insert(0, "/home/astro/phsprd/code/genesis/code") 
# sys.path.insert(0, "/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/genesis/code/opac")

# sys.path.insert(0, "/Users/vatsalpanwar/source/work/astro/projects/Warwick/code/genesis/code/")

import numpy as np
import yaml
import genesis
# from opac.opac import Opac
import scipy.constants as sc
from . import stellcorrection_utils as stc
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

class Data:
    
    def __init__(self, *args, **kwargs):
        with open(kwargs.pop('config')) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
            
            ## Define the N_PCA_dd
            inst_list = self.get_instrument_list
            N_PCA_dd = {}
            
            ####### Compute the N_PCA_dd first 
            for inst in inst_list:
                if self.config['data'][inst]['stellcorr_params']['N_PCA_all_order_common']:
                    N_PCA_dd[inst] = {}
                    for date in self.get_dates(inst):
                        N_PCA_dd[inst][date] = {}
                        for iord in range(self.config['data'][inst]['N_order_all']):# in self.get_use_order_inds(inst = inst, date = date):
                            N_PCA_dd[inst][date][iord] = self.config['data'][inst]['stellcorr_params']['N_PCA_all_order_common_value']
                else:
                    N_PCA_dd[inst] = self.config['data'][inst]['stellcorr_params']['N_PCA_dd']        
            self.N_PCA_dd = N_PCA_dd
            ####### Compute the pre_pca_mask_dd
            pre_pca_mask_dd = {}
            for inst in inst_list:
                pre_pca_mask_dd[inst] = {}
                for date in self.get_dates(inst):
                    pre_pca_mask_dd[inst][date] = {}
                    for iord in range(self.config['data'][inst]['N_order_all']):# self.get_use_order_inds(inst = inst, date = date):
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
        """Method to construct and return the dictionary containing the 
        spdatacubes from all instruments and all dates. 
        """
        spdatacubes_dict = {}
        for inst in self.get_instrument_list:
            spdatacubes_dict[inst] = {}
            for date in self.get_dates(inst):
                spdatacubes_dict[inst][date] = np.load(self.config['data'][inst]['dates'][date][0], allow_pickle = True).item()
        return spdatacubes_dict
    
    def get_use_order_inds(self, inst = None, date = None):

        spdatacubes_dd = self.get_spdatacubes_dict[inst][date]
        use_orders = []
        for i in range(spdatacubes_dd['spdatacube'].shape[0]):
            if i not in self.config['data'][inst]['skip_order_inds']:
                use_orders.append(i)
        return use_orders
    
    def create_pre_pca_mask_deep_tellurics_per_order(self, inst = None, date = None, order_ind = None):
        """If pre_pca_mask_flux_threshold in the config file is not 'none', 
        then create a mask for each detector and date based on the flux threshold. 
        Merge it with colmask before doing PCA because colmask is also a pre PCA mask.
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
        """
        
        # colmask_inds = self.config['data'][inst]['stellcorr_params']['colmask_inds']
        # N_PCA = self.config['data'][inst]['stellcorr_params']['N_PCA']
        
        ## Create the colmask from colmask info 
        colmask_info = self.config['data'][inst]['stellcorr_params']['badcolmask_inds'][date][order_ind]
        colmask_inds = []
        for i in range(len(colmask_info)):
            inds = np.arange(colmask_info[i][0], colmask_info[i][1])
            colmask_inds.extend(inds)
        colmask_inds = np.array(colmask_inds)
        
        ## Get the N_PCA for this date and order 
        N_PCA = self.N_PCA_dd[inst][date][order_ind]
        # self.config['data'][inst]['stellcorr_params']['N_PCA_dd'][date][order_ind]
                
        spdatacubes_dd = self.get_spdatacubes_dict[inst][date]
        
        # phases = spdatacubes_dd['phases']
        # berv = spdatacubes_dd['bary_RV']
        data_wavsoln = spdatacubes_dd['wavsoln'][order_ind,:]
        
        datacube = spdatacubes_dd['spdatacube'][order_ind, :, :]
        nspec, nwav = datacube.shape[0], datacube.shape[1]
        
        # Standardise the datacube first
        datacube_standard = np.empty((nspec, nwav))
        datacube_standard = stc.standardise_data(datacube)
        
        # Create a column mask for bad pixels, same for all dates and detectors
        colmask_ = np.zeros(datacube.shape[1], dtype=bool)
        if len(colmask_inds) > 0:
            colmask_[colmask_inds] = True

        ##############################################
        ## Combine the colmask_ with the pre_pca_mask 
        ##############################################
        colmask = np.logical_or(colmask_, self.pre_pca_mask_dd[inst][date][order_ind])
        
        # import pdb
        # pdb.set_trace()
        
        pca_eigenvectors = np.empty((nspec,N_PCA+1))
        datacube_fit = np.empty((nspec, nwav))
        datacube_detrended = np.empty((nspec, nwav))
        
        fStd = datacube_standard.copy()
        fStd[:, colmask] = 0
        pca_eigenvectors = stc.get_eigenvectors_via_PCA_Matteo(fStd[:, colmask == False], nc=N_PCA)

        datacube_fit = stc.linear_regression(X=pca_eigenvectors, Y=datacube)
        datacube_detrended = datacube/datacube_fit - 1.

        ## Apply the post PCA mask to zero out wavelength channels that were not corrected properly. 
        # Automatically detect these channels by checking if the variance of a channel is larger than sigma times 
        # median variance of all channels, where sigma is the threshold probability below which such a deviation
        # could have occured by random chance. You could use stdev here instead of variance, it will just be less stricter filter in that case. 
        datacube_detrended_post_pca_mask, post_pca_mask =  stc.mask_data_post_pca_per_order(datacube_detrended, maskval = 0., 
                                                                                            threshold = self.config['data'][inst]['stellcorr_params']['post_pca_mask_threshold']) ###  
        # Also zero out the badcolumns on top of this using badcolmask 
        datacube_detrended_post_pca_mask[:, colmask] = 0
        
        datacube_mean_sub = np.empty((nspec, nwav))
        zeroMask = np.tile( np.logical_or(post_pca_mask, colmask), (nspec,1) )  # datacube_detrended_post_pca_mask == 0  ## Get the mask for all the values where the datacube is set to zero.
        # This should have already been done at the stage of removing the PCA linear regression fit from the data.
        datacube_mean_sub = crocut.sub_mask(datacube_detrended_post_pca_mask, zeroMask)
        
        return colmask, post_pca_mask, data_wavsoln, datacube, datacube_fit, datacube_detrended_post_pca_mask, datacube_mean_sub, pca_eigenvectors
