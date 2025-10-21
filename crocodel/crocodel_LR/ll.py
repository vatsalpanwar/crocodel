import numpy as np
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
import scipy.constants as sc
import pandas as pd

class Instrument_LR:

    def __init__(self, inst, data_file_path):

        self.instrument = inst
        self.data_file_path = data_file_path
        data = np.loadtxt(self.data_file_path,ndmin=2)
        self.lam_data = data[:,0]*1.0e3 ## convert um to nm
        self.num_data_pts = len(self.lam_data)
        self.lam_pm = np.array( [np.diff(data[:,0])[0]*1e3] + list(np.diff(data[:,0])*1.0e3) )
        self.y_data = data[:,-2]
        self.y_err = data[:,-1]
        
        ###### Load the following from the data_dict 
        ###### Wavelength must always be in nm
        
        
        # data = np.loadtxt("./data/"+self.instrument+".txt",ndmin=2)
        # self.lam_data = data[:,0]*1.0e-6
        # self.num_data_pts = len(self.lam_data)
        # self.lam_pm = data[:,1]*1.0e-6
        # self.y_data = data[:,-2]
        # self.y_err = data[:,-1]

        #fhp = pd.Series(self.y_data).rolling(window=20,min_periods=1,center=True).median()
        #self.y_data = self.y_data - fhp

        # file_path = os.path.abspath(os.path.dirname(__file__))
        # sens_file = np.loadtxt(os.path.join(file_path, "instrument_sens/"+self.instrument+".txt"))
        
        # lam_sens = sens_file[:,0]*1.0e-6
        # vals_sens = sens_file[:,1]

        # self.sens_func = InterpolatedUnivariateSpline(lam_sens,vals_sens,ext=1)
            
        return

    def loglike(self, lam, spec, beta_err = 1.0):

        self.y_model = self.bin_spectrum(lam, spec)

        #fhp = pd.Series(self.y_model).rolling(window=20,min_periods=1,center=True).median()
        #self.y_model = self.y_model - fhp

        #add_err = jitter*(np.log10(100.0*np.max(self.y_err**2))-np.log10(0.01*np.min(self.y_err**2))) + np.log10(0.01*np.min(self.y_err**2))
        #add_err = (jitter > 0)*np.power(10.0,add_err)
        y_err = beta_err*self.y_err
        
        self.logl_loo = -0.5*(((self.y_model-self.y_data)/y_err)**2)
        self.logl_loo += -0.5*np.log(2.0*np.pi*y_err*y_err)

        self.loglikelihood = -0.5*np.sum(((self.y_model-self.y_data)/y_err)**2)
        self.loglikelihood += -0.5*np.sum(np.log(2.0*np.pi*y_err*y_err))

        return self.loglikelihood


    def bin_spectrum(self,lam, F):
        '''
        Given flux, this is binned and a set of data points is created
        '''

        lam_slice = lam
        F_slice = F

        min_indx = np.searchsorted(lam_slice,self.lam_data-self.lam_pm,side="left")
        max_indx = np.searchsorted(lam_slice,self.lam_data+self.lam_pm,side="left")
        # import pdb; pdb.set_trace()
        if min_indx.size==1:

            min_indx = np.ones((1),dtype=np.int)*min_indx
            max_indx = np.ones((1),dtype=np.int)*max_indx
        
        check_pts = (max_indx-min_indx<=2)
        if np.any(check_pts):
            print("Wavelength grid is not fine enough!!! Some lambda bins are empty in ",self.instrument)
            exit()
        

        F_conv = np.copy(F_slice)

        # sensitivity = self.sens_func(lam_slice)
        sensitivity = np.ones(len(lam_slice))
        F_sens = F_conv*sensitivity

        y_model = np.zeros(self.num_data_pts)
        integral_norm = np.zeros(self.num_data_pts)

        for l in range(self.num_data_pts):

            integral_norm[l] = np.trapz(sensitivity[min_indx[l]:max_indx[l]],  lam_slice[min_indx[l]:max_indx[l]])+1.0e-250
            y_model[l] = np.trapz(F_sens[min_indx[l]:max_indx[l]],  lam_slice[min_indx[l]:max_indx[l]])/integral_norm[l]

        return y_model
