import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

class Cia:

    def __init__(self):
        
        self.model = {}
        self.lam = np.array([0.0])
        self.cia_list = []

        return
        
    def opac(self, atm):

        self.cia_check(atm) 

        self.cia_opac(atm)

        return self.kappa
    
    # Added by Vatsal 
    def opac_without_opac_check(self, atm):

        # self.cia_check(atm) ## Run this only once when initializing the class 

        self.cia_opac(atm)

        return self.kappa

    def cia_opac(self, atm):

        self.kappa = np.zeros((len(atm.lam),len(atm.T)))
        self.dkappadT = np.zeros((len(atm.lam),len(atm.T)))

        for item in atm.model["opac"]["cia_files"]:
            if item[:item.find("_")] in atm.mol_list and item[item.find("_")+1:] in atm.mol_list:
                if not np.all(atm.X[item[:item.find("_")]]==0) and not np.all(atm.X[item[item.find("_")+1:]]==0):
                    lamT = self.lamT_arr(item, atm.lam, atm.T)
                    opac = np.power(10.0,self.cia_dict[item](lamT))
                    self.kappa += atm.X[item[:item.find("_")]][np.newaxis,:]*atm.X[item[item.find("_")+1:]][np.newaxis,:]*atm.n[np.newaxis,:]*atm.n[np.newaxis,:]*opac
                    self.dkappadT += atm.X[item[:item.find("_")]][np.newaxis,:]*atm.X[item[item.find("_")+1:]][np.newaxis,:]*atm.n[np.newaxis,:]*(atm.n[np.newaxis,:]/atm.T)*(self.cia_dict_dT[item](lamT)-2.0*opac)     


    def cia_check(self, atm):
        
        if (self.model != atm.model["opac"]) or (np.amin(self.lam) != np.amin(atm.lam)) or (np.amax(self.lam) != np.amax(atm.lam)) :
            self.model = atm.model["opac"]
            self.lam = atm.lam
            
            self.cia_dict_setup(atm)


    def lamT_arr(self, mol, lam, T, T_low = None, T_high = None):# bar and K

        if T_low is None:
            T_low = np.amin(self.cia_T[mol])

        if T_high is None:
            T_high = np.amax(self.cia_T[mol])
     
        lam_low = np.amin(self.cia_lam[mol])
        lam_high = np.amax(self.cia_lam[mol])
        
        lamT = np.zeros((len(lam),len(T),2))

        lam_ends = np.copy(lam)
        lam_ends[lam>lam_high] = lam_high
        lam_ends[lam<lam_low] = lam_low
        lamT[:,:,0] = lam_ends[:,np.newaxis]

        T_ends = np.copy(T)
        T_ends[T>T_high] = T_high
        T_ends[T<T_low] = T_low
        lamT[:,:,1] = T_ends[np.newaxis,:]

        return lamT   
        
    def cia_dict_setup(self,atm):

        self.cia_dict = {}
        self.cia_dict_dT = {}
        self.cia_T = {}
        self.cia_lam = {}

        for item in atm.model["opac"]["cia_files"]:
            #if item[:item.find("_")] in atm.mol_list and item[item.find("_")+1:] in atm.mol_list:
            self.cia_dict[item], self.cia_lam[item], self.cia_T[item], self.cia_dict_dT[item] = self.cia_add(atm, item)
        return
        
    def cia_add(self, atm, mol):

        f = h5py.File(atm.model["opac"]["file_loc"]+"cia/" + mol +".hdf5", "r")
        opac = f[mol]

        opac_func = RegularGridInterpolator((opac["lam"],opac["T"]),opac["cross_sec"][...],bounds_error=False,fill_value=None)  
        
        dsigdT = np.gradient(opac["cross_sec"],25.0,axis=1)
        dsigdT *= np.power(10.0,opac["cross_sec"])/np.log10(2.718281828)
        
        opac_func_dT = RegularGridInterpolator((opac["lam"],opac["T"]),dsigdT,bounds_error=False,fill_value=None)

        return opac_func, opac["lam"][...], opac["T"][...], opac_func_dT
