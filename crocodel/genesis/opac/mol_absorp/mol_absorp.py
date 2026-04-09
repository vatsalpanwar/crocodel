import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import scipy.constants as sc

class Mol_absorp:

    def __init__(self):

        self.model = {}
        self.lam = np.array([0.0])
        self.P_grid = np.array([0.0])
        self.mol_list = []

        return
        
    def opac(self, atm):

        self.opac_check(atm)

        self.mol_opac(atm)
            
        self.sigma = np.zeros((len(atm.lam), len(atm.P)))
        self.dsigmadT = np.zeros((len(atm.lam), len(atm.P)))
                
        lam_2 = atm.lam*atm.lam
        lam_4 = lam_2*lam_2
        lam_6 = lam_4*lam_2
        lam_8 = lam_4*lam_4
        self.sigma = (atm.n[np.newaxis,:]*((8.14e-53/(lam_4[:,np.newaxis])) + (1.28e-66/(lam_6[:,np.newaxis])) + (1.61e-80/(lam_8[:,np.newaxis]))))/10000.0
        
        self.dsigmadT = -1.0*((atm.n[np.newaxis,:]/atm.T[np.newaxis,:])*((8.14e-53/(lam_4[:,np.newaxis])) + (1.28e-66/(lam_6[:,np.newaxis])) + (1.61e-80/(lam_8[:,np.newaxis]))))/10000.0

        return self.kappa, self.sigma
    
    # Added by Vatsal : 
    # def opac_without_opac_check(self, atm):

    #     # self.opac_check(atm)

    #     self.mol_opac(atm)
            
    #     self.sigma = np.zeros((len(atm.lam), len(atm.P)))
    #     self.dsigmadT = np.zeros((len(atm.lam), len(atm.P)))
                
    #     lam_2 = atm.lam*atm.lam
    #     lam_4 = lam_2*lam_2
    #     lam_6 = lam_4*lam_2
    #     lam_8 = lam_4*lam_4
    #     self.sigma = (atm.n[np.newaxis,:]*((8.14e-53/(lam_4[:,np.newaxis])) + (1.28e-66/(lam_6[:,np.newaxis])) + (1.61e-80/(lam_8[:,np.newaxis]))))/10000.0
        
    #     self.dsigmadT = -1.0*((atm.n[np.newaxis,:]/atm.T[np.newaxis,:])*((8.14e-53/(lam_4[:,np.newaxis])) + (1.28e-66/(lam_6[:,np.newaxis])) + (1.61e-80/(lam_8[:,np.newaxis]))))/10000.0

    #     return self.kappa, self.sigma


    def mol_opac(self, atm):

        ND = len(atm.P)
        NF = len(atm.lam)

        self.kappa = np.zeros((NF,ND))
        
        for key in atm.X:
            if key not in atm.model["opac"]["cia_only"]:
                if not np.all(atm.X[key]==0):
                    lamPT = self.lamPT_arr(key, atm.lam, atm.P, atm.T)
                    # import pdb; pdb.set_trace()
                    self.kappa += atm.X[key][np.newaxis,:]*atm.n[np.newaxis,:]*np.power(10.0,self.opac_dict[key](lamPT))
        
    def opac_check(self, atm):
        
        if (self.model != atm.model["opac"]) or (np.amin(self.lam) != np.amin(atm.lam)) or (np.amax(self.lam) != np.amax(atm.lam)) :
            self.model = atm.model["opac"]
            self.lam = atm.lam

            self.opac_dict_setup(atm)
                
        for item in atm.mol_list:
            if item not in self.opac_dict:
                if item not in atm.model["opac"]["cia_only"]:

                    self.opac_dict[item], self.opac_lam[item], self.opac_P[item], self.opac_T[item] = self.add_mol(atm, item)


    def lamPT_arr(self, mol, lam, P, T, P_low = None, P_high = None, T_low = None, T_high = None):# bar and K
        
        if P_low is None:
            P_low = np.amin(np.power(10.0,self.opac_P[mol]))

        if P_high is None:
            P_high = np.amax(np.power(10.0,self.opac_P[mol]))

        if T_low is None:
            T_low = np.amin(self.opac_T[mol])

        if T_high is None:
            T_high = np.amax(self.opac_T[mol])

        lam_low = np.amin(self.opac_lam[mol])
        lam_high = np.amax(self.opac_lam[mol])
     
        lamPT = np.zeros((len(lam),len(P),3))
        
        lam_ends = np.copy(lam)
        lam_ends[lam>lam_high] = lam_high
        lam_ends[lam<lam_low] = lam_low
        lamPT[:,:,0] = lam_ends[:,np.newaxis]
        
        P_ends = np.copy(P)
        P_ends[P>P_high] = P_high
        P_ends[P<P_low] = P_low
        lamPT[:,:,1] = np.log10(P_ends[np.newaxis,:])

        T_ends = np.copy(T)
        T_ends[T>T_high] = T_high
        T_ends[T<T_low] = T_low
        lamPT[:,:,2] = T_ends[np.newaxis,:]

        return lamPT   
        
    def opac_dict_setup(self,atm):

        self.opac_dict = {}
        self.opac_lam = {}
        self.opac_P = {}
        self.opac_T = {}

        for item in atm.mol_list:
            if item not in atm.model["opac"]["cia_only"]:

                self.opac_dict[item], self.opac_lam[item], self.opac_P[item], self.opac_T[item] = self.add_mol(atm, item)

        return

    def add_mol(self, atm, mol, P_low = None, P_high = None, T_low = None, T_high = None, Rmax = False):
        
        if mol=="h_minus":
            P_array = np.linspace(0,7,num = 8)
            T_array = np.array([300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1200.0,1400.0,1600.0,1800.0,2000.0,2500.0,3000.0,3500.0])
            nu_grid =  np.linspace(sc.c/50.0e-6,sc.c/0.4e-6,num = 24801,endpoint = True)
            lam_array = sc.c/nu_grid[::-1]
            lam_bools = (lam_array>=np.amin(atm.lam)) & (lam_array<=np.amax(atm.lam))
            lam_array = lam_array[lam_bools]
            array = self.h_min_opac(lam_array, np.power(10.0,P_array), T_array)
            opac_func = RegularGridInterpolator((lam_array,P_array,T_array),array,bounds_error=False,fill_value=None)
            return opac_func, lam_array, P_array, T_array
        
        else:
            # if mol in atm.model["opac"]:
            #     f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ atm.model["opac"][mol]+".hdf5", "r")
            # else:
            #     f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ mol+".hdf5", "r")

            # opac = f[mol]

            # lam = opac["lam"][...]
            # f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ mol+".hdf5", "r")
            # lam = f['wave'][...]
            
            
            # lam_bools = ((lam>=np.around(np.amin(atm.lam),15)) & (lam<=np.around(np.amax(atm.lam), 15))).astype(bool)
            # lam_bools = np.array(lam_bools.nonzero())[0]

            # array = opac["cross_sec"][lam_bools,:,:]
            # lam = lam[lam_bools]

            # if (len(lam)<=1):
            #     print("Wavelength slice has <= 1 points for", mol,"in Regular Grid Interpolator!")
            #     exit()

            # P = opac["P"][...]
            # T = opac["T"][...]
            
            ## Updating above from Feb 2025 for the new opacity format : 
            f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ mol+".hdf5", "r")
            lam = f['wave'][...]

            lam_bools = ((lam>=np.around(np.amin(atm.lam),15)) & (lam<=np.around(np.amax(atm.lam), 15))).astype(bool)
            lam_bools = np.array(lam_bools.nonzero())[0]

            array = f["cross_sec"][lam_bools,:,:]
            
            lam = lam[lam_bools]

            if (len(lam)<=1):
                print("Wavelength slice has <= 1 points for", mol,"in Regular Grid Interpolator!")
                exit()

            P = f["P"][...]
            T = f["T"][...]
            ### Uncomment above for the new opacity format :
            
            ### Uncomment below for the old opacity format :
            # if mol in atm.model["opac"]:
            #     f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ atm.model["opac"][mol]+".hdf5", "r")
            # else:
            #     f = h5py.File(atm.model["opac"]["file_loc"] + atm.model["opac"]["opac"] + "/"+ mol+".hdf5", "r")

            # opac = f[mol]

            # lam = opac["lam"][...]
            
            # lam_bools = ((lam>=np.around(np.amin(atm.lam),15)) & (lam<=np.around(np.amax(atm.lam), 15))).astype(bool)
            # lam_bools = np.array(lam_bools.nonzero())[0]

            # array = opac["cross_sec"][lam_bools,:,:]
            # lam = lam[lam_bools]

            # if (len(lam)<=1):
            #     print("Wavelength slice has <= 1 points for", mol,"in Regular Grid Interpolator!")
            #     exit()

            # P = opac["P"][...]
            # T = opac["T"][...]

            ####### Uncomment above for the old opacity format
            
            if P_low is not None:
                assert(P_high is not None)

                P_bools = ((P>=P_low) & (P<=P_high)).astype(bool)
                P_bools = np.array(P_bools.nonzero())[0]

                if len(P[P_bools])>2:
                    P = P[P_bools]
                    array = array[:,P_bools,:]

            if T_low is not None:
                assert(T_high is not None)

                T_bools = ((T>=T_low) & (T<=T_high)).astype(bool)
                T_bools = np.array(T_bools.nonzero())[0]

                if len(T[T_bools])>2:
                    T = T[T_bools]
                    array = array[:,:,T_bools]

            opac_func = RegularGridInterpolator((lam,P,T),array,bounds_error=True,fill_value=None)

        return opac_func, lam, P, T

    def h_min_opac(self, lam, P, T):
        
        f = (1.6419-lam*1.0e6)/(lam*1.0e6*1.6419)
        f[f<0] = 0.0
        func = 152.519 + 49.534*np.power(f,0.5) - 118.858*np.power(f,1) + 92.536*np.power(f,1.5) - 34.194*np.power(f,2) + 4.982*np.power(f,2.5)
        cs = 1.0e-18 * lam*lam*lam * 1.0e18 * np.power(f,1.5) * func
        #cs *= 1.0e-4 #cm^2 to m^2
        kbf = np.zeros((len(lam), len(T)))
        alpha = sc.h*sc.c/sc.k
        #kbf[:,:] = 0.75 * np.power(T,-2.5)[np.newaxis,:] * np.exp(alpha/(1.6419*1.0e-6*T))[np.newaxis,:] * (1.0 - np.exp(-alpha/(lam[:,np.newaxis]*T[np.newaxis,:])) ) * cs[:,np.newaxis]
        #kbf[:,:] = cs[:,np.newaxis]
        #kbf[:,:] = cs[:,np.newaxis] * (1.0 - np.exp(-alpha/(lam[:,np.newaxis]*T[np.newaxis,:])) )
        kbf[:,:] =  cs[:,np.newaxis]*(1.0 - np.exp(-alpha/(lam[:,np.newaxis]*T[np.newaxis,:])) )
        A = np.zeros(6)
        B = np.zeros(6)
        C = np.zeros(6)
        D = np.zeros(6)
        E = np.zeros(6)
        F = np.zeros(6)
        A[1] = 2483.346
        A[2] = -3449.889
        A[3] = 2200.04
        A[4] = -696.271
        A[5] = 88.283
        B[1] = 285.827
        B[2] = -1158.382
        B[3] = 2427.719
        B[4] = -1841.4
        B[5] = 444.517
        C[1] = -2054.291
        C[2] = 8746.523
        C[3] = -13651.105
        C[4] = 8624.97
        C[5] = -1863.864
        D[1] = 2827.776
        D[2] = -11485.632
        D[3] = 16755.524
        D[4] = -10051.53
        D[5] = 2095.288
        E[1] = -1341.537
        E[2] = 5303.609
        E[3] = -7510.494
        E[4] = 4400.067
        E[5] = -901.788
        F[1] = 208.952
        F[2] = -812.939
        F[3] = 1132.738
        F[4] = -655.02
        F[5] = 132.985
        lum = lam*1.0e6
        kff = np.zeros((len(lam),len(T)))
        for i in range(1,6):
            kff[:,:] += 1.0e-29 * np.power(5040.0/T, float((i+2.0)/2.0))[np.newaxis,:] * (lum*lum*A[i] + B[i] + C[i]/lum + D[i]/(lum*lum) + E[i]/(lum*lum*lum) + F[i]/(lum*lum*lum*lum) )[:,np.newaxis]
        kff /=  0.75 * np.power(T,-2.5)[np.newaxis,:] * np.exp(alpha/(1.6419*1.0e-6*T))[np.newaxis,:]
        #kff *= (1.0 - np.exp(-alpha/(lam[:,np.newaxis]*T[np.newaxis,:])) )
        logsigma = np.zeros((len(lam),len(P),len(T)))
        ktot = kff + kbf
        logsigma[:,:,:] = np.log10(1.0e-4*(ktot[:,np.newaxis,:]))
        '''
        X_h = np.zeros((len(P), len(T)))
        Ad = np.power(10.0,-0.14e4/T - 7.7)[np.newaxis,:]*np.power(P*1.0e-5,0.6)[:,np.newaxis]
        A = 1.0/np.sqrt(Ad) + 1.0/np.sqrt(np.power(10.0,-8.3))
        A = 1.0/(A*A)
        '''
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(logsigma.shape[2]):
        #     plt.plot(1e6*lam, logsigma[:,0,i] , alpha = 0.7, linewidth = 0.7)
        # plt.xlabel('Wavelength [micron]')
        # plt.ylabel("log sigma")
        # plt.savefig('/home/astro/phsprd/code/genesis/outputs/'+  'test_cross_section_h_minus.png', format = 'png', dpi = 300, bbox_inches = 'tight')
        
        
        return logsigma
