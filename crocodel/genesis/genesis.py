import numpy as np
from .opac.opac import Opac
from .RT.transmission import RT_tr
from .RT.emission import RT_em
import scipy.constants as sc
import yaml

class Atm:

    def __init__(self,rp_loc = "/rds/projects/p/piettaaa-exo-mapping/code/crocodel/crocodel/genesis/opac/", model_file = "model.yaml"):

        with open(rp_loc+model_file) as f:
            self.model = yaml.load(f,Loader=yaml.FullLoader)
                 

    def atm_set(self, P, T, n, r, rho, g,  Rp, lam, X):

        self.P = P*1.0
        self.T = T*1.0
        self.n = n*1.0
        self.r = r*1.0
        self.rho = rho*1.0
        self.g = g*1.0
        self.Rp = Rp*1.0
        self.lam = lam*1.0
        self.X = X

        self.mol_list = list(X.keys())

class Genesis:

    def __init__(self, Pmin, Pmax, ND, lam_min, lam_max, NF, spacing, method = "transmission", lam = None):

        self.opac = Opac() # This can be initialized once outside, and then taken as an input here. Then just keep running opac_without_opacity_check
        self.atm = Atm()
        if method=="transmission":
            self.RT = RT_tr()
        elif method=="emission":
            self.RT = RT_em()
        else:
            print("RT method ", method, " not recognised!!!")
            exit()

        self.method = method

        self.P = np.logspace(np.log10(Pmax)+5.0,np.log10(Pmin)+5.0, num = ND)
        if spacing is None and lam is not None:
            self.lam = lam
        else:
            if spacing=="R":
                
                R = NF*1.0
                num_vals = R*(np.log(lam_max*1.0e-6) - np.log(lam_min*1.0e-6))
                self.lam = np.logspace(np.log(lam_min*1.0e-6),np.log(lam_max*1.0e-6),num = int(num_vals)+1,endpoint=True,base=np.e) # constant resolution, so need evenly spaced in ln(lambda)
                
            elif spacing=="nu":
                
                nu = np.linspace(sc.c/(lam_min*1.0e-6),sc.c/(lam_max*1.0e-6),num = NF,endpoint=True)
                self.lam = sc.c/nu
                
            elif spacing=="lam":
                
                self.lam = np.linspace(lam_min*1.0e-6,lam_max*1.0e-6,num = NF,endpoint=False)

            else:
                print("spacing not recognised!")
                exit()


    def set_T(self,P1,T1,P2,T2,P0=None,T0=None, type = 'Linear'):#P in Pa
        assert P1 >= P2 # P1 should always be the higher pressure level.
        ### type can be 'Linear', 'Linear_force_inverted' or 'Linear_force_non_inverted'
        if type == 'Linear_force_inverted':
            assert T2>T1
        elif type == 'Linear_force_non_inverted':
            assert T2<T1
        
        log_p = np.log10(self.P)
        T = np.zeros(len(self.P))
        T[self.P<=P2] = T2
        T[self.P>=P1] = T1
        T[(self.P > P2) & (self.P < P1)] = T1*(log_p[(self.P > P2) & (self.P < P1)] - np.log10(P2))/(np.log10(P1) - np.log10(P2)) + T2*(np.log10(P1) - log_p[(self.P > P2) & (self.P < P1)])/(np.log10(P1) - np.log10(P2))
        
        if P0 is not None:
            T[(self.P > P1) & (self.P < P0)] = T0*(log_p[(self.P > P1) & (self.P < P0)] - np.log10(P1))/(np.log10(P0) - np.log10(P1)) + T1*(np.log10(P0) - log_p[(self.P > P1) & (self.P < P0)])/(np.log10(P0) - np.log10(P1))
            T[self.P >= P0] = T0

        self.T = T*1.0

    def profile(self, Rp, grav, Pref, mu = 2.35): #Rp in Rj, grav in log(g/cms^-1), Pref in log(P/bar), mu in atomic mass units

        self.Rp = Rp*7.1492e7 # convert to m
        grav = np.power(10.0,grav)/100.0 # convert to m/s^2
        Pref = np.power(10.0,Pref)*1.0e5 # conver to Pa
        
        self.n = self.P/(sc.k*self.T)
        
        if np.isscalar(mu):
            mu = np.full(len(self.P),mu*sc.u)
        else:
            mu = mu*sc.u
            
        self.rho = self.n*mu

        self.r = np.zeros(len(self.P))
        self.g = np.zeros(len(self.P))
        
        i_Rp = np.argmin(np.abs(self.P-Pref))

        self.r[i_Rp] = self.Rp*1.0
        self.g[i_Rp] = grav
        
        for i in range(i_Rp+1,len(self.P)):
            self.g[i] = self.g[i_Rp]*self.r[i_Rp]*self.r[i_Rp]/(self.r[i-1]*self.r[i-1])
            self.r[i] = self.r[i-1] - ((sc.k*(0.5*(self.T[i-1]+self.T[i])))/((0.5*(mu[i-1]+mu[i]))*self.g[i]))*np.log(self.P[i]/self.P[i-1])

        for i in range(i_Rp-1,-1,-1):
            self.g[i] = self.g[i_Rp]*self.r[i_Rp]*self.r[i_Rp]/(self.r[i+1]*self.r[i+1])
            self.r[i] = self.r[i+1] - ((sc.k*(0.5*(self.T[i+1]+self.T[i])))/((0.5*(mu[i+1]+mu[i]))*self.g[i]))*np.log(self.P[i]/self.P[i+1])
        
    def genesis(self, X, cl_P = None, include_cia = True):
        
        self.atm.atm_set(self.P, self.T, self.n, self.r, self.rho, self.g, self.Rp, self.lam, X)
        self.opac.opacity(self.atm, include_cia=include_cia)
        
        if self.method=="transmission":
            spec = self.RT.rte_solve(self.atm, self.opac.kappa, self.opac.sigma, cl_P = cl_P)
        else:
            assert(cl_P is None)
            spec = self.RT.rte_solve(self.atm, self.opac.kappa, self.opac.sigma)

        return spec

    def contribution_function(self, X, cl_P = None, tau_val = 2./3., include_cia = True):
        self.atm.atm_set(self.P, self.T, self.n, self.r, self.rho, self.g, self.Rp, self.lam, X)
        self.opac.opacity(self.atm, include_cia=include_cia)
        
        contribution_func, tau, P_array, P_tau = self.RT.P_tau(self.atm, self.opac.kappa, self.opac.sigma, tau_val = tau_val)
        
        return contribution_func, tau, P_array, P_tau