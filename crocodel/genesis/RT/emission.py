import numpy as np
import scipy.constants as sc

class RT_em:

    def __init__(self):

        return

    def rte_solve(self, atm, kappa, sigma):

        dtmin = np.zeros((kappa.shape))
        chi = (kappa+sigma)/atm.rho
        dtmin[:,1:] = (0.5/(0.5*(atm.g[np.newaxis,:-1] + atm.g[np.newaxis,1:]) ))*(chi[:,:-1] + chi[:,1:])*(atm.P[np.newaxis,:-1] - atm.P[np.newaxis,1:])

        freq = sc.c/atm.lam

        tt, frqfrq = np.meshgrid(np.array(atm.T),np.array(freq),indexing = "ij")

        B = self.planck(frqfrq,tt)

        dtmin = dtmin.T

        cos = np.array([0.5-0.5/np.sqrt(3),0.5+0.5/np.sqrt(3)])
        weight = np.array([1.0,1.0])
        Imu = np.zeros((len(cos),len(freq)))
        Imu[:,:] = B[0,np.newaxis,:]
        for i in range (1,len(atm.P)):
            Bvals = 0.5*(B[i,:]+B[i-1,:])
            expdtmu = np.exp(-1.0*dtmin[i,np.newaxis,:]/cos[:,np.newaxis])
            Imu[0,:] = (Imu[0,:]-Bvals)*expdtmu[0]+Bvals
            Imu[1,:] = (Imu[1,:]-Bvals)*expdtmu[1]+Bvals
        FImu = 0.5*weight[:,np.newaxis]*cos[:,np.newaxis]*Imu
        return 2.0*np.pi*np.sum(FImu, axis=0)*(freq*freq/sc.c)*(atm.Rp*atm.Rp)
        
        
        # mu = 1.0

        # Imu = np.zeros((kappa.T.shape))
        # Imu[0,:] = B[0,:]
        # # import pdb
        # # pdb.set_trace()
        # expdt = np.exp(-1.0*dtmin/mu)
        
        # for i in range (1,len(atm.P)):
        #     Imu[i,:] = (Imu[i-1,:]-0.5*(B[i,:]+B[i-1,:]))*expdt[i,:]+0.5*(B[i,:]+B[i-1,:])
            
        # return 2.0*np.pi*Imu[-1,:]*(freq*freq/sc.c)*(atm.Rp*atm.Rp)

    def P_tau(self, atm, kappa, sigma, tau_val = 2./3.):
        dtmin = np.zeros((kappa.shape))
        chi = (kappa+sigma)/atm.rho
        dtmin[:,1:] = (0.5/(0.5*(atm.g[np.newaxis,:-1] + atm.g[np.newaxis,1:]) ))*(chi[:,:-1] + chi[:,1:])*(atm.P[np.newaxis,:-1] - atm.P[np.newaxis,1:])
        dtmin = dtmin.T
        
        ND, NF = dtmin.shape ## ND is the number of pressure layers, and NF is the number of frequencies (or wavelengths)
        # tau = np.zeros((NF, ND))
        
        # import pdb
        # pdb.set_trace()
        
        ### Loop across all pressure layers, for each layer find the optical depth contribution from that layer 
        # for i in range(1,ND):
        #     tau[:,ND-1-i] = np.sum(dtmin[ND-1-i:ND-1,:],axis=-1) ## for tau in shape NF x ND 

        ### Flip and do cumulative sum to find total optical depth at each layer, and flip again
        tau = np.cumsum(dtmin[::-1,:],axis=0)[::-1,:]
        # i_rp = np.argmin(np.abs(tau-tau_val),axis=0)        
        tau = tau[::-1,:]
        P = atm.P[::-1]
        log10_tau = np.log10(tau[:-1,:]+1.0e-250)
        log10_P = np.log10(P[:-1])
        
        P_vals = np.zeros(NF)
        for k in range(NF):
            P_vals[k] = np.interp(np.log10(tau_val), log10_tau[:,k], log10_P) ## need to figure this out 
        
        contribution_func = []
        for k in range(NF):
            d_num = np.diff(np.exp(-1. * np.power(10.0,log10_tau[:,k])))  
            d_denom = np.diff(log10_P)
            contribution_func.append(d_num/d_denom) ### See Knutson et al. 2008
        
        contribution_func = np.array(contribution_func) 
        
        return contribution_func, np.power(10.0,log10_tau), np.power(10.0,log10_P)*1.0e-5, np.power(10.0,P_vals)*1.0e-5 ### return the pressure in bars
        
    
    
    def planck(self,nu,T):

        return (2.0*sc.h*nu*nu*nu)/(sc.c*sc.c*(np.expm1((sc.h*nu)/(sc.k*T))))
