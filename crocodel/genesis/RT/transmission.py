import numpy as np

class RT_tr:

    def __init__(self):

        return
        
    def rte_solve(self, atm, kappa, sigma, cl_P = None, gamma = None, a = None):

        if cl_P is None:
            cl_P = np.max(atm.P)

        cl_P = np.power(10.0,cl_P)*1.0e5

        if gamma is None:
            gamma = -4.0

        if a is None:
            a = 1.0

        cloud_params = np.zeros(2)
        cloud_params[0] = np.power(10.0,a)
        cloud_params[1] = gamma*1.0

        sigma_clouds = self.cloud_opac(atm, sigma, cl_P, cloud_params)
        
        self.spec = self.transmission_rte(atm, kappa, sigma_clouds)        

        return self.spec

    def transmission_rte(self, atm, kappa, sigma):

        P = np.array(atm.P)
        r = np.array(atm.r)
        Rp = atm.Rp

        NF = len(kappa[:,0])
        ND = len(kappa[0,:])
                 
        above_Rp = np.zeros((NF))
        tau_tot = np.zeros((ND,NF))

        opac = np.array((kappa+sigma).T)
        opac = (opac[1:,:] + opac[:-1,:])

        for i in range(ND-1):

            s_tot = np.sqrt(r[i:]*r[i:]-r[i]*r[i])
            ds = s_tot[1:]-s_tot[:-1]

            tau_tot[i,:] = np.einsum("jk,j->k", opac[i:,:],ds)

        exptau = np.exp(-1.0*tau_tot)

        for i in range(ND-1):
            if (r[i] >= Rp):
                above_Rp[:] += 0.5*((r[i]*(1.0 - exptau[i, :]) + (r[i+1]*(1.0 - exptau[i+1, :])))*(r[i+1] - r[i]))
            if (r[i] < Rp):
                above_Rp[:] -= 0.5*((r[i]*(exptau[i, :]) + (r[i+1]*(exptau[i+1, :])))*(r[i+1] - r[i]))

        Rp_2 = (Rp*Rp + 2.0*above_Rp)

        return Rp_2

    def cloud_opac(self, atm, sigma, cl_P, cloud_params):
        
        sigma_0 = 5.31e-31
        lam_0 = 350.0e-9

        ND = len(sigma[0,:])

        dz = np.zeros(ND)
        for i in range(ND-1):
            dz[i] = np.sqrt(atm.r[i+1]*atm.r[i+1]-atm.r[i]*atm.r[i])

        dz[-1] =  np.sqrt(atm.r[-1]*atm.r[-1]-atm.r[-2]*atm.r[-2])

        sigma_clouds = atm.n[np.newaxis,:]*cloud_params[0]*sigma_0*((atm.lam[:,np.newaxis]/lam_0)**cloud_params[1])+sigma

        P_cl = np.full(len(sigma[:,0]),cl_P)

        P_cl_ratio = (atm.P[np.newaxis,:] - P_cl[:,np.newaxis])/(P_cl[:,np.newaxis]*dz[np.newaxis,:])
        cond = atm.P[np.newaxis,:] > P_cl[:,np.newaxis]
        
        sigma_clouds[cond] += P_cl_ratio[cond]
                    
        return sigma_clouds
