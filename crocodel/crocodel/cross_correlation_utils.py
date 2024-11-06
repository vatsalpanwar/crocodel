import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import scipy
from scipy import interpolate
from tqdm import tqdm
from scipy import interpolate
import scipy.stats

from . import stellcorrection_utils as stc

def fast_mean(x):
    """
    This function computes the mean of an array faster than np.mean
    :param x: Input array for which the mean needs to be computed. 
    :type x: array_like
    :return: Mean of x.
    :rtype: float64
    
    """
    return (1 / len(x)) * np.dot(x, np.ones(len(x)))


def fast_var(x):
    """
    This function computes the variance of an array faster than np.var
    :param x: Input array for which the mean needs to be computed. 
    :type x: array_like
    :return: Variance of x.
    :rtype: float64
    """
    # return (1 / len(x)) * fast_mean( (x - fast_mean(x)) ** 2.)
    return np.dot(x,x)/len(x) # This assumes that the mean is zero


def fast_cross_corr(data=None, model=None):
    """
    This function computes a fast cross correlation for a given 1D data and 1D model arrays.
    :param data: 1D array of data.
    "type data: array_like
    
    :param model: 1D array of model.
    :type model: array_like
    

    :return: Cross correlation (both normalized and not-normalized) between the data and the model, along with
    the log-likelihood.
    :rtype: float64
    """
    sf2 = fast_var(data)
    sg2 = fast_var(model)
    N_data = len(data)

    R = (1. / N_data) * np.dot(data, model)  ## R in Brogi and Line

    C = R / np.sqrt(sf2 * sg2)  ## C in Brogi and Line

    logL = (-N_data/2) * np.log(sf2 + sg2 - 2.*R)
    return R, C, logL

def get_Vp(Kp = None, phases = None):
    """
    Compute an array of Keplerian velocities given the Kp and phases. For now by default circular but include
    computation for eccentric orbits.

    :param Kp: RV amplitude.
    :type Kp: float64
    
    :param phases: Array of orbital phase values. 
    :type phases: array_like
   
    :return: 1D array of planetary Keplerian velocities.
    :rtype: array_like
    
    """

    Vp = Kp * np.sin( 2. * np.pi * phases )
    return Vp

def doppler_shift_wavsoln(wavsoln=None, velocity=None):
    """
    This function applies a Doppler shift to a 1D array of wavelengths.
    wav_obs = wav_orig (1. + velocity/c) where if velocity is positive it corresponds to a redshift
    (i.e. source moving away from you, so wavelength solution gets shifted towards positive direction) and vice versa
    for a negative velocity corresponding to blueshift i.e. source moving towards you.

    :param wavsoln: 1D array if wavelengths, ideally in nanometers.
    :type wavsoln: array_like
    

    :param velocity: Float value of the velocity of the source, in km/s. Note that the astropy value of speed of light (c) is
    in m/s.
    :type velocity: float64

    :return: Doppler shifted wavelength solution.
    :rtype: array_like
    """
    wavsoln_doppler = wavsoln * (1. + (1000. * velocity) / 299792458.0)
    return wavsoln_doppler


def resample_spectrum(wavsoln_new=None, wavsoln_orig=None, spec_flux_orig=None):
    """
    This function computes an interpolated 1D spectrum for a new wavelength solution, given the original 1D spectrum and
    wavelength solution by performing a BSpline interpolation.
    :param wavsoln_new: array_like
    New wavelength solution.

    :param wavsoln_orig: Original wavelength solution.
    :type wavsoln_orig: array_like
    
    :param spec_flux_orig: Original 1D spectrum flux values.
    :type spec_flux: array_like

    :return: Interpolated 1D spectrum for the new wavelength solution.
    """
    # First create the BSpline interpolation for the wavsoln_orig and spec_flux_orig.
    # spl = splrep(wavsoln_orig, spec_flux_orig)
    spl = interpolate.make_interp_spline(wavsoln_orig, spec_flux_orig)
    spec_flux_new = spl(wavsoln_new)

    return spec_flux_new


def doppler_shift_1d_spectrum(wavsoln=None, spec_flux=None, velocity=None):
    """
    This function computes the Doppler shifted 1D spectrum for a given velocity shift (in km/s).

    :param wavsoln: Original 1D wavelength solution, in nm.
    :type wavsoln: array_like
    

    :param spec_flux: Original 1D spectrum.
    :type: array_like
    

    :param velocity: Value of the velocity of the source, in km/s. Note that the astropy constant value of speed of light (c) is
    in m/s.
    :type velocity: float
    

    :return: The new Doppler shifted wavelength solution and the Doppler shifted spectrum. 
    :rtype: array_like
    
    """
    # First Doppler shift the wavelength solution.
    wavsoln_new = doppler_shift_wavsoln(wavsoln=wavsoln, velocity=velocity)

    # Now compute the Doppler shifted spectrum as the interpolated spectrum with respect to
    # this new Doppler shifted wavelength solution.
    spec_flux_new = resample_spectrum(wavsoln_new=wavsoln_new, wavsoln_orig=wavsoln, spec_flux_orig=spec_flux)

    return wavsoln_new, spec_flux_new


def sub_mask(mat,zM):
    """
    Subtracting the mean from each spectrum *considering non-zero elements only*
    and zero-ing the masked columns so that they do not contribute to the logL
    
    :param mat: Datacube matrix (shape (time, wavelength)).
    :type mat: array_like

    :param zM: 2D mask marking points in the datacube that are zero.
    :type zM: array_like, bool

    :return: Mean subtracted datacube with the zero values excluded from mean calculation.
    :rtype: array_like
    """

    mat[zM] = 0.0
    clipped = mat.copy()
    nf, nx = mat.shape
    idVec = np.ones(nx)
    nn = np.sum(zM==False,axis=1)

    mean = (mat @ idVec)
    mean /= nn
    for j in range(nf):
        clipped[j,] -= mean[j]
    clipped[zM] = 0.0
    return clipped

def sub_mask_1D(arr,zM):
    """
    1D version of the sub_mask function.
    
    :param arr: 1D data array (shape (wavelength)).
    :type arr: array_like

    :param zM: 1D mask marking points in the datacube that are zero.
    :type zM: array_like, bool

    :return: Mean subtracted data array with the zero values excluded from mean calculation.
    :rtype: array_like
    """

    arr[zM] = 0.0
    clipped = arr.copy()
    nx = len(arr)
    idVec = np.ones(nx)
    nn = np.sum(zM==False)

    mean = (arr @ idVec)
    mean /= nn
    # for j in range(nf):
    clipped -= mean
    clipped[zM] = 0.0
    return clipped

def get_cubes_all_exp_fast(datacube=None, data_wavsoln=None, model_spec_flux=None, model_wavsoln=None,
                   velocity_shifts=None):
    """
    This function takes the detrended datacube (shape [time, wavelength], PCA trend removed) and computes the matrix of
    cc values and logL for each given list of velocity shifts. This is the fast method and it needs shifting by Vp for
    a range of Kps at the next stage to make the Kp-Vsys map.

    :param datacube: Numpy array of timeseries high-resolution spectra; dimensions should be [time,wavelength].
    :type datacube: array_like
    

    :param data_wavsoln: Wavelength solution of the data, 1D array.
    :type data_wavsoln: array_like
    

    :param model_spec_flux: 1D array of the model 1D spectrum (from a single exposure), should be the same length as data_spec_flux.
    :type model_spec: array_like
    

    :param model_wavsoln: Wavelength solution of the model, 1D array.
    :type model_wavsoln: array_like
    
    :param velocity_shifts: 1D array of velocity shifts (in km/s) for which you want to calculate the cross-correlation coefficients.
    :type velocity_shifts: array_like
    
    :param data_mask: 1D array of bool values to indicate the wavelength channels with bad pixels that should be excluded from the cross-correlation.
    :type data_mask: array_like, bool
    
    :return: 2D array of CC values with shape [time, velocity_shifts].
    :rtype: array_like
    
    """

    # Subtract the mean from the data first
    zeroMask = datacube == 0 ## Get the mask for all the values where the datacube is set to zero.
    # This should have already been done at the stage of removing the PCA linear regression fit from the data.
    datacube_mean_sub = sub_mask(datacube, zeroMask)

    # Empty matrix to collect the CC values for each exposure (each row) for different velocities (each column).
    cc_matrix = []
    logL_matrix = []

    # model_spl = splrep(model_wavsoln, model_spec_flux)
    model_spl = interpolate.make_interp_spline(model_wavsoln, model_spec_flux)

    for tt in tqdm(range(datacube_mean_sub.shape[0])):
        # Empty row to collect the CC values for given exposure at different velocities
        cc_matrix_row = []
        logL_matrix_row = []

        for vv, vel in enumerate(velocity_shifts):
            # First Doppler shift the data wavelength solution to -vel
            data_wavsoln_shift = doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * vel)

            # Evaluate the model to the data_wavsoln_shifted by -vel,
            # Effectively Doppler shifting the model by +vel
            # model_spec_flux_shift = splev(data_wavsoln_shift, model_spl)
            model_spec_flux_shift = model_spl(data_wavsoln_shift)

            # Subtract the mean from the model
            model_spec_flux_shift = model_spec_flux_shift - fast_mean(model_spec_flux_shift)

            # Compute the cross correlation value between the shifted model and the data
            cc_R, cc_C, logL = fast_cross_corr(data=datacube_mean_sub[tt,:], model=model_spec_flux_shift)

            # Append to the row for this exposure
            cc_matrix_row.append(cc_C)
            logL_matrix_row.append(logL)

        # Append the row for a given exposure to the CC matrix
        cc_matrix.append(cc_matrix_row)
        logL_matrix.append(logL_matrix_row)

    return np.array(cc_matrix), np.array(logL_matrix)

def get_KpVsys_fast(cc_matrix = None, logL_matrix = None, velocity_shifts = None,  phases = None, Kp_range = None, berv = None,
                    vwin = None):
    """
    Convert CC matrix (phases x velocity shifts) obtained after summing across all detectors to Kp-Vsys
    map via fast method. For each Kp value, and for each exposure, this method shifts (through linear interpolation)
    each CC row of the CC matrix by Vp + berv (where berv is the velocity of the barycenter in observer's rest frame;
    this should be Vp - berv if berv is the velocity of the observer in the barycenter rest frame.
    Astropy, in its radial_velocity_correction actually calculates the velocity of observer in barycentric rest frame,
    so in our case here it needs to be subtracted (need to be verified with more cases but it checks out for
    CRIRES tau Boo observations and the RVEL values from Matteo.)

    :param CC_matrix: Matrix of CC values with dimensions as (phases, velocity_shifts).
    :type CC_matrix: array_like
    

    :param logL_matrix:  Matrix of logL values with dimensions as (phases, velocity_shifts).
    :type logL_matrix: array_like
   

    :param velocity_shifts:  1D array of velocity shifts for which the CC_matrix was originally calculated for, by default in km/s.
    :type velocity_shifts: array_like
   

    :param phases: 1D array of the orbital phases for which the observations have been taken.
    :type phases: array_like
    

    :param Kp_range: 1D array of the Range of Kp values you want to construct the Kp-Vsys diagram for.
    :type Kp_range: array_like
    

    :param berv: 1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).
    :type berv: array_like
    

    :param vwin : The index bounds of the velocity_shifts to which you want to restrict the computation of the Kp-Vsys map to.
    :type: list or tuple
    
    :return: Return the Kp-Vsys CC and logL 2D maps.

    """
    cc_Kp_Vsys = []  # nKp x nVsys
    logL_Kp_Vsys = [] # nKp x nVsys

    nspec = len(phases)

    for Kp in Kp_range:

        cc_matrix_shifted = []
        logL_matrix_shifted = []

        for ii in range(nspec):
            phase = phases[ii]

            Vp = Kp * np.sin(2. * np.pi * phase)

            velocity_shifts_shifted = velocity_shifts + Vp + berv[ii]

            func_cc = interpolate.interp1d(velocity_shifts, cc_matrix[ii, :])
            func_logL = interpolate.interp1d(velocity_shifts, logL_matrix[ii, :])

            cc_matrix_shifted_ = func_cc(velocity_shifts_shifted[vwin[0]:vwin[1]])
            logL_matrix_shifted_ = func_logL(velocity_shifts_shifted[vwin[0]:vwin[1]])

            cc_matrix_shifted.append(cc_matrix_shifted_)
            logL_matrix_shifted.append(logL_matrix_shifted_)

        # Convert to array, which is of dimension (nspec, nVsys)
        cc_matrix_shifted = np.array(cc_matrix_shifted)
        logL_matrix_shifted = np.array(logL_matrix_shifted)

        # Sum along the phase axis to get a vector with length (nVsys), hence forming the row of Kp-Vsys
        # matrix for this particular value of Kp.
        cc_Kp_Vsys.append(np.sum(cc_matrix_shifted, axis=0))
        logL_Kp_Vsys.append(np.sum(logL_matrix_shifted, axis=0))

    ## Convert to array, with dimension (nKp, nVsys).
    cc_Kp_Vsys = np.array(cc_Kp_Vsys)
    logL_Kp_Vsys = np.array(logL_Kp_Vsys)

    return cc_Kp_Vsys, logL_Kp_Vsys


def get_KpVsys_slow_per_detector_with_model_reprocess(datacube=None, data_wavsoln=None, model_fpfs=None, model_wavsoln=None,
                   Vsys_range=None, Kp_range = None, berv = None, phases = None, reprocess = False, colmask_inds = None,
                    save = None, savedir = None, date = None, idet = None, mn = None, N_PCA = None):
    """
    For a grid of Kp and Vsys directly compute the KpVsys cross-correlation and logL map.
    This function  can be externally be called within a loop over all detectors/orders for a given date. 
    This function also performs the model reprocessing for each Kp-Vsys pair if needed.

    :param datacube: PCA detrended spectral datacube, dimensions : [nspec, nwav]. nspec is same as length of phases.
    :type datacube: array_like
    

    :param data_wavsoln:  Wavelength solution, dimensions : [nwav].
    :type data_wavsoln: array_like
   

    :param model_fpfs: 1D array Model Fp/Fs.
    :type model_fpfs: array_like
    

    :param model_wavsoln: Model wavelength solution, [model_wavelength]. For non-reprocessed model, not required.
    :type model_wavsoln: array_like
    

    :param Vsys_range: Range of Vsys, km/s.
    :type Vsys_range: array_like
    
    :param Kp_range: Range of Kp, km/s.
    :type Kp_range: array_like
    

    :param berv: 1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).
    :type berv: array_like
    

    :param phases: 1D array of orbital phases spanned by the observations.
    :type phases: array_like
    
    :param reprocess: Is the model to be reprocessed?
    :type reprocess: bool
    

    :param colmask_inds: 1D array of cross-dispersion pixel indices you want to mask out.
    :type colmask_inds: array_like 
    
    :param save: Do you want to save the reprocessed model?
    :type save: bool 

    :param savedir: Directory where you want to save the reprocessed model numpy arrays. Required only if save is True.
    :type savedir: str
    

    :param idet: Detector index string. Required only if save is True.
    :type idet: str

    :param date: Date. Required only if save is True. 
    :type data: str
    
    :param mn: str
    Model string. Required only if save is True.

    :param N_PCA:  Number of PCA iterations being used. Required only if save is True. 
    :type N_PCA: int
   
    :return: The CC and logL KpVsys map for this particular date and detector.
    :rtype: array_like

    """
    nspec, nwav, nKp, nVsys = datacube.shape[0], datacube.shape[1], len(Kp_range), len(Vsys_range)
    R, sg2, CC, logL = np.empty((nspec, nKp, nVsys)), np.empty((nspec, nKp, nVsys)), np.empty((nspec, nKp, nVsys)), np.empty((nspec, nKp, nVsys))

    if not reprocess:
        # If the model is not reprocessed, you only need to run the interpolation once,
        # else you need to repeat this for every nKp, nVsys.
        # model_spl = splrep(model_wavsoln, model_fpfs)
        model_spl = interpolate.make_interp_spline(model_wavsoln, model_fpfs)

        # If the model is not reprocessed, the datacube here is already detrended, so subtract
        # the mean from the data first, ignoring the contribution of zero values in the datacube to the mean contribution
        datacube_mean_sub = np.empty((nspec,nwav))
        zeroMask = datacube[:,:] == 0 ## Get the mask for all the values where the datacube is set to zero.
        # This should have already been done at the stage of removing the PCA linear regression fit from the data.
        datacube_mean_sub[:,:] = sub_mask(datacube[:,:], zeroMask)
        

    for iKp in tqdm(range(nKp)):
        for iVsys in range(nVsys):
            if not reprocess:
                for it in range(nspec):
                    
                    # Compute the total RV
                    RV = Kp_range[iKp] * np.sin(2. * np.pi * phases[it]) + Vsys_range[iVsys] + berv[it]
                    
                    # Doppler shift the wavelength solution of the data by -RV (so when model is interpolated to it the model will be effectively 
                    # shifted by +RV)
                    data_wavsoln_shift = doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * RV)
                    
                    # Evaluate the model to the data wavelength solution shifted by -RV.
                    # This shifts the wavelength solution by -RV, but when the model is interpolated onto 
                    # this shifted wavelength solution, the model is effectively shifted by +RV because it is only the wavelength grid that has moved.
                    # model_spec_flux_shift = splev(data_wavsoln_shift, model_spl)
                    model_spec_flux_shift = model_spl(data_wavsoln_shift)
                    
                    # Subtract the mean of the model from the model
                    model_spec_flux_shift = model_spec_flux_shift - fast_mean(model_spec_flux_shift)
                    
                    # Compute the cross-correlation and log-likelihood 
                    R[it,iKp,iVsys], CC[it, iKp, iVsys], logL[it, iKp, iVsys] = fast_cross_corr(data=datacube_mean_sub[it, :],
                                                                                        model=model_spec_flux_shift)
                    sg2[it,iKp,iVsys] = fast_var(model_spec_flux_shift)
                    
  
            else:
                # First get the reprocessed modelcube (nspec, nwav) for this particular Kp-Vsys value
                datacube_detrended, model_reprocess = stc.reprocess_model_per_detector_per_KpVsys(
                                                        datacube=datacube, data_wavsoln=data_wavsoln,
                                                        model_fpfs=model_fpfs, model_wavsoln=model_wavsoln,
                                                        Vsys=Vsys_range[iVsys], Kp=Kp_range[iKp],
                                                        berv=berv, phases=phases, colmask_inds=colmask_inds,
                                                        N_PCA=N_PCA, save=save, savedir=savedir, date=date, idet=idet, mn=mn)

                # Mean subtract the data, ignoring contributions from zero values
                datacube_mean_sub = np.empty((nspec, nwav))
                zeroMask = datacube_detrended[:, :] == 0  ## Get the mask for all the values where the datacube is set to zero.
                # This should have already been done at the stage of removing the PCA linear regression fit from the data.
                datacube_mean_sub[:, :] = sub_mask(datacube_detrended[:, :], zeroMask)

                # Loop over time to compute the cross-correlation and log-likelihood values.
                for it in range(nspec):
                    model_spec_flux_shift = model_reprocess[it, :]
                    # Mean subtract the model
                    model_spec_flux_shift = model_spec_flux_shift - fast_mean(model_spec_flux_shift)
                    R[it, iKp, iVsys], CC[it, iKp, iVsys], logL[it, iKp, iVsys] = fast_cross_corr(data=datacube_mean_sub[it, :],
                                                                                        model=model_spec_flux_shift)
                    sg2[it,iKp,iVsys] = fast_var(model_spec_flux_shift)

    CC_fin, logL_fin = np.sum(CC, axis=0), np.sum(logL, axis=0)
    R_fin, sg2_fin = np.sum(R, axis=0), np.sum(sg2, axis=0)

    return R_fin, sg2_fin, CC_fin, logL_fin


def subplot_cc_matrix(axis=None, cc_matrix=None, phases=None, velocity_shifts=None, title=None,
                      setxlabel=False, plot_type = 'pcolormesh', vminvmax = None):
    """
    This function makes a 2D plot of a datacube with specified phases, and user specified title.
    
    :param axis: Axis object corresponding to a subplot from a predefined figure.
    :type axis: matplotlib.axes 
    
    :param cc_matrix: 2D array of CC values with shape [time, velocity_shifts].
    :type cc_matrix: array_like
    
    :param phases: Numpy array of the phases corresponding to each exposure, length should be the same as datacube.shape[0].
    :type phases: array_like
    
    :param velocity_shifts: 1D array of velocity shifts (in km/s) for which you want to calculate the cross-correlation coefficients.
    :type velocity_shifts: array_like
    
    :param title: Title for the subplot.
    :type title: str
    
    :param setxlabel: Set true if you want to set the xlabel to Velocity [km/s] for this particular subplot. Default is False.
    :type setxlabel: bool
    
    :return: matplotlib plot object that can be used further.
    :rtype: matplotlib.collections.QuadMesh
    
    """
    if plot_type == 'pcolormesh':
        if vminvmax is None:
            plot_hand = axis.pcolormesh(velocity_shifts, phases, cc_matrix)
        else:
            plot_hand = axis.pcolormesh(velocity_shifts, phases, cc_matrix, norm=mpl.colors.Normalize(vmin=vminvmax[0], vmax=vminvmax[1]), shading='auto' )
    elif plot_type == 'contourf': # Useful for sigma contours
        plot_hand = axis.contourf(velocity_shifts, phases, cc_matrix, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    axis.set_title(title)
    axis.set_ylabel(r'$\phi$')
    if setxlabel:
        axis.set_xlabel('Velocity [km/s]')
    return plot_hand

def get_sigma_contours(logL_KpVsys = None, dof = None):
    """
    Using Wilk's theorem, convert a given logL KpVsys map to sigma values. See Appendix D of
     https://arxiv.org/pdf/2004.11335.pdf and Section 5.1 of https://arxiv.org/pdf/2205.14975.pdf
     for reference.

    :param logL_KpVsys: 2D array of logL values.
    :type logL_KpVsys: array_like
    

    :param dof: Degree of freedom of your model with respect to which you want to calculate the sigma.
    should be 2 for KpVsys map (1 for each Kp and Vsys).
    :type dof: float64

    :return: 2D array of sigma values, same dimension as logL_KpVsys.
    :rtype: array_like
    """

    DeltalogL = logL_KpVsys - np.max(logL_KpVsys)

    chi2 = -2. * DeltalogL

    p_one_tail = 0.5 * scipy.stats.chi2.sf(chi2, dof)

    # sigma_levels = scipy.stats.norm.ppf(1-p_one_tail, dof) ## Note that norm.ppf(0.95) is same as norm.isf(0.05)
    sigma_levels = scipy.stats.norm.isf(p_one_tail) ## Gives better precision

    return sigma_levels

def get_2D_grid_search(logL_maps = None, Kp_range = None, Vsys_range = None, KpVsys_sdd = None):
    """Perform a 2D grid comparison across an array of logL_maps computed for a grid of models with only 2 varying parameters. 

    :param logL_maps: Dictionary of logL KpVsys maps, with the keys as tuples describing the values of 2 varying parameters. Defaults to None.
    :type logL_maps: dict
    :param Kp_range: Array of Kp values for which the logL map has been computed, defaults to None
    :type Kp_range: array_like
    :param Vsys_range: Array of Vsys values for which the logL map has been computed. , defaults to None
    :type Vsys_range: array_like
    :param KpVsys_sdd: Dictionary of two arrays, each specifying the range of Kp and Vsys values. This is to define the region for local search of maximum logL value for a given CC map, defaults to None
    :type KpVsys_sdd: array_like
    :return: Grid of sigma values for each mdoel.
    :rtype: dict
    """
    Kp_srange, Vsys_srange = KpVsys_sdd['Kp'],  KpVsys_sdd['Vsys'],
    Kp_inds = np.array( [min(range(len(Kp_range)), key=lambda i: abs(Kp_range[i]-k)) for k in Kp_srange] )
    Vsys_inds = np.array( [min(range(len(Vsys_range)), key=lambda i: abs(Vsys_range[i]-k)) for k in Vsys_srange] )
    
    logL_max = { mk:np.max(logL_maps[mk][Vsys_inds[0]:Vsys_inds[1], Kp_inds[0]:Kp_inds[1]]) for mk in logL_maps.keys()}
    print(logL_max)
    print(max(logL_max, key = logL_max.get), logL_max[max(logL_max, key = logL_max.get)])
    # Find the model key which has the maximum logL value 
    maxmk = max(logL_max, key=logL_max.get)
    
    logL_sigma_grid = {}
    
    for mk in logL_max.keys():
        
        DeltalogL = logL_max[mk] - logL_max[maxmk]

        chi2 = -2. * DeltalogL

        p_one_tail = 0.5 * scipy.stats.chi2.sf(chi2, 2)

        # sigma_levels = scipy.stats.norm.ppf(1-p_one_tail, dof) ## Note that norm.ppf(0.95) is same as norm.isf(0.05)
        sigma_level = scipy.stats.norm.isf(p_one_tail) ## Gives better precision
        
        logL_sigma_grid[mk] = sigma_level
    
    return logL_sigma_grid

    # """

    # Args:
    #     axis (matplotlib axis object, optional): Axis, if for a previously generated figure with subplots. Defaults to None.
    #     matrix_2D (array, optional): 2D color map. Defaults to None.
    #     Y (array): Array of Y axis values. Defaults to None.
    #     X (array): Array of X axis values. Defaults to None.
    #     title (str): Title of the plot. Defaults to None.
    #     setxlabel (str): Label of the X axis. Defaults to False.
    #     setylabel (str): Label of the Y axis. Defaults to None.
    #     plot_type (str): The type of colormap plot. Defaults to 'pcolormesh'.
    # """
def plot_2D_cmap(axis=None, matrix_2D=None, Y=None, X=None, title=None,
                      setxlabel=False, setylabel=False, plot_type = 'pcolormesh', ylabel = None, xlabel = None):
    """Plot a 2D colormap for a given 2D matrix.

    :param axis: Axis object corresponding to a subplot from a predefined figure., defaults to None
    :type axis: matplotlib.axes, optional
    
    :param matrix_2D: 2D matrix to be plotted, defaults to None
    :type matrix_2D: array_like
    
    :param Y: Vector of Y values, defaults to None
    :type Y: array_like
    
    :param X: Vector of X values, defaults to None
    :type X: array_like
    
    :param title: Title of the plot, defaults to None
    :type title: str
    
    :param setxlabel: Set True if you want to set the xlabel for this particular subplot, defaults to False
    :type setxlabel: bool, optional
    
    :param xlabel: X label you want to set, defaults to None.
    :type xlabel: str, optional
    
    :param setylabel: Set True if you want to set the ylabel, defaults to False
    :type setylabel: bool, optional
    
    :param ylabel: Y label you want to set, defaults to None.
    :type ylabel: str, optional

    
    :param plot_type: _description_, defaults to 'pcolormesh'
    :type plot_type: str, optional
    
    :return: Figure object and plot object.
    :rtype: matplotlib.figure and matplotlib.collections.QuadMesh
    """
    
    if axis is None:
        fig = plt.figure(figsize = (10,10))
        axis = plt.gca()
        
    if plot_type == 'pcolormesh':
        plot_hand = axis.pcolormesh(X, Y, matrix_2D)
    elif plot_type == 'contourf': # Useful for sigma contours
        plot_hand = axis.contourf(X, Y, matrix_2D, [0,1,2,3,4,5,6,7,8,9,10])
    axis.set_title(title)
    if setylabel:
        axis.set_ylabel(ylabel)
    if setxlabel:
        axis.set_xlabel(xlabel)
    return fig, plot_hand

