import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from . import cross_correlation_utils as crocut
# from scipy.interpolate import splev, splrep
from scipy import interpolate
from tqdm import tqdm
from scipy import stats
from memory_profiler import profile
import gc
import corner

#########################################################################################################
# General purpose functions ###############################################################
#########################################################################################################

def polyfunc(x, *coeff_array):
    """
    This function computes a polynomial of N degree, where N = len(coeff_array) - 1.
    :param x: Independent variable for which to compute the polynomial function.
    :param coeff_array: Numpy array, 1D, length = N+1, of coefficients for an N degree polynomial.
    :return: Computed value of the N degree polynomial.
    """
    return sum([p * (x ** i) for i, p in enumerate(coeff_array)])


def multi_linfunc(x, *coeff_array):
    """
    This function computes a linear combination of multiple regressors.
    :param x: Array of regressors, each of the same length.
    :param coeff_array: Numpy array, 1D, coefficients for each regressor.
    :return: Computed value of the linear combination of each regressor.
    """
    return sum([p * x[i] for i, p in enumerate(coeff_array)])


def expfunc(x, *coeff_array):
    """
    This function computes an exponential of x along with an additive coefficent.
    :param x: Independent variable for which to compute the polynomial function.
    :param coeff_array: Numpy array, 1D, length = N+1, of coefficients for an N degree polynomial.
    :return: Computed value of the N degree polynomial.
    """
    a, b, c = coeff_array
    return a * np.exp(b * x) + c


def do_polyreg(regressor=None, data=None, degree=1, method='lm'):
    """
    This function performs a regression between a polynomial of a regressor (independent variable)
    and data (dependent variable) using the LM non-linear least squares fit. data = f(regressor),
    where f(x) is a polynomial of degree 'degree'.
    :param method: Method of the fit you want to perform, choose between ‘lm’, ‘trf’, ‘dogbox’. Default is 'lm'.
    :param regressor: Numpy array, 1D.
    :param data: Numpy array, 1D.
    :param degree: Degree of the polynomial you want to use for the regression. Default is 1.
    :return: A dictionary of the regression solution containing the best fit coefficients and the best fit polynomial
    solution.
    """
    soln_dd = {}
    p0 = [1] * (degree + 1)
    popt, pcov, infodict, mesg, ier = curve_fit(polyfunc, regressor, data, full_output=True, method=method, p0=p0)
    # Passing the starting guess p0 as the length degree+1 will make sure curve_fit passes the correct length of params
    # to polyfunc (which can take N length of params).
    soln_dd['popt'], soln_dd['pcov'], soln_dd['infodict'] = popt, pcov, infodict  # Save the solution to the dictionary.
    soln_dd['func_val'] = polyfunc(regressor, *popt)
    return soln_dd


def do_multi_linreg(regressors=None, data=None, method='lm'):
    """
    This function performs a regression between linear combination of multiple regressors (independent variables)
    and data (dependent variable) using the LM non-linear least squares fit. data = f(regressor),
    where f(x) is a linear combination of each regressor.
    More specifically, data = a1*x^deg + a2*x^deg + a3*x^deg ... + an*x^deg.
    :param method: Method of the fit you want to perform, choose between ‘lm’, ‘trf’, ‘dogbox’. Default is 'lm'.
    :param regressors: Numpy array of arrays, each array an individual regressor of length same as time.
    :param data: Numpy array, 1D.
    :return: A dictionary of the regression solution containing the best fit coefficients and the best fit polynomial
    solution.
    """
    soln_dd = {}
    p0 = [1] * regressors.shape[0]
    popt, pcov, infodict, mesg, ier = curve_fit(multi_linfunc, regressors, data, full_output=True, method=method,
                                                p0=p0, maxfev=10000000)
    soln_dd['popt'], soln_dd['pcov'], soln_dd['infodict'] = popt, pcov, infodict  # Save the solution to the dictionary.
    soln_dd['func_val'] = multi_linfunc(regressors, *popt)
    return soln_dd


def do_expreg(regressor=None, data=None, method='lm'):
    """
    This function performs a regression between exponential of a regressor (independent variable)
    and data (dependent variable) using the LM non-linear least squares fit. data = f(regressor),
    where f(x) is a polynomial of degree 'degree'.
    :param method: Method of the fit you want to perform, choose between ‘lm’, ‘trf’, ‘dogbox’. Default is 'lm'.
    :param regressor: Numpy array, 1D.
    :param data: Numpy array, 1D.
    :return: A dictionary of the regression solution containing the best fit coefficients and the best fit polynomial
    solution.
    """
    soln_dd = {}
    p0 = [1, -1, 0]
    popt, pcov, infodict, mesg, ier = curve_fit(expfunc, regressor, data, full_output=True, method=method, p0=p0,
                                                maxfev=10000000)
    # Passing the starting guess p0 as the length degree+1 will make sure curve_fit passes the correct length of params
    # to polyfunc (which can take N length of params).
    soln_dd['popt'], soln_dd['pcov'], soln_dd['infodict'] = popt, pcov, infodict  # Save the solution to the dictionary.
    soln_dd['func_val'] = expfunc(regressor, *popt)
    return soln_dd


#########################################################################################################
# HIRES analysis specific functions #######################################################
#########################################################################################################

def subplot_datacube(axis=None, datacube=None, phases=None, wavsoln=None, title=None, setxlabel=False,
                     vminvmax=None):
    """
    This function makes a 2D plot of a datacube with specified phases, and user specified title.
    :param axis: Axis object corresponding to a subplot from a predefined figure.

    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra; dimensions should be [time,wavelength].

    :param phases: array_like
    Numpy array of the phases corresponding to each exposure, length should be the same as datacube.shape[0].

    :param wavsoln: array_like
    Numpy array of the wavelength solution, length should be the same as datacube.shape[1]. Should always be in nm.

    :param title: str
    Title for the subplot.

    :param setxlabel: bool
    Set true if you want to set the xlabel to Wavelength for this particular subplot. Default is False.
    :return: plot_hand, the handle of the plot.
    """
    if vminvmax is None:
        plot_hand = axis.pcolormesh(wavsoln, phases, datacube,
                                    norm=mpl.colors.Normalize(), shading='auto', rasterized = True)
    else:
        plot_hand = axis.pcolormesh(wavsoln, phases, datacube,
                                    norm=mpl.colors.Normalize(vmin=vminvmax[0], vmax=vminvmax[1]), shading='auto', rasterized = True)
    axis.set_title(title)
    axis.set_ylabel(r'$\phi$')
    if setxlabel:
        axis.set_xlabel('Wavelength [nm]')
    return plot_hand


def normalize_data(datacube=None, norm_method='median_first_exp'):
    """
    This function normalizes spectra time series data cube by the method prescribed by the user. :param norm_method:
    str, optional Choose from 'median_first_exp', 'median_per_exp'. Method of normalizing the spectra for each time
    instance. Default is 'median_first_exp' of flux values for the whole wavelength range for the first exposure.
    'median_per_exp' would normalize each exposure by the median of the respective exposure. :param datacube:
    array_like Numpy array of timeseries high-resolution spectra; dimensions should be [time,wavelength]. :return:
    array_like Normalized datacube.
    """
    if norm_method == 'median_first_exp':
        datacube_norm = np.array([datacube[i, :] / np.median(datacube[0, :]) for i in range(datacube.shape[0])])
    elif norm_method == 'median_per_exp':
        datacube_norm = np.array([datacube[i, :] / np.median(datacube[i, :]) for i in range(datacube.shape[0])])
    return datacube_norm


def detrend_airmass_polyreg(datacube=None, airmass=None):
    """
    This function performs column by column detrending of the datacube using airmass as a linear regressor.

    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already; dimensions should be [time,wavelength].

    :param airmass: array_like
    Numpy array of airmass values for each time
    instance the spectra is measured.

    :return: array_like
    Numpy array of timeseries high-resolution spectra with each channel time series
    now detrended using a linear function of airmass. The linear regression is performed using a polyreg
    (only a simple non-linear least-squares fit for now and no MCMC.)
    """
    datacube_detrended = []
    datacube_fit_vals = []
    for ww in range(datacube.shape[1]):
        obs_channel_vals = datacube[:, ww]  # This is the time series for the ww'th pixel channel.
        fit_channel_soln = do_polyreg(regressor=airmass,
                                      data=obs_channel_vals)  # Fit the linear function of airmass.
        # to the observed channel time series values.
        fit_channel_vals = fit_channel_soln['func_val']
        fit_channel_detrended = obs_channel_vals / fit_channel_vals
        # Append to the lists
        datacube_detrended.append(fit_channel_detrended)
        datacube_fit_vals.append(fit_channel_vals)
    return np.array(datacube_detrended).T, np.array(datacube_fit_vals).T


def detrend_airmass_expreg(datacube=None, airmass=None):
    """
    This function performs column by column detrending of the datacube using exponential of airmass as a linear regressor.
    The atmospheric extinction coefficient is A = exp(-b*airmass).

    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already; dimensions should be [time,wavelength].

    :param airmass: array_like
    Numpy array of airmass values for each time
    instance the spectra is measured.

    :return: array_like
    Numpy array of timeseries high-resolution spectra with each channel time series
    now detrended using a linear function of airmass. The linear regression is performed using a polyreg
    (only a simple non-linear least-squares fit for now and no MCMC.)
    """
    datacube_detrended = []
    datacube_fit_vals = []
    for ww in range(datacube.shape[1]):
        obs_channel_vals = datacube[:, ww]  # This is the time series for the ww'th pixel channel.
        fit_channel_soln = do_expreg(regressor=airmass,
                                     data=obs_channel_vals)  # Fit the linear function of logarithm of airmass.
        # to the observed channel time series values.
        fit_channel_vals = fit_channel_soln['func_val']
        fit_channel_detrended = obs_channel_vals / fit_channel_vals
        # Append to the lists
        datacube_detrended.append(fit_channel_detrended)
        datacube_fit_vals.append(fit_channel_vals)
    return np.array(datacube_detrended).T, np.array(datacube_fit_vals).T


def detrend_lines_linreg(datacube=None, lines_wbounds=None, lines_wbounds_form='wavelength', wavsoln=None):
    """
    This function performs column by column detrending of the datacube using a linear combination of
    integrated flux time series in the wavelength ranges defined by the line_wbounds.

    :param lines_wbounds: Array of wavelength bound range for the lines you want to use as regressors.
    Should be an array of arrays, the individual arrays containing only two elements [start,stop] in wavelength and in nm.

    :param lines_wbounds_form: str
    Format of lines_wbounds, should be either 'wavelength' or 'pixels'. Default is 'wavelength'.

    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already and airmass detrended; dimensions should be [time,wavelength].

    :param wavsoln: array_like
    Numpy array of the wavelength solution, length should be the same as datacube.shape[1]. Should always be in nm.

    :return: np.array(datacube_detrended).T, np.array(datacube_fit_vals).T
    Numpy array of timeseries high-resolution spectra with each channel time series
    now detrended using a linear function of airmass. The linear regression is performed using a polyreg
    (only a simple non-linear least-squares fit for now and no MCMC.)
    """
    # Calculate the time series of integrated flux in line specified by the line_wbounds
    line_integflux_list = []
    for ww in range(lines_wbounds.shape[0]):
        line_integflux = get_line_integflux_tseries(datacube=datacube, line_wbounds=lines_wbounds[ww, :],
                                                    line_wbounds_form=lines_wbounds_form,
                                                    wavsoln=wavsoln)
        line_integflux_list.append(line_integflux)
    line_integflux_array = np.array(line_integflux_list)

    datacube_detrended = []
    datacube_detrended_renorm = []
    datacube_fit_vals = []
    for ww in range(datacube.shape[1]):
        obs_channel_vals = datacube[:, ww]  # This is the time series for the ww'th pixel channel.
        fit_channel_soln = do_multi_linreg(regressors=line_integflux_array,
                                           data=obs_channel_vals)  # Fit the linear function of airmass
        # to the observed channel time series values.
        fit_channel_vals = fit_channel_soln['func_val']
        fit_channel_detrended = obs_channel_vals / fit_channel_vals

        # Also re-normalize the column by first subtracting its mean and normalizing by standard deviation
        fit_channel_detrended_renorm = (fit_channel_detrended - np.nanmean(fit_channel_detrended))/ np.nanstd(fit_channel_detrended)
        # Currently not sure if the renorm above is useful.

        # Append to the lists
        datacube_detrended.append(fit_channel_detrended)
        datacube_detrended_renorm.append(fit_channel_detrended_renorm)
        datacube_fit_vals.append(fit_channel_vals)
    return np.array(datacube_detrended_renorm).T, np.array(datacube_detrended).T, np.array(datacube_fit_vals).T


def get_wave_indices(wave_points=None, wave_soln_list=None):
    """
    :param wave_points: array_like
    Values of wavelengths at which you want to find the closest index in the wavelength solution(s).

    :param wave_soln_list: array_like
    Array of different wavelength solutions for which you want to find the closest indices for the given wave_points.
    Must be 2D with dimensions [num, wavelength] where num is the number of different wavelength solutions you want to do the computations for. num >= 1.
    :return: Array of indices in the corresponding wavelength solution(s) for the given wave_points. Format : [[start,stop],[start,stop], ...]
    """
    inds_list = []
    for l in range(wave_soln_list.shape[0]):
        inds = np.array(
            [min(range(len(wave_soln_list[l])), key=lambda i: abs(wave_soln_list[l][i] - k)) for k in wave_points])
        inds_list.append(inds)
    return np.array(inds_list)
    # else:
    #     inds = np.array(
    #         [min(range(len(wave_soln_list)), key=lambda i: abs(wave_soln_list[i] - k)) for k in wave_points])
    #     return inds[0]


def get_line_integflux_tseries(datacube=None, line_wbounds=None, line_wbounds_form='wavelength', wavsoln=None):
    """
    This function computes the time series of the integrated flux for a single chosen line in the spectrum
    (defined by specifying a list of wavelength ranges). Need to define at least one of the two: line_wbounds or
    line_wbounds_pix.

    :param line_wbounds_form: str
    Format of line_wbounds, can be either 'wavelength' or 'pixels', default is 'wavelength'.

    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already; dimensions should be [time,wavelength].

    :param wavsoln: array_like
    Numpy array of the wavelength solution, length should be the same as datacube.shape[1]. Should always be in nm.

    :param line_wbounds: Wavelength bound range for the line. Should be a list of only two elements [start,stop] in wavelength.

    :return: Array of integrated flux of the line with time; shape [time,].
    """
    # Get the indices for the line_wbounds
    if line_wbounds_form == 'wavelength':
        wave_inds = get_wave_indices(wave_points=line_wbounds, wave_soln_list=np.array([wavsoln]))
        # Pay attention to the way get_wave_indices outputs the wave_inds, it has a different shape than line_wbounds.
        # It actually maintains the same shape in the first dimension as wave_soln_list (it repeats the calcs for each
        # wavelength solution you pass in wave_soln_list.
        print('Calculated wavelength indices ', wave_inds)
        print('The line for which we are integrating the flux for is between ', wavsoln[wave_inds[0][0]], ' and ',
              wavsoln[wave_inds[0][1]], ' nm...')
        # Sum the flux in the range wave_inds[0][0] to wave_inds[0][1]
        line_integflux = np.nansum(datacube[:, wave_inds[0][0]:wave_inds[0][1]], axis=1)

    elif line_wbounds_form == 'pixels':
        wave_inds = line_wbounds
        print('Calculated wavelength indices ', wave_inds)
        print('The line for which we are integrating the flux for is between ', wavsoln[wave_inds[0]], ' and ',
              wavsoln[wave_inds[1]], ' nm...')
        # Sum the flux in the range wave_inds[0][0] to wave_inds[0][1]
        line_integflux = np.nansum(datacube[:, wave_inds[0]:wave_inds[1]], axis=1)

    return line_integflux


# Function to standardise data
def standardise_data(datacube=None):
    """
    Standardise datacube before running the PCA on it.
    :param datacube: array_like
    Numpy array of timeseries high-resolution spectra, ideally with each exposure normalized
    already; dimensions should be [time,wavelength].

    :return: Standardised array of datacube, in the same format as the original datacube.
    """
    nf, nx = datacube.shape
    fStd = datacube.copy()
    for i in range(nx):
        fStd[:,i] -= np.mean(fStd[:,i])
        # This is the biased stdev (normalised by nx rather than nx-1)
        # It needs changing to match CORRELATE.pro
        fStd[:,i] /= np.std(fStd[:,i])
    fStd = np.nan_to_num(fStd,0.) ## This is in case a whole spectral channel was set to zero pre-standardisation, which can lead to some spectral channels being nans. 
    return fStd

def get_eigenvalues_via_PCA(datacube=None,reconstruct=False, data_mask = None):
    """
    Run the SVD on the datacube [nf, nx] and return all the eigenvalues in addition to a
    column of 1s.
    :param datacube: array_like
    Spectral datacube, already standardised.

    :param data_mask: array_like, bool
    1D array of bool values to indicate the wavelength channels with bad pixels that should be excluded from the cross-correlation.

    :return: Array of eigenvalues.
    """
    nf, nx = datacube.shape # nf : number of frames, nx : number of wavelength channels
    u, s, vh = np.linalg.svd(datacube[:,data_mask], full_matrices=False) # u is the matrix of eigenvectors (shape : (nf, nx) ),
                                                       # s is a vector of eigenvalues. vh is the Unitary array.

    return s

def get_eigenvectors_via_PCA(datacube=None,nc=None,reconstruct=False, data_mask = None):
    """
    Run the SVD on the datacube [nf, nx] and return the first 'nc' eigenvectors in addition to a
    column of 1s. If 'reconstruct=True', it also returns the reconstructed matrix with the other (nf-nc) components.

    :param datacube: array_like
    Spectral datacube, already standardised.

    :param nc: float
    Number of eigenvector components you want to select for later performing linear regression.

    :param reconstruct: bool
    True, if you want to reconstruct matrix from the SVD solution by combination of nc eigenvectors.

    :param data_mask: array_like, bool
    1D array of bool values to indicate the wavelength channels with bad pixels that should be excluded from the cross-correlation.

    :return: Depends on reconstruct; if reconstruct is True, returns the vector of eigenvectors and reconstructed matrix. Else only returns the former.
    """
    nf, nx = datacube.shape # nf : number of frames, nx : number of wavelength channels
    xMat = np.ones((nf,nc+1)) # The second dimension is nc+1 because besides the nc eigenvectors
                              # you want to have the first component to be 1 (required for the multi-linear regression).
    u, s, vh = np.linalg.svd(datacube[:, data_mask], full_matrices=False) # u is the matrix of eigenvectors (shape : (nf, nx) ),
                                                       # s is a vector of eigenvalues. vh is the Unitary array.
    xMat[:,1:] = u[:,0:nc] # Take only nc eigenvectors.
    if reconstruct:
        ss = s.copy()
        ss[0:nc] = 0.0
        ww = np.diag(ss)
        res = u.dot(ww.dot(vh))
        return xMat, res
    else:
        return xMat

def get_eigenvectors_via_PCA_Matteo(datacube=None, nc = None, reconstruct = False):
    nf, nx = datacube.shape
    xMat = np.ones((nf,nc+1))
    u, s, vh = np.linalg.svd(datacube, full_matrices=False)
    # import pdb
    # pdb.set_trace()
    xMat[:,1:] = u[:,0:nc]
    if reconstruct:
        ss = s.copy()
        ss[0:nc] = 0.0
        ww = np.diag(ss)
        res = u.dot(ww.dot(vh))
        return xMat, res
    else:
        return xMat

def get_eigenvalues_via_PCA_Matteo(datacube=None):
    nf, nx = datacube.shape
    u, s, vh = np.linalg.svd(datacube, full_matrices=False)

    return s


def linear_regression(X=None,Y=None):
    """
    Calculate the multi-variate linear regression fit between the matrix of
    eigenvectors X [nf, nc] and the observed spectral datacube Y [nf, nx].
    :param X: array_like
     Matrix of eigenvectors, shape [nf, nc] i.e. [time, number of components]

    :param Y: array_like
    Datacube, shape [nf, nx] i.e. [time, wavelength]

    :return: Calculated PCA fit to the datacube using the input eigenvector matrix.
    """

    XT = X.T
    term1 = np.linalg.inv(np.dot(XT,X))
    term2 = np.dot(term1,XT)
    beta = np.dot(term2,Y)
    return np.dot(X,beta)

def reprocess_model(datacube=None, data_wavsoln=None, model_fpfs=None, model_wavsoln=None,
                   Vsys_range=None, Kp_range=None, berv=None, phases=None, colmask_inds = None, N_PCA = None):
    """
    Compute the reprocessed model to replicate the effect of PCA on the planetary signal in the data
    on the model planetary signal.

    :param datacube: array_like
    Raw spectral datacube, dimensions : [nDet, nspec, nwav]. nspec should be same as length of phases.

    :param data_wavsoln: array_like
    Wavelength solution, dimensions : [nDet, nwav].

    :param model_fpfs: array_like
    1D array of Model Fp/Fs.

    :param model_wavsoln: array_like
    Model wavelength solution, [model_wavelength].

    :param Vsys_range: array_like
    Range of Vsys, km/s.

    :param Kp_range: array_like
    Range of Kp, km/s.

    :param berv: array_like
    1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).

    :param phases: array_like
    1D array of orbital phases spanned by the observations. Length same as nspec.

    :param colmask_inds: array_like
    1D array of cross-dispersion pixel indices you want to mask out.

    :param N_PCA: int
    Number of PCA components

    :return: Reprocessed modelcube, with dimensions [nKp, nVsys, nDet, time, model_wavelength].
    """
    ndet, nspec, nwav = datacube.shape[0], datacube.shape[1], datacube.shape[2]
    nKp, nVsys = len(Kp_range), len(Vsys_range)

    # Initialize the reprocessed modelcube
    model_reprocess = np.empty((nKp, nVsys, ndet, nspec, nwav))
    datamodel = np.empty((nKp, nVsys, ndet, nspec, nwav))
    datamodel_fit = np.empty((nKp, nVsys, ndet, nspec, nwav))
    datamodel_detrended = np.empty((nKp, nVsys, ndet, nspec, nwav))

    # Standardise the datacube first
    datacube_standard = np.empty((ndet, nspec, nwav))
    for idet in range(ndet):
        datacube_standard[idet, :, :] = standardise_data(datacube[idet, :, :])

    # Create a column mask for bad pixels, same for all dates and detectors
    colmask = np.zeros(datacube.shape[2], dtype=bool)
    colmask[colmask_inds] = True

    # Run the PCA just once on the datacube
    pca_eigenvectors = np.empty((ndet,nspec,N_PCA+1))
    datacube_fit = np.empty((ndet, nspec, nwav))
    datacube_detrended = np.empty((ndet, nspec, nwav))

    for idet in range(ndet):
        fStd = datacube_standard[idet, :, :].copy()
        fStd[:, colmask] = 0
        pca_eigenvectors[idet, :, :] = get_eigenvectors_via_PCA_Matteo(fStd[:, colmask == False], nc=N_PCA)

        datacube_fit[idet, :, :] = linear_regression(X=pca_eigenvectors[idet, :, :], Y=datacube[idet, :, :])
        datacube_detrended[idet, :, :] = datacube[idet, :, :]/datacube_fit[idet, :, :] - 1.

        datacube_detrended[idet, :, colmask] = 0

    # Start the loop for nKp, nVsys. For each pair shift and inject the model in the data
    # And run the PCA with the same eigenvectors
    # model_spl = splrep(model_wavsoln, model_fpfs)
    model_spl = interpolate.make_interp_spline(model_wavsoln, model_fpfs)

    for iKp in tqdm(range(nKp)):
        for iVsys in range(nVsys):
            for idet in range(ndet):
                # Replicate the model_fpfs_shift across all exposures
                model_fpfs_shift_cube = np.empty((nspec, nwav))
                for it in range(nspec):
                    RV = Kp_range[iKp] * np.sin(2. * np.pi * phases[it]) + Vsys_range[iVsys] + berv[it]
                    data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[idet, :], velocity=-1. * RV)
                    # model_fpfs_shift_exp = splev(data_wavsoln_shift, model_spl)
                    model_fpfs_shift_exp = model_spl(data_wavsoln_shift)
                    model_fpfs_shift_cube[it, :] = model_fpfs_shift_exp

                # Inject the model to the data
                datamodel[iKp, iVsys, idet, :, :] = datacube[idet, :, :] * (1. + model_fpfs_shift_cube)

                datamodel_fit[iKp, iVsys, idet, :, :] = linear_regression(X=pca_eigenvectors[idet, :, :],
                                                        Y=datamodel[iKp, iVsys, idet, :, :])

                datamodel_detrended[iKp, iVsys, idet, :, :] = datamodel[iKp, iVsys, idet, :, :]/datamodel_fit[iKp, iVsys, idet, :, :] - 1.

                # Mask the same channels/columns as done for the datacube
                datamodel_detrended[iKp, iVsys, idet, :, colmask] = 0

                model_reprocess[iKp, iVsys, idet, :, :] = datamodel_detrended[iKp, iVsys, idet, :, :] - datacube_detrended[idet, :, :]
                # Can also try (datamodel_detrended - datacube_detrended ) / datacube_detrended

    return datacube_detrended, model_reprocess

def reprocess_model_per_detector(datacube=None, data_wavsoln=None, model_fpfs=None, model_wavsoln=None,
                   Vsys_range=None, Kp_range=None, berv=None, phases=None, colmask_inds = None, N_PCA = None):
    """
    Compute the reprocessed model to replicate the effect of PCA on the planetary signal in the data
    on the model planetary signal. Do this per detector/order.

    :param datacube: array_like
    Raw spectral datacube, dimensions : [nspec, nwav]. nspec should be same as length of phases.

    :param data_wavsoln: array_like
    Wavelength solution, dimensions : [nwav].

    :param model_fpfs: array_like
    1D array of Model Fp/Fs.

    :param model_wavsoln: array_like
    Model wavelength solution, [model_wavelength].

    :param Vsys_range: array_like
    Range of Vsys, km/s.

    :param Kp_range: array_like
    Range of Kp, km/s.

    :param berv: array_like
    1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).

    :param phases: array_like
    1D array of orbital phases spanned by the observations. Length same as nspec.

    :param colmask_inds: array_like
    1D array of cross-dispersion pixel indices you want to mask out.

    :param N_PCA: int
    Number of PCA components

    :return: Reprocessed modelcube, with dimensions [nKp, nVsys, time, model_wavelength].
    """
    nspec, nwav = datacube.shape[0], datacube.shape[1]
    nKp, nVsys = len(Kp_range), len(Vsys_range)

    # Initialize the reprocessed modelcube
    model_reprocess = np.empty((nKp, nVsys, nspec, nwav))
    datamodel = np.empty((nKp, nVsys, nspec, nwav))
    datamodel_fit = np.empty((nKp, nVsys, nspec, nwav))
    datamodel_detrended = np.empty((nKp, nVsys, nspec, nwav))

    # Standardise the datacube first
    datacube_standard = np.empty((nspec, nwav))
    datacube_standard[:, :] = standardise_data(datacube[:, :])

    # Create a column mask for bad pixels, same for all dates and detectors
    colmask = np.zeros(datacube.shape[1], dtype=bool)
    colmask[colmask_inds] = True

    # Run the PCA just once on the datacube
    pca_eigenvectors = np.empty((nspec,N_PCA+1))
    datacube_fit = np.empty((nspec, nwav))
    datacube_detrended = np.empty((nspec, nwav))

    # for idet in range(ndet):
    fStd = datacube_standard[:, :].copy()
    fStd[:, colmask] = 0
    pca_eigenvectors[:, :] = get_eigenvectors_via_PCA_Matteo(fStd[:, colmask == False], nc=N_PCA)

    datacube_fit[:, :] = linear_regression(X=pca_eigenvectors[:, :], Y=datacube[:, :])
    datacube_detrended[:, :] = datacube[:, :]/datacube_fit[:, :] - 1.

    datacube_detrended[:, colmask] = 0

    # Start the loop for nKp, nVsys. For each pair shift and inject the model in the data
    # And run the PCA with the same eigenvectors
    # model_spl = splrep(model_wavsoln, model_fpfs)
    model_spl = interpolate.make_interp_spline(model_wavsoln, model_fpfs)

    for iKp in tqdm(range(nKp)):
        for iVsys in range(nVsys):
            # for idet in range(ndet):
            # Replicate the model_fpfs_shift across all exposures
            model_fpfs_shift_cube = np.empty((nspec, nwav))
            for it in range(nspec):
                RV = Kp_range[iKp] * np.sin(2. * np.pi * phases[it]) + Vsys_range[iVsys] + berv[it]
                data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[:], velocity=-1. * RV)
                # model_fpfs_shift_exp = splev(data_wavsoln_shift, model_spl)
                model_fpfs_shift_exp = model_spl(data_wavsoln_shift)
                model_fpfs_shift_cube[it, :] = model_fpfs_shift_exp

            # Inject the model to the data
            datamodel[iKp, iVsys, :, :] = datacube[:, :] * (1. + model_fpfs_shift_cube)

            datamodel_fit[iKp, iVsys, :, :] = linear_regression(X=pca_eigenvectors[:, :],
                                                    Y=datamodel[iKp, iVsys,:, :])

            datamodel_detrended[iKp, iVsys, :, :] = datamodel[iKp, iVsys, :, :]/datamodel_fit[iKp, iVsys, :, :] - 1.

            # Mask the same channels/columns as done for the datacube
            datamodel_detrended[iKp, iVsys, :, colmask] = 0

            model_reprocess[iKp, iVsys, :, :] = datamodel_detrended[iKp, iVsys, :, :] - datacube_detrended[:, :]
            # Can also try (datamodel_detrended - datacube_detrended ) / datacube_detrended

    return datacube_detrended, model_reprocess

def reprocess_model_per_detector_per_KpVsys(datacube=None, data_wavsoln=None, model_fpfs=None, model_wavsoln=None,
                   Vsys=None, Kp=None, berv=None, phases=None, colmask_inds = None, N_PCA = None,
                                            save = None, savedir = None, date = None, idet = None, mn = None):
    """
    Compute the reprocessed model to replicate the effect of PCA on the planetary signal in the data
    on the model planetary signal. Do this per detector/order and per Kp-Vsys pair.

    :param datacube: array_like
    Raw spectral datacube, dimensions : [nspec, nwav]. nspec should be same as length of phases.

    :param data_wavsoln: array_like
    Wavelength solution, dimensions : [nwav].

    :param model_fpfs: array_like
    1D array of Model Fp/Fs.

    :param model_wavsoln: array_like
    Model wavelength solution, [model_wavelength].

    :param Vsys: float
    Value of Vsys, km/s.

    :param Kp: float
    Value of Kp, km/s.

    :param berv: array_like
    1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).

    :param phases: array_like
    1D array of orbital phases spanned by the observations. Length same as nspec.

    :param colmask_inds: array_like
    1D array of cross-dispersion pixel indices you want to mask out.

    :param N_PCA: int
    Number of PCA components

    :param save: bool
    If you want to save the reprocessed model.

    :param savedir: str
    Directory where you want to save the reprocessed model numpy arrays. Required only if save is True.

    :param idet: str
    Detector. Required only if save is True.

    :param date: str
    Date. Required only if save is True.

    :param mn: str
    Model string. Required only if save is True.

    :return: Detrended datacube and the reprocessed modelcube, with dimensions
    [time, model_wavelength] each.
    """
    nspec, nwav = datacube.shape[0], datacube.shape[1]
    # nKp, nVsys = len(Kp_range), len(Vsys_range)

    # Initialize the reprocessed modelcube
    model_reprocess = np.empty((nspec, nwav))
    datamodel = np.empty((nspec, nwav))
    datamodel_fit = np.empty((nspec, nwav))
    datamodel_detrended = np.empty((nspec, nwav))

    # Standardise the datacube first
    datacube_standard = np.empty((nspec, nwav))
    datacube_standard[:, :] = standardise_data(datacube[:, :])

    # Create a column mask for bad pixels, same for all dates and detectors
    colmask = np.zeros(datacube.shape[1], dtype=bool)
    colmask[colmask_inds] = True

    # Run the PCA just once on the datacube
    pca_eigenvectors = np.empty((nspec,N_PCA+1))
    datacube_fit = np.empty((nspec, nwav))
    datacube_detrended = np.empty((nspec, nwav))

    # for idet in range(ndet):
    fStd = datacube_standard[:, :].copy()
    fStd[:, colmask] = 0
    pca_eigenvectors[:, :] = get_eigenvectors_via_PCA_Matteo(fStd[:, colmask == False], nc=N_PCA)

    datacube_fit[:, :] = linear_regression(X=pca_eigenvectors[:, :], Y=datacube[:, :])
    datacube_detrended[:, :] = datacube[:, :]/datacube_fit[:, :] - 1.

    datacube_detrended[:, colmask] = 0

    # Start the loop for nKp, nVsys. For each pair shift and inject the model in the data
    # And run the PCA with the same eigenvectors
    # model_spl = splrep(model_wavsoln, model_fpfs)
    model_spl = interpolate.make_interp_spline(model_wavsoln, model_fpfs)

    # for iKp in tqdm(range(nKp)):
        # for iVsys in range(nVsys):
            # for idet in range(ndet):
            # Replicate the model_fpfs_shift across all exposures
    model_fpfs_shift_cube = np.empty((nspec, nwav))
    for it in range(nspec):
        RV = Kp * np.sin(2. * np.pi * phases[it]) + Vsys + berv[it]
        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln[:], velocity=-1. * RV)
        # model_fpfs_shift_exp = splev(data_wavsoln_shift, model_spl)
        model_fpfs_shift_exp = model_spl(data_wavsoln_shift)
        model_fpfs_shift_cube[it, :] = model_fpfs_shift_exp

    # Inject the model to the data
    datamodel[:, :] = datacube[:, :] * (1. + model_fpfs_shift_cube)

    datamodel_fit[:, :] = linear_regression(X=pca_eigenvectors[:, :],
                                            Y=datamodel[:, :])

    datamodel_detrended[:, :] = datamodel[:, :]/datamodel_fit[:, :] - 1.

    # Mask the same channels/columns as done for the datacube
    datamodel_detrended[:, colmask] = 0

    model_reprocess[:, :] = datamodel_detrended[:, :] - datacube_detrended[:, :]
    # Can also try (datamodel_detrended - datacube_detrended ) / datacube_detrended

    if save:
        model_reprocess_dd = {}
        model_reprocess_dd['phases'] = phases
        model_reprocess_dd['model_orig'] = model_fpfs
        model_reprocess_dd['model_wav'] = model_wavsoln
        model_reprocess_dd['datamodel'] = datamodel
        model_reprocess_dd['datamodel_detrended'] = datamodel_detrended
        model_reprocess_dd['data_wavsoln'] = data_wavsoln
        model_reprocess_dd['datamodel_fit'] = datamodel_fit
        model_reprocess_dd['model_reprocess'] = model_reprocess
        model_reprocess_dd['datacube_orig'] = datacube
        model_reprocess_dd['datacube_fit'] = datacube_fit
        model_reprocess_dd['datacube_detrended'] = datacube_detrended
        np.save(savedir + 'model_reprocess_' + mn + '_date-' + str(date) + '_det-' + str(idet) + '_KpVsys-'+str(Kp)+'_'+str(Vsys), model_reprocess_dd)

    return datacube_detrended, model_reprocess


def mask_data_post_pca_per_order(cube, maskval = 0., threshold = 'var'):
    nf, nx = cube.shape
    masked = cube.copy()

    if threshold == 'var':
        varVec = np.var(cube,axis=0) #  should be np.std? ## Using variance is just stricter 
    elif threshold == 'std':
        varVec = np.std(cube,axis=0)
    elif threshold == 'none':
        post_pca_mask = np.zeros(nx, dtype = bool)
        return masked, post_pca_mask
        
    ivalid = varVec > 0
    nvalid = ivalid.sum()
    sigVal = stats.norm.isf(1.0/nvalid)
    # print(nvalid, sigVal)
    medVar = np.median(varVec[ivalid])        
    masked[:,varVec > sigVal*medVar] = maskval 
    post_pca_mask = varVec > sigVal*medVar
    return masked, post_pca_mask

# @profile
def get_telluric_trail_matrix_per_detector(datacube=None, data_wavsoln=None, model_tell=None, model_tell_wavsoln=None,
                   vel_range = None, berv=None, phases = None, N_PCA = None,
                                            save = True, savedir = None, date = None, idet = None,
                                            colmask_info = None, return_trail_matrix = False, post_pca_threshold_type = 'var'):
    """
    For a given N_PCA, detrend the data and then cross-correlate the telluric model with the detrended datacube, 
    with the goal to determine the optimal N_PCA for detrending. Will need to run this module in a loop for a range of N_PCA to be able to 
    determine the effect of changing the number of PCA components when detrending. 

    :param datacube: array_like
    Raw spectral datacube, dimensions : [nspec, nwav]. nspec should be same as length of phases.

    :param data_wavsoln: array_like
    Wavelength solution, dimensions : [nwav].

    :param model_tell: array_like
    1D array of telluric model, e.g. from ESO Sky Calc.

    :param model_tell_wavsoln: array_like
    Wavelength solution of the telluric model, wavelength solution, [model_wavelength].

    :param vel_range: float
    Range of vel by which you shift the telluric model before cross-correlating, km/s.

    :param berv: array_like
    1D array of BERV, by default the velocity of the barycenter in the observer's rest frame (-ve of what Astropy calculates).

    :param phases: array_like
    1D array of orbital phases spanned by the observations. Length same as nspec.

    :param colmask_inds: array_like
    1D array of cross-dispersion pixel indices you want to mask out.

    :param N_PCA: int
    Number of PCA components

    :param save: bool
    If you want to save the reprocessed model.

    :param savedir: str
    Directory where you want to save the trail matrix for this detector. Required only if save is True.

    :param idet: str
    Detector. Required only if save is True.

    :param date: str
    Date. Required only if save is True.
    
    :param FEED_ESO_SKYCALC: bool, 
    Do you want to feed the ESO SkyCalc model to decide which datacube columns are used to compute the PCA eigenvectors?
    
    :param pca_mask_fin: array of bools
    Mask with shape (nwave,) with True for the eavelength channel you do want to include when computing the eigenvectors. 
    
    :param return_trail_matrix: bool, default False
    Do you want to return trail matrix dictionary? 
    
    :return: 
    If return_trail_matrix is True, then return the trail matrix dictionary.
    """
    
    nspec, nwav = datacube.shape[0], datacube.shape[1]

    # Initialize the reprocessed modelcube
    trail_matrix = np.empty((nspec, len(vel_range)))
    
    # Standardise the datacube first
    datacube_standard = np.empty((nspec, nwav))
    datacube_standard = standardise_data(datacube)

    ## colmask_info is info for only one detector, selected outside. 
    colmask_inds = []
    for i in range(len(colmask_info)):
        inds = np.arange(colmask_info[i][0], colmask_info[i][1])
        colmask_inds.extend(inds)
    colmask_inds = np.array(colmask_inds)
    ## Create a column mask for bad pixels, same for all dates and detectors
    colmask = np.zeros(datacube.shape[1], dtype=bool)
    if len(colmask_inds) > 0:
        colmask[colmask_inds] = True
    
    # Run the PCA just once on the datacube
    pca_eigenvectors = np.empty((nspec,N_PCA+1))
    datacube_fit = np.empty((nspec, nwav))
    datacube_detrended = np.empty((nspec, nwav))
    
    # for idet in range(ndet):
    fStd = datacube_standard.copy()
    fStd[:, colmask] = 0
    pca_eigenvectors = get_eigenvectors_via_PCA_Matteo(fStd[:, colmask==False], nc=N_PCA)

    datacube_fit[:, :] = linear_regression(X=pca_eigenvectors, Y=datacube[:, :])
    datacube_detrended[:, :] = datacube/datacube_fit - 1.

    ## Apply the post PCA mask to zero out wavelength channels that were not corrected properly. 
    # Automatically detect these channels by checking if the variance of a channel is larger than sigma times 
    # median variance of all channels, where sigma is the threshold probability below which such a deviation
    # could have occured by random chance. You could use stdev here instead of variance, it will just be less stricter filter in that case. 
    datacube_detrended_post_pca_mask, post_pca_mask =  mask_data_post_pca_per_order(datacube_detrended, maskval = 0., threshold=post_pca_threshold_type) ###  
    # Also zero out the basdcolumns on top of this using badcolmask 
    datacube_detrended_post_pca_mask[:, colmask] = 0
    
    datacube_mean_sub = np.empty((nspec, nwav))
    zeroMask = datacube_detrended_post_pca_mask == 0  ## Get the mask for all the values where the datacube is set to zero.
    # This should have already been done at the stage of removing the PCA linear regression fit from the data.
    datacube_mean_sub = crocut.sub_mask(datacube_detrended_post_pca_mask, zeroMask)
    
    avoid_mask = np.logical_or(post_pca_mask, colmask) 
    
    ########### For plotting that will show the masked out columns more clearly 
    datacube_detrended_post_pca_mask_fp, post_pca_mask_fp =  mask_data_post_pca_per_order(datacube_detrended, maskval = -0.03, threshold=post_pca_threshold_type) ###  
    # Also zero out the basdcolumns on top of this using badcolmask 
    datacube_detrended_post_pca_mask_fp[:, colmask] = -0.03
    
    datacube_mean_sub_fp = np.empty((nspec, nwav))
    zeroMask_fp = datacube_detrended_post_pca_mask_fp == 0  ## Get the mask for all the values where the datacube is set to zero.
    # This should have already been done at the stage of removing the PCA linear regression fit from the data.
    datacube_mean_sub_fp = crocut.sub_mask(datacube_detrended_post_pca_mask_fp, zeroMask_fp)
    
    # model_spl = splrep(model_tell_wavsoln, model_tell)
    model_spl = interpolate.make_interp_spline(model_tell_wavsoln, model_tell)
    
    for ivel in range(len(vel_range)):

        RV = vel_range[ivel] #+ berv[it]
        data_wavsoln_shift = crocut.doppler_shift_wavsoln(wavsoln=data_wavsoln, velocity=-1. * RV)
        ## Shift the telluric model by this RV, do it only once since we are not accounting for the BERV here 
        # model_tell_shift = splev(data_wavsoln_shift, model_spl)
        model_tell_shift = model_spl(data_wavsoln_shift)
        
        ## Zero out the same columns in model as the data
        model_tell_shift[post_pca_mask] = 0.
        model_tell_shift[colmask] = 0.
        model_tell_shift_mean_sub = crocut.sub_mask_1D(model_tell_shift, avoid_mask)
        
        for it in range(nspec):
            ## Cross correlate the model with the data for this exposure and populate the trail matrix for this exposure and velocity 
            # Mean subtract the model

            _, trail_matrix[it, ivel], _= crocut.fast_cross_corr(data=datacube_mean_sub[it, ~avoid_mask], model=model_tell_shift_mean_sub[~avoid_mask])

            ## For only the first case make a plot and save 
            # if ivel == 0 and it == 0:
            #     plt.figure()
            #     plt.plot(data_wavsoln_shift, datacube_mean_sub[it, :], 'k', label = 'Data')
            #     plt.plot(data_wavsoln_shift, model_tell_shift_mean_sub, 'r', label = 'Model')
            #     plt.legend()
            #     plt.savefig(savedir + 'model_data_comparison'+str(N_PCA) + str(date) + '_idet-' + str(idet) + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
            #     plt.close('all')
            
    trail_matrix_sum = np.sum(abs(trail_matrix), axis = 0)
         
    if save:
        trail_matrix_dd = {}
        trail_matrix_dd['trail_matrix'] = trail_matrix
        trail_matrix_dd['datacube_detrended'] = datacube_mean_sub # datacube_detrended
        trail_matrix_dd['datacube_detrended_fp'] = datacube_mean_sub_fp # datacube_detrended for plotting 
        trail_matrix_dd['N_PCA'] = datacube_detrended
        trail_matrix_dd['date'] = date
        trail_matrix_dd['idet'] = idet
        np.save(savedir + 'N_PCA-'+str(N_PCA)+'_trail_matrix_date-' + str(date) + '_idet-' + str(idet) + '.npy', trail_matrix_dd)

        ## Plot the trail matrix 
        fig, axx = plt.subplots(2, 1, figsize=(12, 12), clear = True)
        plt.subplots_adjust(hspace=0.8)

        hnd1 = crocut.subplot_cc_matrix(axis=axx[0],
                                      cc_matrix=trail_matrix,
                                      phases=phases,
                                      velocity_shifts=vel_range,
                                      ### check if this plotting is correct, perhaps you need to plot with respect to shifted (by Kp and bary_RV) Vsys values and not the original Vsys (this would mean a different Vsys array for each row)
                                      title= 'N_PCA = ' + str(N_PCA) + '\n' + 'date: ' + str(date) + ' ; idet: ' + str(idet),
                                      setxlabel=True, plot_type = 'pcolormesh', vminvmax = [-0.25,0.25])
        fig.colorbar(hnd1, ax=axx[0])
        
        axx[1].plot(vel_range, (trail_matrix_sum-np.mean(trail_matrix_sum)) / np.std(trail_matrix_sum), 'k')
        axx[1].set_xlabel('V rest [km/s]')
        axx[1].set_ylabel('CCF')
        axx[1].set_ylim(-5,5)
        
        plt.savefig(savedir + 'N_PCA-'+str(N_PCA)+'_trail_matrix_date-' + str(date) + '_idet-' + str(idet) + '.png', format='png', dpi=300, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
        plt.close('all')
        # Clear the current axes.
        # plt.cla() 
        # # Clear the current figure.
        # plt.clf() 
        # # Closes all the figure windows.
        # plt.close('all')   
        # plt.close(fig)
        # gc.collect()
        
        ## Plot the original and detrended datacube 
        fig, axx = plt.subplots(2, 1, figsize=(12, 12), clear = True)
        plt.subplots_adjust(hspace=0.8)

        hnd1 = subplot_datacube(axis=axx[0], datacube=datacube, phases=phases, wavsoln=data_wavsoln, 
                  title='N_PCA = ' + str(N_PCA) + '\n' + 'date: ' + str(date) + ' ; idet: ' + str(idet), setxlabel=False,
                     vminvmax=None)
        fig.colorbar(hnd1, ax=axx[0])
        
        hnd2 = subplot_datacube(axis=axx[1], datacube=datacube_mean_sub_fp, phases=phases, wavsoln=data_wavsoln, 
                  title='Detrended', setxlabel=True,
                     vminvmax=[-0.05,0.05])
        fig.colorbar(hnd2, ax=axx[1])
        
        
        # axx[1].set_xlabel('V rest [km/s]')
        # axx[1].set_ylabel('CCF')

        plt.savefig(savedir + 'N_PCA-'+str(N_PCA)+'_datacube-' + str(date) + '_idet-' + str(idet) + '.png', format='png', dpi=200, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
        plt.close('all')
        # Clear the current axes.
        # plt.cla() 
        # # Clear the current figure.
        # plt.clf() 
        # # Closes all the figure windows.
        # plt.close('all')   
        # plt.close(fig)
        # gc.collect()
        
    if return_trail_matrix:
        return trail_matrix_dd
    else:
        return


######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
######## ######## ######## PCA Optimization utils ######## ######## ######## ######## ###
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 
######## ######## ######## ######## ######## ######## ######## ######## ######## ######## 

def visualize_runs_histogram(case = None, date = None, det = None, N_PCA_range = None,
                             savefig = True, trail_matrix_dd = None, savedir = None):
    
    for N_PCA in N_PCA_range:
        tm_dd = np.load(trail_matrix_dd[case][date][det][N_PCA], allow_pickle = True).item()

        print(tm_dd.keys())
        trail_matrix = tm_dd['trail_matrix']
        
        tm_dd = np.load(trail_matrix_dd[case][date][det][N_PCA], allow_pickle = True).item()

        print(tm_dd.keys())

        ccf_av = normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))
        # print(np.mean(abs(ccf_av - np.mean(ccf_av))))
        mad = str( round (np.mean(abs(ccf_av)) , 2))
        
        vel = np.arange(-100,100)

        ccf_av_max = str(round(np.max(normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))), 2))
        ccf_av_max_arg = np.argmax(ccf_av)
        ccf_av_at_zero = str(round(ccf_av[100],2))

        # title = 'N_PCA = ' + str(N_PCA) +'\n Max = ' + ccf_av_max + r' $\sigma$ at V = ' + str(vel[ccf_av_max_arg]) + ' km/s' + '\n' + 'CCF at V = 0 : ' + ccf_av_at_zero + r'$\sigma$' + '; mad(S/N) =  ' + mad
        title = 'Date: '+date+'; Det: '+det+'; N_PCA = ' + str(N_PCA)
        
        fig, ax = plt.subplots(1,2,figsize = (8,3), gridspec_kw={'width_ratios':[2,1]})
        
        plt.suptitle(title)
        
        ax[0].plot(vel, ccf_av, color = 'k')
        ax[0].set_ylabel('Normalized Average CCF')
        ax[0].set_xlabel('V [km/s]')
        ax[0].axvline(x = 0., linestyle = 'dashed', color = 'r')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        ax[0].axvline(x = vel[75], linestyle = 'dashed')
        ax[0].axvline(x = vel[125], linestyle = 'dashed')
        ax[0].set_xlim(-100,100)
        ax[0].set_ylim(-3,5)

        ax[1].hist( tm_dd['trail_matrix'][:,75:125].flatten(), histtype = 'step', bins = 50, density = True, alpha = 0.8, color = 'k' )
        ax[1].set_ylabel('Counts')
        ax[1].set_xlabel('CCF')
        # plt.xlabel('V [km/s]')
        # plt.axvline(x = 0., linestyle = 'dashed', color = 'r')
        # plt.axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        ax[1].set_xlim(-0.2,0.2)
        ax[1].set_ylim(0,13)
        plt.savefig(savedir + 'N_PCA_' + str(N_PCA) + 'summary.png', format = 'png', dpi = 200, bbox_inches = 'tight')
        # plt.show()
        plt.close('all')
        
        ### Could also make the same plot as above but with the detrended datacubes instead of the averaged CCF 
        
def visualize_runs_with_hist_trail_matrix(case = None, date = None, det = None, N_PCA_range = None, 
                                          savefig = True, trail_matrix_dd = None, savedir = None, return_summary_dd = True, wav_phase_dd = None):
    
    summary_dd = {}
    
    for N_PCA in N_PCA_range:
        
        summary_dd[N_PCA] = {}
        
        tm_dd = np.load(trail_matrix_dd[case][date][det][N_PCA], allow_pickle = True).item()

        print(tm_dd.keys())
        trail_matrix = tm_dd['trail_matrix']
        datacube_detrended = tm_dd['datacube_detrended_fp']
        
        ccf_av = normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))
        # print(np.mean(abs(ccf_av - np.mean(ccf_av))))
        mad = str( round (np.mean(abs(ccf_av)) , 2))
        
        vel = np.arange(-100,100)

        ccf_av_max = str(round(np.max(normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))), 2))
        ccf_av_max_arg = np.argmax(ccf_av)
        ccf_av_at_zero = str(round(ccf_av[100],2))

        # title = 'N_PCA = ' + str(N_PCA) +'\n Max = ' + ccf_av_max + r' $\sigma$ at V = ' + str(vel[ccf_av_max_arg]) + ' km/s' + '\n' + 'CCF at V = 0 : ' + ccf_av_at_zero + r'$\sigma$' + '; mad(S/N) =  ' + mad
        title = 'Date: '+date+'; Det: '+det+'; N_PCA = ' + str(N_PCA)
        
        # fig, ax = plt.subplots(1,2,figsize = (8,3), gridspec_kw={'width_ratios':[2,1]})
        fig = plt.figure(constrained_layout=True, figsize = (15,9))
        
        gs = fig.add_gridspec(3, 3)
        
        ax_dat = fig.add_subplot(gs[0, :1])
        ax_tm = fig.add_subplot(gs[1, :1])
        ax_ccf = fig.add_subplot(gs[2, :1])
        ax_hist = fig.add_subplot(gs[1, 1])
        
        plt.suptitle(title)
        
        ## Plot the detrended datacube first 
        ax_dat.pcolormesh(wav_phase_dd[date][det]['wavsoln'], wav_phase_dd[date][det]['phases'], datacube_detrended,
                         norm=mpl.colors.Normalize(vmin=-0.025, vmax=0.025), shading='auto' )
        ax_dat.set_ylabel(r'$\phi$')
        ax_dat.set_xlabel('Wavelength [nm]')
        # ax[0].axvline(x = 0., linestyle = 'dotted', color = 'k')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        
        # ax[0].axvline(x = vel[75], linestyle = 'dashed', color = 'k')
        # ax[0].axvline(x = vel[125], linestyle = 'dashed', color = 'k')
        # ax[0].set_xlim(-100,100)
        
        
        ## Plot the trail matrix
        ax_tm.pcolormesh(vel, wav_phase_dd[date][det]['phases'], trail_matrix, 
                         norm=mpl.colors.Normalize(vmin=-0.2, vmax=0.2), shading='auto' )
        ax_tm.set_ylabel(r'$\phi$')
        ax_tm.set_xlabel('Velocity [km/s]')
        ax_tm.axvline(x = 0., linestyle = 'dotted', color = 'k')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        
        ax_tm.axvline(x = vel[75], linestyle = 'dashed', color = 'k')
        ax_tm.axvline(x = vel[125], linestyle = 'dashed', color = 'k')
        ax_tm.set_xlim(-100,100)
        
        ## CCF 
        ax_ccf.plot(vel, ccf_av, color = 'k')
        ax_ccf.set_ylabel('Normalized Average CCF')
        ax_ccf.set_xlabel('V [km/s]')
        ax_ccf.axvline(x = 0., linestyle = 'dashed', color = 'r')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        ax_ccf.axvline(x = vel[75], linestyle = 'dashed')
        ax_ccf.axvline(x = vel[125], linestyle = 'dashed')
        ax_ccf.set_xlim(-100,100)
        ax_ccf.set_ylim(-3,5)
        
        ######## Histogram 
        ax_hist.hist( tm_dd['trail_matrix'][:,75:125].flatten(), histtype = 'step', bins = 50, density = True, alpha = 0.8, color = 'k' )
        ## Mark the quantiles 
        mean, mean_minus, mean_plus = corner.quantile(tm_dd['trail_matrix'][:,75:125].flatten(), [0.5, 0.5-0.341, 0.5+0.341])
        ax_hist.axvline(x = mean_plus, linestyle = 'dotted', color = 'k')
        ax_hist.axvline(x = mean_minus, linestyle = 'dotted', color = 'k')
        ax_hist.axvline(x = mean, linestyle = 'dashed', color = 'k')
        
        title_mean_pm = str(round(mean, 4)) + '$^{+'+ str(round(mean_plus-mean, 4)) +'}$' + '$_{-'+ str(round(mean-mean_minus, 4)) +'}$'
        
        ax_hist.set_title(title_mean_pm)
        ax_hist.set_ylabel('Counts')
        ax_hist.set_xlabel('CCF')
        ax_hist.set_xlim(-0.2,0.2)
        ax_hist.set_ylim(0,13)
        
        if savefig:
            plt.savefig(savedir + 'N_PCA_' + str(N_PCA) + 'summary.png', format = 'png', dpi = 300, bbox_inches = 'tight')
        # plt.show()
        plt.clf()
        plt.close('all')
        ## save in summary dd for later plotting 
        ## For trail matrix : vel, wav_phase_dd[date][det]['phases'], trail_matrix
        ## For CCF : ccf_av
        ## For histogram : take hist of trail_matrix.flatten() ; second axis spliced between 75 and 125 
        ## For quantiles of the histogram : mean, mean_minus, mean_plus = corner.quantile(trail_matrix[:,75:125].flatten(), [0.5, 0.5-0.341, 0.5+0.341])
        
        summary_dd[N_PCA]['vel'] = vel
        summary_dd[N_PCA]['wave_phase_dd'] = wav_phase_dd[date][det]['phases']
        summary_dd[N_PCA]['trail_matrix'] = trail_matrix
        summary_dd[N_PCA]['hist_sig'] = 0.5*( (mean-mean_minus) + (mean_plus-mean) )
        
    if return_summary_dd:
        return summary_dd


######## Function to plot only the detrended datacube, trail matrix, and histogram in 1 x 3 subplots 
def normalize(x):
    norm_x = (x - np.mean(x) ) / np.std(x)
    return norm_x


def make_paper_figure_PCA_optim_demo(case = None, date = None, det = None, N_PCA_range = None, 
                                          savefig = True, trail_matrix_dd = None, savedir = None, wav_phase_dd = None):
    
    summary_dd = {}
    
    for N_PCA in N_PCA_range:
        
        summary_dd[N_PCA] = {}

        tm_dd = np.load(trail_matrix_dd[case][date][det][N_PCA], allow_pickle = True).item()

        print(tm_dd.keys())
        trail_matrix = tm_dd['trail_matrix']

        datacube_detrended = tm_dd['datacube_detrended_fp']
        
        ## Ad hoc detect post pca mask because they are set to 0 and need to plot them as white in the plot
        ## LAter actually save this in the tm_dd when running PCA optim 
        
        # import pdb
        # pdb.set_trace()
        post_pca_mask = np.zeros((datacube_detrended.shape[1],), dtype = bool)
        for kk in range(datacube_detrended.shape[1]):
            if np.all(datacube_detrended[:,kk] < -0.025):
                post_pca_mask[kk] = True
        
        ccf_av = normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))
        # print(np.mean(abs(ccf_av - np.mean(ccf_av))))
        mad = str( round (np.mean(abs(ccf_av)) , 2))
        
        vel = np.arange(-100,100)

        ccf_av_max = str(round(np.max(normalize(np.sum(abs(tm_dd['trail_matrix']), axis = 0))), 2))
        ccf_av_max_arg = np.argmax(ccf_av)
        ccf_av_at_zero = str(round(ccf_av[100],2))

        # title = 'N_PCA = ' + str(N_PCA) +'\n Max = ' + ccf_av_max + r' $\sigma$ at V = ' + str(vel[ccf_av_max_arg]) + ' km/s' + '\n' + 'CCF at V = 0 : ' + ccf_av_at_zero + r'$\sigma$' + '; mad(S/N) =  ' + mad
        title = 'Date: '+date+'; Det: '+str(int(det)+1)+'; N$_{PCA}$ = ' + str(N_PCA)
        
        # fig, ax = plt.subplots(1,2,figsize = (8,3), gridspec_kw={'width_ratios':[2,1]})
        fig = plt.figure(constrained_layout=True, figsize = (18,5))
        
        gs = fig.add_gridspec(1, 3)
        
        ax_dat = fig.add_subplot(gs[0,0])
        ax_tm = fig.add_subplot(gs[0,1])
        # ax_ccf = fig.add_subplot(gs[2, :1])
        ax_hist = fig.add_subplot(gs[0,2])
        
        plt.suptitle(title)
        
        ## Plot the detrended datacube first 
        ax_dat.pcolormesh(wav_phase_dd[date][det]['wavsoln'], wav_phase_dd[date][det]['phases'], datacube_detrended,
                         norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05), shading='auto' )
        ax_dat.set_ylabel(r'$\phi$')
        ax_dat.set_xlabel('Wavelength [nm]')
        ax_dat.set_title('Detrended data')
        
        ## Post PCA Mask : color them all white 
        for ix in range(len(wav_phase_dd[date][det]['wavsoln'])):
            if post_pca_mask[ix] == True:
                ax_dat.axvline(x = wav_phase_dd[date][det]['wavsoln'][ix], ymin = 0, ymax = 1, color = 'w')
        
        
        # ax[0].axvline(x = 0., linestyle = 'dotted', color = 'k')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        
        # ax[0].axvline(x = vel[75], linestyle = 'dashed', color = 'k')
        # ax[0].axvline(x = vel[125], linestyle = 'dashed', color = 'k')
        # ax[0].set_xlim(-100,100)
        
        
        ## Plot the trail matrix
        ax_tm.pcolormesh(vel, wav_phase_dd[date][det]['phases'], trail_matrix, 
                         norm=mpl.colors.Normalize(vmin=-0.15, vmax=0.15), shading='auto' )
        ax_tm.set_ylabel(r'$\phi$')
        ax_tm.set_xlabel('Velocity [km s$^{-1}$]')
        ax_tm.axvline(x = 0., linestyle = 'dotted', color = 'k')
        # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        
        ax_tm.axvline(x = vel[75], linestyle = 'dashed', color = 'k')
        ax_tm.axvline(x = vel[125], linestyle = 'dashed', color = 'k')
        ax_tm.set_xlim(-100,100)
        ax_tm.set_title('CCF')
        ## CCF 
        # ax_ccf.plot(vel, ccf_av, color = 'k')
        # ax_ccf.set_ylabel('Normalized Average CCF')
        # ax_ccf.set_xlabel('V [km/s]')
        # ax_ccf.axvline(x = 0., linestyle = 'dashed', color = 'r')
        # # ax[0].axvline(x = vel[ccf_av_max_arg], linestyle = 'dashed')
        # ax_ccf.axvline(x = vel[75], linestyle = 'dashed')
        # ax_ccf.axvline(x = vel[125], linestyle = 'dashed')
        # ax_ccf.set_xlim(-100,100)
        # ax_ccf.set_ylim(-3,5)
        
        ######## Histogram 
        ax_hist.hist( tm_dd['trail_matrix'][:,75:125].flatten(), histtype = 'step', bins = 50, density = True, alpha = 0.8, color = 'k' )
        ## Mark the quantiles 
        mean, mean_minus, mean_plus = corner.quantile(tm_dd['trail_matrix'][:,75:125].flatten(), [0.5, 0.5-0.341, 0.5+0.341])
        ax_hist.axvline(x = mean_plus, linestyle = 'dotted', color = 'k')
        ax_hist.axvline(x = mean_minus, linestyle = 'dotted', color = 'k')
        ax_hist.axvline(x = mean, linestyle = 'dashed', color = 'k')
        
        # title_mean_pm = str(round(mean, 4)) + '$^{+'+ str(round(mean_plus-mean, 4)) +'}$' + '$_{-'+ str(round(mean-mean_minus, 4)) +'}$'
        # title_mean_pm = 'Date: '+ date + '; Detector #'+ str(det) +'/n'+ 'N$_{PCA}$ = ' + str(N_PCA) + '; '+ '$\sigma$_{tell} = ' + str(round(0.5*( (mean-mean_minus) + (mean_plus-mean) ), 3))
        title_mean_pm = '$\sigma_{tell}$ = ' + str(round(0.5*( (mean-mean_minus) + (mean_plus-mean) ), 3))
        ax_hist.set_title(title_mean_pm)
        ax_hist.set_ylabel('Counts')
        ax_hist.set_xlabel('CCF')
        ax_hist.set_xlim(-0.2,0.2)
        ax_hist.set_ylim(0,13)
        
        if savefig:
            plt.savefig(savedir + date + '_' + det + '_'+ 'N_PCA_' + str(N_PCA) + 'demo_figure.png', format = 'png', dpi = 300, bbox_inches = 'tight')
        # plt.show()
        plt.close('all')
        plt.clf()
        # save in summary dd for later plotting 
        # For trail matrix : vel, wav_phase_dd[date][det]['phases'], trail_matrix
        # For CCF : ccf_av
        # For histogram : take hist of trail_matrix.flatten() ; second axis spliced between 75 and 125 
        # For quantiles of the histogram : mean, mean_minus, mean_plus = corner.quantile(trail_matrix[:,75:125].flatten(), [0.5, 0.5-0.341, 0.5+0.341])
        
        summary_dd[N_PCA]['vel'] = vel
        summary_dd[N_PCA]['wave_phase_dd'] = wav_phase_dd[date][det]['phases']
        summary_dd[N_PCA]['trail_matrix'] = trail_matrix
        summary_dd[N_PCA]['hist_sig'] = 0.5*( (mean-mean_minus) + (mean_plus-mean) )
    # np.save(savedir + date + '_' + det + '_summary.npy', summary_dd)
        
    # if return_summary_dd:
    return summary_dd         


