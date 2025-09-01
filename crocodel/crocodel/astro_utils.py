import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import scipy
from astropy import units as un
from astropy import constants as con
from scipy.interpolate import splev, splrep, interp1d
from astropy.modeling import models
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.convolution import Box1DKernel, Gaussian1DKernel, convolve
from tqdm import tqdm
import glob
import astropy.io.fits as fits
from scipy.special import comb
import scipy.constants as sc

def compute_equivalent_width(spectrum = None, wavsoln = None, line_range = None, continuum_range = None):
    """Compute Equivalent Width of a spectral line.

    :param spectrum: 1D rray of flux values, defaults to None
    :type spectrum: array_like
    :param wavsoln: 1D array of wavelength solution in nm, defaults to None
    :type wavsoln: array_like
    :param line_range: start and stop wavelength values marking the spectral line, defaults to None
    :type line_range: array_like
    :param continuum_range: start and stop wavelength values marking the continuum, defaults to None
    :type continuum_range: array_like
    :return: Scalar value of the equivalent width of the line.
    :rtype: float64
    """
    
    line_range_inds = [np.argmin(abs(line_range[i]-wavsoln)) for i in range(len(line_range))]
    continuum_range_inds = [np.argmin(abs(continuum_range[i]-wavsoln)) for i in range(len(continuum_range))]
    
    EW = np.sum(spectrum[line_range_inds[0]:line_range_inds[1]])/np.median(spectrum[continuum_range_inds[0]:continuum_range_inds[1]])
    return EW

def doppler_shift_wavsoln(wavsoln=None, velocity=None):
    
    """This function applies a Doppler shift to a 1D array of wavelengths.
     wav_obs = wav_orig (1. + velocity/c) where if velocity is positive it corresponds to a redshift
     (i.e. source moving away from you, so wavelength solution gets shifted towards positive direction) and vice versa
    for a positive velocity - blueshift.
    
    :param wavsoln: 1D array of wavelength solution (in nm)
    :type wavsoln: array_like

    :param velocity: Float value of the velocity of the source, in km/s. Note that the astropy value of speed of light (c) is in m/s.
    :type velocity: float64 
    
    :return: Doppler shifted wavelength solution.
    :rtype: array_like
    
    """
    wavsoln_doppler = wavsoln * (1. + (1000. * velocity) / con.c.value)
    return wavsoln_doppler

def BB_flux(temperature=None, wavelength=None):
    """
    Calculate the Blackbody flux in the SI units. Should have the units as J s-1 m-2 m-1 sr-1
    (same as the planet flux intensity Ip from the model from Matteo/Remco).
    
    :param wavelength: array_like
    1D array of wavelengths, in Angstroms.
    :type wavelength: array_like
    
    :param temperature: Equilibrium temperature of the star or the object, in K.
    :type temperature: float64

    :return: wavelength array in Angstroms and array of BB flux in SI units.
    :rtype: array_like
    """
    BB_scale = 1 * un.J / (un.m * un.m * un.s * un.AA * un.sr)

    BB = models.BlackBody(temperature=temperature * un.K, scale=BB_scale)
    waverange = wavelength * un.AA

    BB_flux_val = BB(waverange)

    return waverange, BB_flux_val

def planck_func(lam, T):
        """Compute the Planck function.
        """
        #### lam should be in m
        lam_5 = lam*lam*lam*lam*lam
        vel_c_SI = 299792458.0
        Bs = (2.0*sc.h*vel_c_SI*vel_c_SI)/(lam_5*(np.expm1((sc.h*vel_c_SI)/(lam*sc.k*T))))

        return Bs### *np.pi*Rs*Rs*6.957e8*6.957e8
    
def compute_FpFs(planet_temp = None, star_temp = None, wavelength = None, rprs = None):
    """
    Compute the simulated blackbody emission for a planet assuming only blackbody radiation from both the star and the planet.
    :param planet_temp: Equilibrium/Dayside Temperature of the planet, in Kelvin.
    :type planet_temp: float
    
    :param star_temp: Effective temperature of the star, in Kelvin.
    :type star_temp: float
    

    :param wavelength: 1D array of wavelength in which you want to compute the emission spectrum of the planet in nm.
    :type wavelength: array_like
    
    :param rprs: float
    Planet to star radius ratio.

    :return: Arrays of Wavelength in nm and FpFs.
    :rtype: array_like 
    """

    # _, BB_model_planet = BB_flux(temperature=planet_temp, wavelength=wavelength)
    # _, BB_model_star = BB_flux(temperature=star_temp, wavelength=wavelength)
    
    BB_model_planet = planck_func(wavelength*1e-9, planet_temp)
    BB_model_star = planck_func(wavelength*1e-9, star_temp)

    FpFs = (rprs ** 2.) * (BB_model_planet / BB_model_star)

    return wavelength, FpFs

def compute_TSM(scale_factor=None, Rp=None, Mp=None, Rs=None, Teq=None, Jmag=None):
    """
    Compute the TSM given the parameters.
    :param scale_factor: From Table 1 in https://arxiv.org/pdf/1805.03671.pdf :
    Rp < 1.5 Rearth -> 0.19 ;
    1.5 Rearth < Rp < 2.75 Rearth -> 1.26 ;
    2.75 Rearth < Rp < 4 Rearth -> 1.28 ;
    4 Rearth < Rp < 10 Rearth -> 1.15 ;
    :type param: float64

    :param Rp: Radius of the planet in earth radii.
    :type Rp: astropy.units
     
    :param Mp: Mass of the planet in earth masses.
    :type Mp: astropy.units
    
    :param Rs: Radius of the star in solar radii.
    :type Rs: astropy.units
    
    :param Teq: Equilibrium temperature of the planet.
    :type Teq: astropy.units
    
    :param Jmag: J band magnitude.
    :type Jmag: float64
    
    :return: Transmission Spectrum Metric as defined in Newton et al. 2018 (https://arxiv.org/pdf/1805.03671.pdf).
    :rtype: float64
    
    """
    
    TSM = (scale_factor * Rp ** 3. * Teq * 10 ** (-Jmag / 5.)) / (Mp * Rs ** 2.)
    return TSM


def compute_SH_signal(Teq=None, Mp=None, Rp=None, Rs=None, mu=None):
    """
    Calculate the SH (in metres) and the signal due to 1 scale height of the atmosphere. All quantities below should be
    taken as an input along with their astropy units.
    
    :param Teq: Equilibrium temperature of the planet.
    :type Teq: astropy.units
    
    :param Mp: Mass of the planet.
    :type Mp: astropy.units
    
    :param Rp: Radius of the planet.
    :type Rp: astropy.units
    
    :param Rs: Radius of the star.
    :type Rs: astropy.units
    
    :param mu: Mean Molecular Weight of the atmosphere (2.33 for H-He dominated atmosphere)
    :type mu: float64
    
    :return: Signal due to 1 scale height (in ppm) and the scale height in metres.
    :rtype: float64
    
    """
    SH = (con.k_B * Teq) / (2.33 * con.u * ((con.G * Mp) / (Rp ** 2.)))

    del_SH = 2 * (SH / Rp) * ((Rp / Rs)) ** 2.

    return 1e6*del_SH.si.value, SH.si  # del_SH in ppm, SH in metres

def compute_SH_signal_relative(Teq=None, Mp=None, Rp=None, Rs=None, mu=None, del_TD_val = None, TD_ref = None):
    """
    Calculate the relative change in scale height for given relative change in transit depth.
    
    :param Teq: Equilibrium temperature of the planet.
    :type Teq: astropy.units
    
    :param Mp: Mass of the planet.
    :type Mp: astropy.units
    
    :param Rp: Radius of the planet.
    :type Rp: astropy.units
    
    :param Rs: Radius of the star.
    :type Rs: astropy.units
    
    :param mu: Mean Molecular Weight of the atmosphere (2.33 for H-He dominated atmosphere)
    :type mu: float64
    
    :param del_TD_val: Given change in transit depth.
    :type del_TD_val: float64
    
    :param TD_ref: Reference transit depth value.
    :type del_TD_val: float64
    
    :return: Change in scale height due to given change in transit depth.
    :rtype: float64

    
    """
    SH = (con.k_B * Teq) / (2.33 * con.u * ((con.G * Mp) / (Rp ** 2.)))

    del_SH =  (del_TD_val * Rp)/(2.*TD_ref) ## del_TD_val should be in same units as TD_ref 
    
    del_SH_rel = del_SH/SH

    return del_SH_rel.si.value 

def compute_log_g(Mp=None, Rp=None, out_unit = 'cgs'):
    """
    Calculate the log_g of a planet in cgs.
    
    
    
    """
    gravity =  ((con.G * Mp) / (Rp ** 2.))
    print(gravity.si)
    if out_unit == 'cgs':
        return np.log10(gravity.cgs.value)
    elif out_unit == 'si':
        return np.log10(gravity.si.value)

def get_BERV(time_array = None, time_format = None, ra_dec = None, obs_location = None):
    """
    Compute the Barycentric Earth Radial Velocity for given time stamps and for
    observations taken from a given observatory.
    :param time_array: Array of time stamps in BJD.
    :param time_format: Format of the time ('mjd', 'jd', etc. ; see Time module of astropy for accepted keywords).
    :param ra_dec: Coordinates of the target, in ICRS (hourangle, degrees) format.
    :param obs_location: latitude, longitude, and altitude of the observatory location.
    :return: BERV values (in km/s).
    """

    location = EarthLocation.from_geodetic(lat=obs_location[0] * un.deg, lon= obs_location[1]* un.deg,
                                       height= obs_location[2] * un.m)

    time_array = Time(time_array, format=time_format, scale='utc')

    # location = EarthLocation.of_site(obs_location)

    sc = SkyCoord(ra_dec[0], ra_dec[1], unit=(un.hourangle, un.deg), frame='icrs')

    berv_list = []

    for t in time_array:
        berv = sc.radial_velocity_correction(kind="barycentric",
                                                     obstime=t,
                                                     location=location)  ### by default barycentric
        berv_list.append(berv)

    berv_array = np.array([j.to((un.km / un.s)).value for j in berv_list])

    return berv_array

def guillot_TP(pressure_levels = None, T_int = None, T_eq = None, gamma = None, gravity = None, 
               kappa_IR = None, f_global = None, T_irr = None):
    """Compute a parametrized TP profile from Guillot (2010) Equation 29. 

    :param pressure_levels: _description_, defaults to None
    :type pressure_levels: _type_, optional
    :param T_int: _description_, defaults to None
    :type T_int: _type_, optional
    :param T_eq: _description_, defaults to None
    :type T_eq: _type_, optional
    :param gamma: _description_, defaults to None
    :type gamma: _type_, optional
    :param gravity: _description_, defaults to None
    :type gravity: _type_, optional
    """
    # f_global = 1 at the substellar point,
    # f_global = 1/2 for a day-side average and f = 1/4 for an averaging over the whole planetary surface (Burrows et al. 2003)
    tau = (pressure_levels * 1e6 * kappa_IR ) / gravity
    
    if T_irr is None and T_eq is not None:
        T_irr = T_eq*np.sqrt(2.)
    
    term1 = 0.25 * 3. * (T_int**4.) * (2./3. + tau)
    term2_1 = 1./(np.sqrt(3) * gamma) 
    term2_2 = ( gamma/np.sqrt(3.) -  1./(gamma * np.sqrt(3.)) ) * np.exp(-1. * gamma * tau * np.sqrt(3.)) 
    
    term2 = f_global * 0.25 * 3. * (T_irr ** 4.) * (2./3. + term2_1 + term2_2 )
    
    temperature_levels = (term1 + term2) ** 0.25
    

    return temperature_levels

def madhusudhan_seager_TP(pressure_levels = None, log_Pset = 0.1, Tset = None, alpha1 = None, alpha2 = None, log_P1 = None, log_P2 = None, log_P3 = None, beta = 0.5):
    """
    Compute the Madhusudhan and Seager 2009 parametrized TP profile. 
    log_P1 < log_P2 < log_P3.
    
    
    """
    
    P1, P2, P3 = 1e5*10**log_P1, 1e5*10**log_P2, 1e5*10**log_P3 
    assert(P1 < P3)
    
    ########### convert pressure_levels to Pa
    pressure_levels_Pa = pressure_levels * 1e5
    i_set = np.argmin(np.abs(pressure_levels_Pa - np.power(10.0,log_Pset+5.0)))
    
    assert (i_set>0 and i_set<len(pressure_levels_Pa))
    P_set_i = pressure_levels_Pa[i_set]
    P_top = np.min(pressure_levels_Pa) #Pa
    if (P_set_i >= P3):   # If the temperature parameter in layer 3
        T_3 = Tset
        T_2 = T_3 - ((1.0/alpha2)*(np.log(P3/P2)))**2
        T_1 = T_2 + ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_0 = T_1 - ((1.0/alpha1)*(np.log(P1/(P_top))))**2
    elif (P_set_i >= P1):   # If the temperature parameter in layer 2
        T_2 = Tset - ((1.0/alpha2)*(np.log(P_set_i/P2)))**2
        T_1 = T_2 + ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_3 = T_2 + ((1.0/alpha2)*(np.log(P3/P2)))**2
        T_0 = T_1 - ((1.0/alpha1)*(np.log(P1/(P_top))))**2
    elif (P_set_i < P1):  # If the temperature parameter in layer 1
        T_0 = Tset - ((1.0/alpha1)*(np.log(P_set_i/(P_top))))**2
        T_1 = T_0 + ((1.0/alpha1)*(np.log(P1/(P_top))))**2
        T_2 = T_1 - ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_3 = T_2 + ((1.0/alpha2)*(np.log(P3/P2)))**2
        
    temperature_levels = np.zeros(len(pressure_levels_Pa))
    temperature_levels[pressure_levels_Pa < P1] = T_0 + np.power((1.0/alpha1)*np.log(pressure_levels_Pa[pressure_levels_Pa < P1]/(P_top)),1.0/beta)#run_params.P_min*1.0e5
    temperature_levels[(pressure_levels_Pa <=P3) & (pressure_levels_Pa >= P1)] = T_2 + np.power((1.0/alpha2)*np.log(pressure_levels_Pa[(pressure_levels_Pa <=P3) & (pressure_levels_Pa >= P1)]/P2),1.0/beta)
    temperature_levels[pressure_levels_Pa > P3]  = T_3
    
    ##### Make sure the temperatures are within 400 K and 4000 K 
    assert(np.min(temperature_levels) > 300)
    assert(np.max(temperature_levels) < 5000)
    
    return temperature_levels

### Above function, but without assert statements 
def madhusudhan_seager_TP_pure_function(pressure_levels = None, log_Pset = 0.1, Tset = None, alpha1 = None, alpha2 = None, log_P1 = None, log_P2 = None, log_P3 = None, beta = 0.5):
    
    ### log_Pset, log_P1, log_P2, log_P3 should be in log_bars.
    ### pressure_levels should be in bars. 
    
    P1, P2, P3 = 1e5*10**log_P1, 1e5*10**log_P2, 1e5*10**log_P3 
    
    ########### convert pressure_levels to Pa
    pressure_levels_Pa = pressure_levels * 1e5
    i_set = np.argmin(np.abs(pressure_levels_Pa - np.power(10.0,log_Pset+5.0)))
    print(i_set)
    assert (i_set>0 and i_set<len(pressure_levels_Pa))
    P_set_i = pressure_levels_Pa[i_set]
    P_top = np.min(pressure_levels_Pa) #Pa
    if (P_set_i >= P3):   # If the temperature parameter in layer 3
        T_3 = Tset
        T_2 = T_3 - ((1.0/alpha2)*(np.log(P3/P2)))**2
        T_1 = T_2 + ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_0 = T_1 - ((1.0/alpha1)*(np.log(P1/(P_top))))**2
    elif (P_set_i >= P1):   # If the temperature parameter in layer 2
        T_2 = Tset - ((1.0/alpha2)*(np.log(P_set_i/P2)))**2
        T_1 = T_2 + ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_3 = T_2 + ((1.0/alpha2)*(np.log(P3/P2)))**2
        T_0 = T_1 - ((1.0/alpha1)*(np.log(P1/(P_top))))**2
    elif (P_set_i < P1):  # If the temperature parameter in layer 1
        T_0 = Tset - ((1.0/alpha1)*(np.log(P_set_i/(P_top))))**2
        T_1 = T_0 + ((1.0/alpha1)*(np.log(P1/(P_top))))**2
        T_2 = T_1 - ((1.0/alpha2)*(np.log(P1/P2)))**2
        T_3 = T_2 + ((1.0/alpha2)*(np.log(P3/P2)))**2
        
    temperature_levels = np.zeros(len(pressure_levels_Pa))
    temperature_levels[pressure_levels_Pa < P1] = T_0 + np.power((1.0/alpha1)*np.log(pressure_levels_Pa[pressure_levels_Pa < P1]/(P_top)),1.0/beta)#run_params.P_min*1.0e5
    temperature_levels[(pressure_levels_Pa <=P3) & (pressure_levels_Pa >= P1)] = T_2 + np.power((1.0/alpha2)*np.log(pressure_levels_Pa[(pressure_levels_Pa <=P3) & (pressure_levels_Pa >= P1)]/P2),1.0/beta)
    temperature_levels[pressure_levels_Pa > P3]  = T_3
        
    return temperature_levels

def TP_4_nodes_B_Spline(pressure_levels = None, log_P0 = None, log_P3 = None, 
                        log_P1 = None, log_P2 = None,
                        T0 = None, T1 = None, T2 = None, T3 = None):
    
    ### All pressure inputs should be in bars 
    ### In order of increasing pressures (or going from top to down in altitude) : P3,P2,P1,P0 
    
    ##If P0 and P3 are None, set them to the max and minimum of the pressure_levels 
    if log_P0 is None and log_P3 is None:
        P0, P3 = np.max(pressure_levels * 1e5), np.min(pressure_levels * 1e5)
    else:
        P0, P3 = 1e5*10**log_P0, 1e5*10**log_P3
    P1, P2 = 1e5*10**log_P1, 1e5*10**log_P2 ## convert to Pascals 
    print(P0, P1, P2, P3)
    assert(P1 > P2)
    
    ########### convert pressure_levels to Pa
    pressure_levels_Pa = pressure_levels * 1e5    
    # # PT_0, PT_1, PT_2, PT_3 = [P0, T0],[P1, T1],[P2, T2],[P3, T3]
    print(np.array([P3,P2,P1,P0]))
    print(np.array([T3,T2,T1,T0]))
    # PT_spl = splrep([P0, P1, P2, P3], [T0,T1,T2,T3])
    PT_spl = splrep([P3,P2,P1,P0], [T3,T2,T1,T0])
    temperature_levels = splev(pressure_levels_Pa, PT_spl) 
    
    assert(np.min(temperature_levels) > 400)
    assert(np.max(temperature_levels) < 4000)
    
    return temperature_levels

################################################################################
################################################################################
################ Bezier TP Functions from Peter Smith ############## ################
################################################################################
################################################################################

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def PTbez(Ps, Ts, logParr):
    """
    Simple PT profile bezier interpolating between pressure nodes.
    Inputs: Ps - pressure nodes
            Ts - Temperatures at each node
            logParr - array of all logPressures
    Output: T - temperature array
    """
    nlev = logParr.shape[0]

    pts = np.column_stack((Ts, Ps))
    t, p = bezier_curve(pts, nlev)
    
    spl = splrep(p[::-1],t[::-1])
    T = splev(logParr, spl)
    
    assert(np.min(T) > 400)
    assert(np.max(T) < 4000)
    
    return T


################################################################################
################################################################################
################################################################################

def resample_spectrum(wavsoln_new=None, wavsoln_orig=None, spec_flux_orig=None):
    """
    This function computes an interpolated 1D spectrum for a new wavelength solution, given the original 1D spectrum and
    wavelength solution by performing a BSpline interpolation.
    :param wavsoln_new: array_like
    New wavelength solution.

    :param wavsoln_orig: array_like
    Original wavelength solution.

    :param spec_flux_orig: array_like
    Original 1D spectrum.

    :return: Interpolated 1D spectrum for the new wavelength solution.
    """
    # First create the BSpline interpolation for the wavsoln_orig and spec_flux_orig.
    spl = splrep(wavsoln_orig, spec_flux_orig)
    spec_flux_new = splev(wavsoln_new, spl)

    return spec_flux_new

def convolve_to_inst_resolution(delwav_by_wav_inst = 1/100000, model_wav = None, model_spec = None):

    delwav_by_wav_model = np.diff(model_wav)/model_wav[1:]
    FWHM = np.mean(delwav_by_wav_inst/delwav_by_wav_model)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
    model_spec_conv = convolve(model_spec, Gaussian1DKernel(stddev=sig), boundary='extend')
    return model_spec_conv

def broaden_spec(velocity = None, model_wav = None, model_spec = None):
    ### velocity in km/s
    delwav_by_wav_orig= np.diff(model_wav)/model_wav[1:]
    delwav_by_wav_broad = (1000. * velocity) / con.c.value
    
    FWHM = np.mean(delwav_by_wav_broad/delwav_by_wav_orig)
    sig = FWHM / (2. * np.sqrt(2. * np.log(2.) ) )           
    model_spec_conv = convolve(model_spec, Gaussian1DKernel(stddev=sig), boundary='extend')
    return model_spec_conv

def binning_pandexo_output(x,y, yerr, R=None, inst = 'NIRSpec'):
    h = np.mean(x)/R
    if inst == 'NIRSpec':
        wvls1 = np.arange(x[0], 3.72-h/2, h)
        wvls2 = np.arange(3.83+h/2, x[-1], h)
        wvls = np.concatenate((wvls1, wvls2))
    elif inst == 'NIRISS':
        wvls = np.arange(0.6, 2.8,h)
    elif inst == 'MIRI-LRS':
        wvls = np.arange(5., 12.1,h)
        
    nbins = len(wvls)
    y_binned = np.zeros(nbins)
    y_err_binned = np.zeros(nbins)
    y_binned_array = {}
    y_binned_array_squared = {}
    y_binned_array_error = {}
    x_binned = np.zeros(nbins)
    
    for j in range(nbins):
        y_binned_array['bin_{index}'.format(index=j)]=[]
        y_binned_array_squared['bin_{index}'.format(index=j)]=[]
        y_binned_array_error['bin_{index}'.format(index=j)]=[]
    
    for i in range(len(y)):
        for j in range(nbins):
            if (x[i] > (wvls[j] - h/2) and x[i] <= (h/2+wvls[j])):
                #print(j*h, (j+1)*h)
                y_binned_array['bin_{index}'.format(index=j)].append(y[i])
                y_binned_array_squared['bin_{index}'.format(index=j)].append(yerr[i]**2)
                y_binned_array_error['bin_{index}'.format(index=j)].append(yerr[i])
    for j in range(nbins):
        y_binned[j] = np.average(y_binned_array['bin_{index}'.format(index=j)])
        y_err_binned[j] = np.sqrt(np.sum(y_binned_array_squared['bin_{index}'.format(index=j)]))/len(y_binned_array['bin_{index}'.format(index=j)])
    
    return wvls, y_binned, y_err_binned, h

def binning_model_to_resolution(x,y, R=None, inst = 'NIRSpec'):
    h = np.mean(x)/R
    if inst == 'NIRSpec':
        wvls1 = np.arange(x[0], 3.72-h/2, h)
        wvls2 = np.arange(3.83+h/2, x[-1], h)
        wvls = np.concatenate((wvls1, wvls2))
    elif inst == 'NIRISS':
        wvls = np.arange(0.6, 2.8,h)
    elif inst == 'MIRI-LRS':
        wvls = np.arange(5., 12.1,h)
        
    nbins = len(wvls)
    y_binned = np.zeros(nbins)
    # y_err_binned = np.zeros(nbins)
    y_binned_array = {}
    # y_binned_array_squared = {}
    # y_binned_array_error = {}
    x_binned = np.zeros(nbins)
    
    for j in range(nbins):
        y_binned_array['bin_{index}'.format(index=j)]=[]
        # y_binned_array_squared['bin_{index}'.format(index=j)]=[]
        # y_binned_array_error['bin_{index}'.format(index=j)]=[]
    
    for i in range(len(y)):
        for j in range(nbins):
            if (x[i] > (wvls[j] - h/2) and x[i] <= (h/2+wvls[j])):
                #print(j*h, (j+1)*h)
                y_binned_array['bin_{index}'.format(index=j)].append(y[i])
                # y_binned_array_squared['bin_{index}'.format(index=j)].append(yerr[i]**2)
                # y_binned_array_error['bin_{index}'.format(index=j)].append(yerr[i])
    for j in range(nbins):
        y_binned[j] = np.average(y_binned_array['bin_{index}'.format(index=j)])
        # y_err_binned[j] = np.sqrt(np.sum(y_binned_array_squared['bin_{index}'.format(index=j)]))/len(y_binned_array['bin_{index}'.format(index=j)])
    
    return wvls, y_binned, h


def generate_distinct_colors(num_colors):
    # Get a list of distinct colors from a matplotlib colormap
    colormap = plt.cm.get_cmap('tab10', num_colors)
    colors = [colormap(i) for i in range(num_colors)]
    
    # Convert RGB tuples to hexadecimal color codes
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
    
    return hex_colors

