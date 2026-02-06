import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from functools import partial 
# from splinex import BSpline
from jax.numpy import interp
import jax
import numpy as np
from jax import config
from jax import lax

config.update("jax_enable_x64", True)

@jit
def get_R(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    # breakpoint()
    R = (1. / data.shape[0]) * jnp.dot(data, model)  ## R in Brogi and Line
    return R

@jit
def get_C(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    data = data # - jnp.mean(data)
    model = model - jnp.mean(model)
    R = get_R(data, model)
    C = R / jnp.sqrt(jnp.var(data) * jnp.var(model))  ## C in Brogi and Line
    return C

@jit
def get_logL(data: ArrayLike, model: ArrayLike) -> ArrayLike:
    """
    """
    data = data # - jnp.mean(data)
    model = model - jnp.mean(model)
    R = get_R(data, model)
    logL = (-data.shape[0]/2) * jnp.log(jnp.var(data) + jnp.var(model) - 2.*R)
    return logL

@jit
def doppler_shift_wavsoln(velocity: float, wavsoln: ArrayLike) -> ArrayLike:
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

@jit
def compute_RV(Kp: float, Vsys: float, phases: ArrayLike, berv: ArrayLike) -> ArrayLike:
    return Kp * jnp.sin(2. * jnp.pi * phases) + Vsys + berv

@jit
def compute_RV_eccentric(Kp: float, Vsys: float, phases: ArrayLike, berv: ArrayLike, wp: float, ecc: float) -> ArrayLike:
    ### phases are f values 
    return Kp * (np.cos(phases+wp) + ecc*np.cos(wp)) + Vsys + berv


@jit
def doppler_shift_modelcube(modelcube: ArrayLike, RV: ArrayLike, model_wavsoln: ArrayLike, data_wavsoln: ArrayLike) -> ArrayLike:
    def doppler_shift_model1D(model_1D, RV_val, model_wavsoln, data_wavsoln):
        data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
        model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_1D)
        return model_shifted
    return jax.vmap(doppler_shift_model1D, in_axes = (0,0,None,None))(modelcube, RV, model_wavsoln, data_wavsoln)

@jit
def doppler_shift_model_single(model_single, RV_val, model_wavsoln, data_wavsoln) -> ArrayLike:
    data_wavsoln_shifted = doppler_shift_wavsoln(-RV_val, data_wavsoln)
    model_shifted = interp(data_wavsoln_shifted, model_wavsoln, model_single)
    return model_shifted


### Emission specific functions
@jit
def logL_per_KpVsys_emission(Kp, Vsys, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
    RV_p = compute_RV(Kp, Vsys, phases, berv)
    RV_s = compute_RV(0, Vsys, phases, berv)
    
    modelcube_shifted_Fp = doppler_shift_modelcube(modelcube_Fp, RV_p, model_wavsoln, data_wavsoln)
    modelcube_shifted_Fs = doppler_shift_modelcube(modelcube_Fs, RV_s, model_wavsoln, data_wavsoln)
    modelcube_shifted = modelcube_shifted_Fp/modelcube_shifted_Fs
    
    return jnp.sum(jax.vmap(get_logL, in_axes=(0, 0))(datacube, modelcube_shifted))

@jit
def compute_logL_map_per_order_emission(datacube: ArrayLike, modelcube_Fp: ArrayLike, modelcube_Fs: ArrayLike,
                               Kp_range: ArrayLike, 
                           model_wavsoln: ArrayLike, data_wavsoln: ArrayLike,
                           Vsys_range: ArrayLike, phases: ArrayLike, berv: ArrayLike) -> ArrayLike:
    
    def vectorize_1D_row(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv):
        # jax.debug.print("Value of Kp_row: {Kp_row}", Kp_row = Kp_row)
        # jax.debug.print("Value of Vsys_row: {Vsys_row}", Vsys_row = Vsys_row)
        return jax.vmap(logL_per_KpVsys_emission, in_axes=(0, 0, None, None, None, None, None, None, None))(Kp_row, Vsys_row, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)

    Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range, indexing='ij')

    vectorized_grid_func = jax.vmap(vectorize_1D_row, in_axes=(0, 0, None, None,None, None, None, None, None))

    return vectorized_grid_func(Kp_grid, Vsys_grid, datacube, modelcube_Fp, modelcube_Fs, model_wavsoln, data_wavsoln, phases, berv)


### Transmission specific functions
@jit
def logL_per_KpVsys_transmission(Kp, Vsys, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc):
    RV_p = compute_RV_eccentric(Kp, Vsys, phases, berv, wp, ecc)
    
    modelcube_shifted_RpRs = doppler_shift_modelcube(modelcube_RpRs, RV_p, model_wavsoln, data_wavsoln)
    
    return jnp.sum(jax.vmap(get_logL, in_axes=(0, 0))(datacube, modelcube_shifted_RpRs))

@jit
def CCF_per_KpVsys_transmission(Kp, Vsys, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc):
    RV_p = compute_RV_eccentric(Kp, Vsys, phases, berv, wp, ecc)
    
    modelcube_shifted_RpRs = doppler_shift_modelcube(modelcube_RpRs, RV_p, model_wavsoln, data_wavsoln)
    
    return jnp.sum(jax.vmap(get_C, in_axes=(0, 0))(datacube, modelcube_shifted_RpRs))


@jit
def logL_trail_per_RV_transmission(RV, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask):
    
    def impl(RV_value, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask):
        nspec = modelcube_RpRs.shape[0]
        RV_p = jnp.tile(RV_value, (nspec,)) ## RV must be the same values for all time points for obtaining a trail matrix.    
        modelcube_shifted_RpRs = doppler_shift_modelcube(modelcube_RpRs, RV_p, model_wavsoln, data_wavsoln)
        modelcube_shifted_RpRs_masked = jnp.where(avoid_mask[None,:], modelcube_shifted_RpRs,0.0)
        return jax.vmap(get_logL, in_axes=(0, 0))(datacube, modelcube_shifted_RpRs_masked)
    
    ### Continue here; make sure that the RV values are taken in the correct shape to return the trail matrix. 
    return jax.vmap(impl, in_axes=(0, None, None, None, None, None))(RV, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask)

@jit
def CCF_trail_per_RV_transmission(RV, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask):
    
    def impl(RV_value, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask):
        nspec = modelcube_RpRs.shape[0]
        RV_p = jnp.tile(RV_value, (nspec,)) ## RV must be the same values for all time points for obtaining a trail matrix.
        modelcube_shifted_RpRs = doppler_shift_modelcube(modelcube_RpRs, RV_p, model_wavsoln, data_wavsoln)
        modelcube_shifted_RpRs_masked = jnp.where(avoid_mask[None,:], modelcube_shifted_RpRs,0.0)
        return jax.vmap(get_C, in_axes=(0, 0))(datacube, modelcube_shifted_RpRs_masked)

    return jax.vmap(impl, in_axes=(0, None, None, None, None, None))(RV, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, avoid_mask)


@jit
def compute_logL_map_per_order_transmission(datacube: ArrayLike, modelcube_RpRs: ArrayLike,
                               Kp_range: ArrayLike, 
                           model_wavsoln: ArrayLike, data_wavsoln: ArrayLike,
                           Vsys_range: ArrayLike, phases: ArrayLike, berv: ArrayLike, wp: float, ecc: float) -> ArrayLike:
    
    def vectorize_1D_row(Kp_row, Vsys_row, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc):
        # jax.debug.print("Value of Kp_row: {Kp_row}", Kp_row = Kp_row)
        # jax.debug.print("Value of Vsys_row: {Vsys_row}", Vsys_row = Vsys_row)
        return jax.vmap(logL_per_KpVsys_transmission, in_axes=(0, 0, None, None, None, None, None, None, None, None))(Kp_row, Vsys_row, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc)

    Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range, indexing='ij')
    
    vectorized_grid_func = jax.vmap(vectorize_1D_row, in_axes=(0, 0, None, None,None, None, None, None, None, None))

    return vectorized_grid_func(Kp_grid, Vsys_grid, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc)


@jit
def compute_CCF_map_per_order_transmission(datacube: ArrayLike, modelcube_RpRs: ArrayLike,
                               Kp_range: ArrayLike, 
                           model_wavsoln: ArrayLike, data_wavsoln: ArrayLike,
                           Vsys_range: ArrayLike, phases: ArrayLike, berv: ArrayLike, wp: float, ecc: float) -> ArrayLike:
    
    def vectorize_1D_row(Kp_row, Vsys_row, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc):
        # jax.debug.print("Value of Kp_row: {Kp_row}", Kp_row = Kp_row)
        # jax.debug.print("Value of Vsys_row: {Vsys_row}", Vsys_row = Vsys_row)
        return jax.vmap(CCF_per_KpVsys_transmission, in_axes=(0, 0, None, None, None, None, None, None, None, None))(Kp_row, Vsys_row, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc)

    Kp_grid, Vsys_grid = jnp.meshgrid(Kp_range, Vsys_range, indexing='ij')
    
    vectorized_grid_func = jax.vmap(vectorize_1D_row, in_axes=(0, 0, None, None,None, None, None, None, None, None))

    return vectorized_grid_func(Kp_grid, Vsys_grid, datacube, modelcube_RpRs, model_wavsoln, data_wavsoln, phases, berv, wp, ecc)