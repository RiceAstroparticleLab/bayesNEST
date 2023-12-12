from functools import partial
import numpy as np
import scipy.stats as spystats
import corner
import arviz as az
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# JAX imports
import jax
from jax import config, jit, lax, random
import jax.numpy as jnp
from jax.scipy import stats
from jax.scipy.special import ndtri, erf, gammaln, ndtr

# Numpyro and related imports
import numpyro
from numpyro.infer.reparam import LocScaleReparam
from numpyro import infer
import numpyro.distributions as ndist

# Tensorflow Probability (TFP) imports
from tensorflow_probability.substrates import jax as tfp

from .distributions import *

# Global vars
sqrt2 = np.sqrt(2)
inv_sqrt2_PI = 1/np.sqrt(2*jnp.pi)

@jit
def fano_ER(mean_N_q, E_field, density):
    """
    Calculates the a Fano-like factor to determine the variance of quanta generation.

    Parameters
    ----------
    mean_N_q : scalar or array
        Mean number of quanta produced per event.
    E_field : scalar or array
        Electric field strength in the detector.
    density : scalar or array
        Density of the detector material.

    Returns
    -------
    Fano factor : scalar or array
        The calculated Fano factor based on the input parameters.
    
    """

    return (0.13 - 0.030*density - 0.0057*density**2 + 0.0016*density**3 + 0.0015*jnp.sqrt(mean_N_q*E_field))

@partial(jax.jit, static_argnums=(1,2,3,4))
def workfunction(density, molar_mass=131.293, avo=6.0221409e+23, atom_num=54.0, old_13eV=1.1716263232):
    alpha = 0.067366 + density * 0.039693
    eDensity = (density / molar_mass) * avo * atom_num
    return old_13eV*(18.7263 - 1.01e-23 * eDensity), alpha

def calculate_yield_parameters(E_field):
    m01 = 30.66 + (6.1978 - 30.66) / (1. + (E_field / 73.855)**2.0318)**0.41883
    m02 = 77.2931084
    m03 = jnp.log10(E_field) * 0.13946236 + 0.52561312
    m04 = 1.82217496 + (2.82528809 - 1.82217496) / (1. + (E_field / 144.65029656)**-2.80532006)
    # m05 = Nq / energy / (1. + alpha * erf(0.05 * E)) - m01
    m07 = 7.02921301 + (98.27936794 - 7.02921301) / (1. + (E_field / 256.48156448)**1.29119251)
    m08 = 4.285781736
    m09 = 0.3344049589
    m10 = 0.0508273937 + (0.1166087199 - 0.0508273937) / (1. + (E_field / 1.39260460e+02)**-0.65763592)
    return m01, m02, m03, m04, m07, m08, m09, m10

def get_yields_beta(E, density, exciton_ion_ratio, W, m01=7.096208, m02=77.2931084, m03=0.7155229, m04=1.8279102, m07=94.39740941928082, m08=4.285781736, m09=0.3344049589, m10=0.06623858):
    """
    Calculates the yields of quanta, electrons and photons for beta radiation in a detector.

    Parameters
    ----------
    E : scalar or array
        Energy of the incident beta radiation.
    density : scalar or array
        Density of the detector material.
    exciton_ion_ratio : scalar
        Ratio of excitons to ions produced in the detector.
    W : scalar
        Average energy required to produce an electron-ion pair.
    m01, m02, m03, m04, m07, m08, m09, m10 : scalars
        Parameter values used in the empirical formula for calculating yields.

    Returns
    -------
    mean_N_q : scalar or array
        Mean number of quanta produced per event.
    Ne : scalar or array
        Number of electrons produced.
    Nph : scalar or array
        Number of photons produced.

    Notes
    -----
    This function calculates the yields of quanta, electrons and photons produced by beta radiation 
    in LXe based on the energy of the incident radiation, the density of the detector material, 
    and the ratio of excitons to ions produced.
    """
    DENSITY = 2.9
    mean_N_q = E/W*1e3
    m05 = mean_N_q/E/(1 + exciton_ion_ratio) - m01
    Qy = m01 + (m02 - m01) / ((1. + (E / m03)**m04))**m09 + m05 + (0.0 - m05) / ((1. + (E / m07)**m08))**m10
    coeff_TI = (1. / DENSITY)**0.3
    coeff_Ni = (1. / DENSITY)**1.4
    coeff_OL = (1. / DENSITY)**-1.7 / jnp.log(1. + coeff_TI * coeff_Ni * (DENSITY**1.7))
    Qy *= coeff_OL * jnp.log(1. + coeff_TI * coeff_Ni * (density**1.7)) * (density**-1.7)
    Ly = mean_N_q / E - Qy
    Ne = Qy * E
    Nph = Ly * E
    return mean_N_q, Ne, Nph

def Nei_ratio(E, density, alpha):
    """
    Calculates the ratio of the number of electrons to ions for a given energy and material density.

    Parameters
    ----------
    E : scalar or array
        Energy of the incident radiation.
    density : scalar or array
        Density of the detector material.

    Returns
    -------
    Nei_ratio : scalar or array
        The ratio of the number of electrons to ions.

    Notes
    -----
    This function calculates the ratio of the number of electrons to ions produced by 
    ERs.
    """

    return alpha*erf(0.05*E)

@jit
def recom_omega_ER(E_field, elecFrac, width_param_7=0.046452, width_param_8=0.205, width_param_9=0.45, width_param_10=-0.2):
    '''
    This function calculates the omega parameter for electron recoil (ER) events based on certain input parameters.
    
    Parameters
    ----------
    E_field : float 
        The electric field value.
    elecFrac : float 
        The fraction of energy deposited by electrons.
    width_param_7 : float, optional 
        Parameter A from Table VI of https://arxiv.org/pdf/2211.10726.pdf; default is 0.046452.
    width_param_8 : float, optional 
        Omega parameter from Table VI of https://arxiv.org/pdf/2211.10726.pdf; default is 0.205.
    width_param_9 : float, optional 
        Xi parameter from Table VI of https://arxiv.org/pdf/2211.10726.pdf; default is 0.45.
    width_param_10 : float, optional 
        Skewness parameter; default is -0.2.

    Returns
    -------
    omega : float 
        The calculated omega parameter for ER events.
    width_param_10 : float 
        The skewness parameter.

    Notes
    -----
    The function uses the provided parameters to calculate the value of omega for ER events. It also returns the skewness parameter.
    '''
    A = 0.086036 + (width_param_7 - 0.086036) / (1 + (E_field/295.2)**251.6)**0.0069114
    wide = width_param_8
    cntr = width_param_9
    skew = width_param_10
    mode = cntr + 2 * (inv_sqrt2_PI) * skew * wide / jnp.sqrt(1. + skew * skew)
    norm = 1. / (jnp.exp(-0.5 * (mode - cntr)**2 / (wide * wide)) * (1. + erf(skew * (mode - cntr) / (wide * sqrt2))))
    omega = norm * A * jnp.exp(-0.5 * (elecFrac - cntr)**2 / (wide * wide)) * (1. + erf(skew * (elecFrac - cntr) / (wide * sqrt2)))
    return omega, width_param_10

# @partial(jax.checkpoint, static_argnums=(1))
@partial(jax.jit, static_argnums=(1))
def ER_skew(E, E_field, alpha0=1.39, cc0=4.0, cc1=22.1, E0=7.7, E1=54., E2=26.7, E3=6.4, F0=225., F1=71.):
    """
    LUX ER Skewness model for computing the recombination skewness. Default parameters from NEST 2.3.12. 
    
    Parameters:
    E (float): Energy in keV.
    E_field (float): Electric field strength in V/cm.
    alpha0 (float): Baseline scale factor for the ER response. Default value is 1.39.
    cc0 (float): Scale factor for the first exponential term. Default value is 4.0.
    cc1 (float): Scale factor for the second exponential term. Default value is 22.1.
    E0 (float): Decay constant for the first exponential term. Default value is 7.7.
    E1 (float): Decay constant for the second exponential term. Default value is 54.
    E2 (float): Inflection point for the sigmoid function. Default value is 26.7.
    E3 (float): Scale factor for the sigmoid function. Default value is 6.4.
    F0 (float): Decay constant for the field dependence of the first exponential term. Default value is 225.
    F1 (float): Decay constant for the field dependence of the second exponential term. Default value is 71.

    Returns:
    float: The ER response given the input parameters. If E_field is less than 50 or greater than 1e4, it returns 0.0.
    """
    if E_field < 50 or E_field > 1e4:
        return 0.0
    else:
        return 1. / (1. + jnp.exp((E - E2) / E3)) * (alpha0 + cc0 * jnp.exp(-1. * E_field / F0) * (1. - jnp.exp(-1. * E / E0))) + 1. / (1. + jnp.exp(-1. * (E - E2) / E3)) * cc1 * jnp.exp(-1. * E / E1) * jnp.exp(-1. * jnp.sqrt(E_field) / jnp.sqrt(F1))

@partial(jax.jit, static_argnums=(1, 2))
def marginalized_prob_quantas(E, E_field=23.0, density=2.8619, Nq_bins = jnp.arange(1,1400)-0.5, Nq_bins_centers=jnp.arange(1,1399), Ni_bin_centers=jnp.arange(1,900), Ne_bins=(jnp.arange(1,401)-0.5)):
    """
    This function calculates the marginalized probabilities of quantas given certain conditions (E, theta).
    Specifically, it computes $p(N_q, N_i, N_e | E, \theta)$.
    
    Note: plate dim is rightmost, specify in numpyro. The dimensions are: Ne, Ni, Nq, plate/batch dim.
    
    Parameters:
    E (float): Energy in keV.
    E_field (float): Electric field strength in V/cm. Default value is 23.0.
    density (float): Density of the medium. Default value is 2.8619.
    Nq_bins (ndarray): Bins for quanta number. Default starts from 0.5 to 1399.5.
    Nq_bins_centers (ndarray): Center points for Nq bins. Default starts from 1 to 1398.
    Ni_bin_centers (ndarray): Center points for Ni bins. Default starts from 1 to 899.
    Ne_bins (ndarray): Bins for electron number. Default starts from -0.5 to 398.5.

    Returns:
    tuple: Returns a tuple containing probability densities for Ne, Ni and Nq. 
    The product of the three output arrays will broadcast correctly to give $p(N_q, N_i, N_e | E, \theta)$;
    however, this product is not explicitly computed to reduce memory usage.
    """
    W, alpha = workfunction(density)
    m01, m02, m03, m04, m07, m08, m09, m10 = calculate_yield_parameters(E_field)
    exciton_ion_ratio = Nei_ratio(E, density, alpha)
    mean_N_q, mean_Ne, mean_Nph = get_yields_beta(E, density, exciton_ion_ratio, W, m01=m01, m02=m02, m03=m03, m04=m04, m07=m07, m08=m08, m09=m09, m10=m10)
    # fano_factor = fano_ER(mean_N_q, E_field, density)
    fano_factor = 1 #For some reason fano got set to 1 in NEST 2.3.12, but not in 2.3.11.
    Nq_ps = DiscreteNormalPs(mean_N_q, jnp.sqrt(fano_factor*mean_N_q), bin_edges=Nq_bins)
    alf = 1/(1 + exciton_ion_ratio)
    Ni_ps = stats.binom.pmf(n=Nq_bins_centers[jnp.newaxis, :,jnp.newaxis], p=alf, k=Ni_bin_centers[:, jnp.newaxis, jnp.newaxis])
    # Ni_ps = Ni_ps/jnp.sum(Ni_ps, axis=0)
    # Nq_Ni_ps = Nq_ps[:,jnp.newaxis,:] * Ni_ps

    e_frac = mean_Ne/mean_N_q
    recombProb = 1. - (exciton_ion_ratio + 1.) * e_frac
    recomb_omega, _ = recom_omega_ER(E_field, e_frac)
    skewness = ER_skew(E, E_field)
    # skewness = 0.
    recomb_variance = recombProb*(1 - recombProb)*Ni_bin_centers[:, jnp.newaxis, jnp.newaxis] + (recomb_omega*Ni_bin_centers[:, jnp.newaxis, jnp.newaxis])**2
    widthCorrection = jnp.sqrt(1. - (2. / jnp.pi) * skewness * skewness / (1. + skewness * skewness))
    muCorrection = (jnp.sqrt(recomb_variance) / widthCorrection) * (skewness / jnp.sqrt(1. + skewness * skewness)) * 2. * inv_sqrt2_PI
    Ne_mu = (1. - recombProb) * Ni_bin_centers[:, jnp.newaxis, jnp.newaxis] - muCorrection
    Ne_sigma = jnp.sqrt(recomb_variance) / widthCorrection
    # import pdb; pdb.set_trace()
    Ne_ps = DiscreteSkewNormalPs(Ne_mu, Ne_sigma, skewness, bin_edges=Ne_bins[:,jnp.newaxis,jnp.newaxis])
    # Ne_ni_nq_ps = Ne_ps*Ni_ps*Nq_ps
    return Ne_ps, Ni_ps, Nq_ps
    # return Ne_mu, Ne_sigma, skewness

    