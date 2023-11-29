from functools import partial
import numpy as np
import scipy.stats as spystats
import corner
import arviz as az

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

@jit
def DiscreteNormalPs(loc, scale, bin_edges=(jnp.linspace(-0,201,202)-0.5)):
    """
    Computes the probability mass function of a discretized Normal distribution.
    
    This function calculates the difference between the cumulative distribution function (CDF) values at 
    given bin edges and then normalizes these probabilities.

    Parameters:
    loc (float): The mean (mu) of the Normal distribution.
    scale (float): The standard deviation (sigma) of the Normal distribution.
    bin_edges (array_like): The array containing the edges of the bins. Default is an array of shape (202, 1)
                             with values ranging from -0.5 to 200.5.

    Returns:
    array_like: A transposed version of the normalized PDF values corresponding to the given bin edges.
    """

    # Calculate the differences in CDF values for the provided bin edges
    cdf = stats.norm.cdf(bin_edges[:, jnp.newaxis],loc,scale)
    ps = jnp.diff(cdf, axis=0)

    # Obtain the machine limits for floating point types
    # finfo = jnp.finfo(jnp.result_type(ps, float))

    # Clip the calculated probabilities to avoid underflow or overflow issues
    # ps = jnp.clip(ps, a_min=finfo.eps, a_max=1.0 - finfo.eps)

    # Normalize and return the probabilities so that their sum is 1
    return ps/(1-cdf[0,:])


def DiscreteNormal(loc, scale, validate_args=None):
    """
    Computes the Categorical distribution probabilities using a discretized Normal distribution.

    This function calculates the differences in cumulative distribution function (CDF) values 
    for a provided Normal distribution defined by its mean (loc) and standard deviation (scale). 
    It then uses these probabilities to instantiate a Categorical distribution.

    Parameters:
    loc (float): The mean (mu) of the Normal distribution.
    scale (float): The standard deviation (sigma) of the Normal distribution.
    validate_args (bool, optional): Whether to enable validation of input arguments. 
                                    Default is None, meaning no argument validation.

    Returns:
    Categorical Distribution: A Categorical distribution object instantiated with 
                              the calculated probabilities.
    """
    # Calculate the differences in CDF values for a Normal distribution
    categorical_p_arr = DiscreteNormalPs(loc, scale)

    # Instantiate a Categorical distribution with the calculated probabilities
    return ndist.CategoricalProbs(categorical_p_arr, validate_args=validate_args)

class SkewNormal(ndist.Distribution):
    """
    Implementation of the Skew Normal distribution.

    Attributes
    ----------
    arg_constraints : dict
        A dictionary indicating the constraints for the distribution parameters.
    support : ndist.constraints.real
        The real value support for the distribution.
    reparametrized_params : list
        List of reparametrized parameters.
    loc : scalar or array, optional
        Location parameter, default is 0.0.
    scale : scalar or array, optional
        Scale parameter, default is 1.0.
    skew : scalar or array, optional
        Skew parameter, default is 0.
    validate_args : bool, optional
        If true, distribution parameters are checked for validity despite possibly degrading runtime performance, default is None.

    Methods
    -------
    __init__(self, loc=0.0, scale=1.0, skew=0, *, validate_args=None)
        Constructor method.
    sample(self, key, sample_shape=())
        Draw samples from the distribution.
    log_prob(self, value)
        Compute the log probability density function at a given point.
    cdf(self, value)
        Compute the cumulative distribution function at a given point.
    log_cdf(self, value)
        Compute the log of the cumulative distribution function at a given point.
    mean(self)
        Compute the expected value of the distribution.
    variance(self)
        Compute the variance of the distribution.

    """

    arg_constraints = {"loc": ndist.constraints.real, "scale": ndist.constraints.positive, "skew": ndist.constraints.real}
    support = ndist.constraints.real
    # reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, skew=0, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), jnp.shape(skew))
        self.loc, self.scale, self.skew = ndist.util.promote_shapes(loc, scale, skew, shape=batch_shape)
        self._delta = self.skew/jnp.sqrt(1 + self.skew**2)
        super(SkewNormal, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert numpyro.util.is_prng_key(key)
        eps = spystats.skewnorm.rvs(
            self.skew, size=sample_shape + self.batch_shape + self.event_shape, random_state=np.random.RandomState(key)
        )
        return self.loc + eps * self.scale

    @ndist.util.validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(0.5*jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled**2 - normalize_term + stats.norm.logcdf(self.skew*value_scaled)

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return ndtr(scaled) - 2*tfp.math.owens_t(scaled, self.skew)

    def log_cdf(self, value):
        return jnp.log(self.cdf(value))

#     def icdf(self, q):
#         return self.loc + self.scale * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc + self.scale*self._delta*jnp.sqrt(2/jnp.pi), self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2*(1 - 2*self._delta**2/jnp.pi), self.batch_shape)
    
@jit
def DiscreteSkewNormalPs(loc, scale, skew, bin_edges=(jnp.linspace(-0,201,202)-0.5)):
    """
    Computes the probability mass function (pmf) for a discretized version of the Skew Normal distribution.

    Parameters
    ----------
    loc : scalar or array
        Location parameter of the Skew Normal distribution.
    scale : scalar or array
        Scale parameter of the Skew Normal distribution.
    skew : scalar or array
        Skew parameter of the Skew Normal distribution.
    bin_edges : array, optional
        The bin edges for the discretization of the Skew Normal distribution. 
        Default is a linspace from -0 to 201 with 202 points reshaped into a column vector.

    Returns
    -------
    ps : array
        The pmf values corresponding to each bin edge, normalized such that the sum across all bins is 1.

    Notes
    -----
    This function works by first creating a Skew Normal distribution with the given parameters. 
    Then it computes the cumulative distribution function (cdf) at each bin edge, and takes the differences 
    to find the probability mass in each bin. The resulting pmf is then normalized so that its total mass is 1.

    """

    skew_normal = SkewNormal(loc, scale, skew)
    cdf = skew_normal.cdf(bin_edges[:, jnp.newaxis])
    ps = jnp.diff(cdf, axis=0)
    # finfo = jnp.finfo(jnp.result_type(ps, float))
    # ps = jnp.clip(ps, a_min=finfo.eps, a_max=1.0 - finfo.eps)
    return ps/(1-cdf[0,:])

class GaussianMixture2D(ndist.Distribution):
    """
    Implementation of the Gaussian Mixture distribution in 2D.

    Attributes
    ----------
    arg_constraints : dict
        A dictionary indicating the constraints for the distribution parameters.
    support : ndist.constraints.real
        The real value support for the distribution.
    means : array
        Array of means. Last dimension should have a size of 2.
    stds : array
        Array of standard deviations. Last dimension should have a size of 2.
    weights : array
        Array of weights. Should sum to 1.
    validate_args : bool, optional
        If true, distribution parameters are checked for validity despite possibly degrading runtime performance, default is None.

    Methods
    -------
    __init__(self, means, stds, weights, *, validate_args=None)
        Constructor method.
    sample(self, key, sample_shape=())
        Draw samples from the distribution.
    log_prob(self, value)
        Compute the log probability density function at a given point.

    """

    arg_constraints = {"means": ndist.constraints.real, "stds": ndist.constraints.positive, "weights": ndist.constraints.simplex}
    support = ndist.constraints.real

    def __init__(self, means, stds, weights, *, validate_args=None):
        batch_shape = lax.broadcast_shapes(jnp.shape(means)[:-2], jnp.shape(stds)[:-2], jnp.shape(weights)[:-1])
        self.mixture_shape = jnp.shape(means)[-2:-1]
        event_shape = (2,)
        self.means, self.stds = ndist.util.promote_shapes(means, stds, shape=batch_shape+self.mixture_shape+event_shape)
        self.weights, = ndist.util.promote_shapes(weights, shape=batch_shape+self.mixture_shape)
        super(GaussianMixture2D, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert numpyro.util.is_prng_key(key)
        component = jax.random.categorical(key, jnp.log(self.weights), shape=sample_shape + self.batch_shape)
        cov_matrices = (self.stds**2)[..., component, :, jnp.newaxis] * jnp.identity(*self.event_shape)
        return jax.random.multivariate_normal(key, self.means[..., component, :], cov_matrices, shape=sample_shape + self.batch_shape)

    @ndist.util.validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi)**2*self.stds[...,0]*self.stds[...,1])
        value_scaled = (value - self.means) / self.stds
        return jnp.sum(-0.5 * value_scaled**2, axis=-1) - normalize_term


    # def cdf(self, value):
    #     scaled = (value - self.loc) / self.scale
    #     return ndtr(scaled) - 2*tfp.math.owens_t(scaled, self.skew)

    # def log_cdf(self, value):
    #     return jnp.log(self.cdf(value))

#     def icdf(self, q):
#         return self.loc + self.scale * ndtri(q)

    # @property
    # def mean(self):
    #     return jnp.broadcast_to(self.loc + self.scale*self._delta*jnp.sqrt(2/jnp.pi), self.batch_shape)

    # @property
    # def variance(self):
    #     return jnp.broadcast_to(self.scale**2*(1 - 2*self._delta**2/jnp.pi), self.batch_shape)