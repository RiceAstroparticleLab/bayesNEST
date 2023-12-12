from functools import partial

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np

import scipy.stats as spystats

from tqdm import trange, tqdm

import jax
from jax import config, jit, lax, random
# config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import stats

from . import NEST
from .distributions import *

Nph_bins_centers = jnp.arange(1,1399)
eps = np.finfo(float).eps

def safe_for_grad_log(x):
    return jnp.log(jnp.where(x > 0., x, eps))

@jit
# @jax.checkpoint
def calculate_Nph_prob_inner_loop(j, i, summed_marg_prob, Nph_bins_centers):
    Ni_ps = jnp.zeros_like(Nph_bins_centers, dtype=float)
    return Ni_ps.at[j-i].set(summed_marg_prob[i, j] + Ni_ps[j-i])

calculate_Nph_prob_inner_loop_vmap = jax.vmap(calculate_Nph_prob_inner_loop, in_axes=(0, None, None, None))
calculate_Nph_prob_inner_loop_vmap_vmap = jax.vmap(calculate_Nph_prob_inner_loop_vmap, in_axes=(None, 0, None, None))

@jit
# @jax.checkpoint
def calculate_Nph_prob_middle_loop(i, summed_marg_prob, Nph_bins_centers):
    Ni_ps = jnp.zeros_like(Nph_bins_centers, dtype=float)
    return jax.lax.fori_loop(0, 1398, lambda j, x: calculate_Nph_prob_inner_loop(j, i, summed_marg_prob, Nph_bins_centers) + x, Ni_ps)

calculate_Nph_prob_middle_loop_vmap = jax.vmap(calculate_Nph_prob_middle_loop, in_axes=(0, None, None))

@partial(jax.jit, static_argnums=(2))
# @jax.checkpoint
def compute_Nph_ps(summed_marg_prob, Ni_bins_centers, Ne_range=399):
    Nph_ps = jnp.zeros(1398, dtype=float)
    return jax.lax.fori_loop(0, Ne_range, lambda i,x: calculate_Nph_prob_middle_loop(i, (summed_marg_prob), Ni_bins_centers)+x, Nph_ps)

    # for i in range(Ne_range):
    #     Nph_ps += jnp.sum(calculate_Nph_prob_inner_loop_vmap(jnp.arange(0, 1398), i, (summed_marg_prob), Ni_bins_centers), axis=0)
    # return Nph_ps
    # Nph_ps = calculate_Nph_prob_inner_loop_vmap_vmap(jnp.arange(0, 1398), jnp.arange(0, 399), summed_marg_prob, Nph_bins_centers)
    # return jnp.sum(Nph_ps, axis=(0,1))

@partial(jax.jit, static_argnums=(1, 2, 3))
@jax.checkpoint
def compute_Ne_Nph_ps(summed_marg_prob, Nph_bins_centers=Nph_bins_centers, Ni_range=1398, Ne_range=399):
    # Ne_Nph_ps = jnp.zeros((Ne_range, Ni_range), dtype=float)
    # return jax.lax.fori_loop(0, Ne_range, lambda i,x: x.at[i].set(calculate_Nph_prob_middle_loop(i, (summed_marg_prob), Nph_bins_centers)), Ne_Nph_ps)
    # for i in range(Ne_range):
    #     Ne_Nph_ps = Ne_Nph_ps.at[i].set(jnp.sum(calculate_Nph_prob_inner_loop_vmap(jnp.arange(0, 1398), i, (summed_marg_prob), Nph_bins_centers), axis=0))
    # return Ne_Nph_ps
    return calculate_Nph_prob_middle_loop_vmap(jnp.arange(0, Ne_range), summed_marg_prob, Nph_bins_centers)


compute_Ne_Nph_ps_vmap = jax.checkpoint(jax.vmap(compute_Ne_Nph_ps, in_axes=(0)))

@jax.jit
def binom_log_pmf(n, p, k):
    logp = gammaln(n + 1) - gammaln(k + 1) - gammaln(n-k + 1) + xlogy(k, p) + xlog1py(n-k, -p)
    return jnp.where(k > n, -jnp.inf, logp)

@jax.jit
@jax.checkpoint
def g1_binom_inner_loop(Ne_Nph_ps, i, g1, Nphdet_bin_centers, Nph_bins_centers):
    return binom_log_pmf(n=Nph_bins_centers[i], p=g1, k=Nphdet_bin_centers)[jnp.newaxis, :] + safe_for_grad_log(Ne_Nph_ps)[:, i, jnp.newaxis]
    # return stats.binom.logpmf(n=Nph_bins_centers[i], p=g1, k=Nphdet_bin_centers)[jnp.newaxis, :] + safe_for_grad_log(Ne_Nph_ps)[:, i, jnp.newaxis]
    # return jnp.exp(stats.binom.logpmf(n=Nph_bins_centers[i], p=g1, k=Nphdet_bin_centers))[jnp.newaxis, :]*Ne_Nph_ps[:, i, jnp.newaxis]

g1_binom_inner_loop_vmap = jax.vmap(g1_binom_inner_loop, in_axes=(None, 0, None, None, None))

@jax.jit
@jax.checkpoint
def marginalise_g1_binom(Ne_Nph_ps, g1, Ne_range=399, Nphdet_range=200, Ni_range=1398, Nphdet_bin_centers=jnp.arange(1, 201), Nph_bins_centers=Nph_bins_centers):
    Ne_Nphdet_ps = jnp.zeros((Ne_range, Nphdet_range), dtype=float)-1e100
    return jax.lax.fori_loop(0, Ni_range, lambda i,x: jax.nn.logsumexp(jnp.array([g1_binom_inner_loop(Ne_Nph_ps, i, g1, Nphdet_bin_centers, Nph_bins_centers), x]), axis=0), Ne_Nphdet_ps)
    # for i in range(Ni_range):
    #     Ne_Nphdet_ps += g1_binom_inner_loop(Ne_Nph_ps, i, g1, Nphdet_bin_centers, Nph_bins_centers)
    # return Ne_Nphdet_ps
    # return jax.nn.logsumexp(g1_binom_inner_loop_vmap(Ne_Nph_ps, jnp.arange(Ni_range), g1, Nphdet_bin_centers, Nph_bins_centers), axis=0)
    # return jnp.sum(g1_binom_inner_loop_vmap(Ne_Nph_ps, jnp.arange(Ni_range), g1, Nphdet_bin_centers, Nph_bins_centers), axis=0)

marginalise_g1_binom_vmap = jax.checkpoint(jax.vmap(marginalise_g1_binom, in_axes=(0, None)))

@jax.checkpoint
def sum_marg_probs(marg_prob_Ne, marg_prob_Ni, marg_prob_Nq, Ni_range=899, Nq_range=1398, Ne_range=399, N_events=1):
    output = jnp.zeros((Ne_range, Nq_range, N_events))
    fori_func = lambda i, x: x + marg_prob_Ne[:, i, :, :] *  marg_prob_Ni[jnp.newaxis, i,:,:] * marg_prob_Nq[jnp.newaxis, :, :]
    # for i in range(Ni_range):
    #     output += marg_prob_Ne[:, i, :, :] *  marg_prob_Ni[jnp.newaxis, i,:,:] * marg_prob_Nq[jnp.newaxis, jnp.newaxis, :, :]
    # return jnp.moveaxis(output, -1, 0)[:,0,:,:]
    return jnp.moveaxis(jax.lax.fori_loop(0, Ni_range, fori_func, output), -1, 0)