# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import math

import numpy as np

from scipy.stats import norm

from cgpm2.utils import get_prng
from cgpm2.utils import pflip


def gen_data_table(n_rows, view_weights, cluster_weights, cctypes, distargs,
        separation, view_partition=None, rng=None):
    """Generates data, partitions, and Dim.

     Parameters
     ----------
     n_rows : int
        Mumber of rows (data points) to generate.
     view_weights : list<float>
        An n_views length list of floats that sum to one. The weights indicate
        the proportion of columns in each view.
    cluster_weights : list<list<float>>
        An n_views length list of n_cluster length lists that sum to one.
        The weights indicate the proportion of rows in each cluster.
     cctypes : list<str>
        n_columns length list of string specifying the distribution types for
        each column.
     distargs : list
        List of distargs for each column (see documentation for each data type
            for info on distargs).
     separation : list
        An n_cols length list of values between [0,1], where seperation[i] is
        the seperation of clusters in column i. Values closer to 1 imply higher
        seperation.

     Returns
     -------
     T : np.ndarray
        An (n_cols, n_rows) matrix, where each row T[i,:] is the data for
        column i (tranpose of a design matrix).
    Zv : list
        An n_cols length list of integers, where Zv[i] is the view assignment
        of column i.
    Zc : list<list>
        An n_view length list of lists, where Zc[v][r] is the cluster assignment
        of row r in view v.

    Example
    -------
    >>> n_rows = 500
    >>> view_weights = [.2, .8]
    >>> cluster_weights = [[.3, .2, .5], [.4, .6]]
    >>> cctypes = ['lognormal','normal','poisson','categorical',
    ...     'vonmises', 'bernoulli']
    >>> distargs = [None, None, None, {'k':8}, None, None]
    >>> separation = [.8, .7, .9, .6, .7, .85]
    >>> T, Zv, Zc, dims = tu.gen_data_table(n_rows, view_weights,
    ...     cluster_weights, dists, distargs, separation)
    """
    if rng is None:
        rng = get_prng()

    n_cols = len(cctypes)

    if view_partition:
        Zv = list(view_partition)
    else:
        Zv = gen_partition(n_cols, view_weights, rng)

    Zc = [gen_partition(n_rows, cw, rng) for cw in cluster_weights]

    assert len(Zv) == n_cols
    assert len(Zc) == len(set(Zv))
    assert len(Zc[0]) == n_rows

    T = np.zeros((n_cols, n_rows))

    for col in range(n_cols):
        cctype = cctypes[col]
        args = distargs[col]
        view = Zv[col]
        Tc = _gen_data[cctype](
            Zc[view],
            rng,
            separation=separation[col],
            distargs=args)
        T[col] = Tc

    return T, Zv, Zc

def _gen_beta_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    K = np.max(Z)+1
    alphas = np.linspace(.5 - .5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        alpha = alphas[cluster]
        beta = (1.-alpha) * 20.* (norm.pdf(alpha, .5, .25))
        alpha *= 20. * norm.pdf(alpha, .5, .25)
        Tc[r] = rng.beta(alpha, beta)

    return Tc

def _gen_normal_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (5.*separation)
        sigma = 1.0
        Tc[r] = rng.normal(loc=mu, scale=sigma)

    return Tc

def _gen_normal_trunc_data(Z, rng, separation=.9, distargs=None):
    l, h = distargs['l'], distargs['h']
    max_draws = 100
    n_rows = len(Z)

    K = max(Z) + 1
    mean = (l+h)/2.

    bins = np.linspace(l, h, K+1)
    bin_centers = [.5*(bins[i-1]+bins[i]) for i in range(1, len(bins))]
    distances = [mean - bc for bc in bin_centers]
    mus = [bc + (1-separation)*d for bc, d in zip(bin_centers, distances)]

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        sigma = 1
        i = 0
        while True:
            i += 1
            x = rng.normal(loc=mus[cluster], scale=sigma)
            if l <= x <= h:
                break
            if max_draws < i:
                raise ValueError('Could not generate normal_trunc data.')
        Tc[r] = x

    return Tc

def _gen_vonmises_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    num_clusters = max(Z)+1
    sep = 2*math.pi / num_clusters

    mus = [c*sep for c in range(num_clusters)]
    std = sep/(5.*separation**.75)
    k = 1 / (std*std)

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = mus[cluster]
        Tc[r] = rng.vonmises(mu, k) + math.pi

    return Tc

def _gen_poisson_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        lam = cluster * (4.*separation) + 1
        Tc[r] = rng.poisson(lam)

    return Tc

def _gen_exponential_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)

    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (4.*separation) + 1
        Tc[r] = rng.exponential(mu)

    return Tc

def _gen_geometric_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)
    Tc = np.zeros(n_rows)
    K = np.max(Z)+1

    ps = np.linspace(.5 - .5*separation*.85, .5 + .5*separation*.85, K)
    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        Tc[r] = rng.geometric(ps[cluster]) -1

    return Tc

def _gen_lognormal_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    if separation > .9:
        separation = .9

    Tc = np.zeros(n_rows)
    for r in range(n_rows):
        cluster = Z[r]
        mu = cluster * (.9*separation**2)
        Tc[r] = rng.lognormal(mean=mu, sigma=(1.-separation)/(cluster+1.))

    return Tc

def _gen_bernoulli_data(Z, rng, separation=.9, distargs=None):
    n_rows = len(Z)

    Tc = np.zeros(n_rows)
    K = max(Z)+1
    thetas = np.linspace(0., separation, K)

    for r in range(n_rows):
        cluster = Z[r]
        theta = thetas[cluster]
        x = 0.0
        if rng.rand() < theta:
            x = 1.0
        Tc[r] = x

    return Tc

def _gen_categorical_data(Z, rng, separation=.9, distargs=None):
    k = int(distargs['k'])
    n_rows = len(Z)

    if separation > .95:
        separation = .95

    Tc = np.zeros(n_rows, dtype=int)
    C = max(Z)+1
    theta_arrays = [rng.dirichlet(np.ones(k)*(1.-separation), 1)
        for _ in range(C)]

    for r in range(n_rows):
        cluster = Z[r]
        thetas = theta_arrays[cluster][0]
        x = pflip(thetas, rng=rng)
        Tc[r] = int(x)
    return Tc

def gen_partition(N, weights, rng):
    assert all(w != 0 for w in weights)
    assert np.allclose(sum(weights), 1)
    K = len(weights)
    assert K <= N    # XXX FIXME
    Z = list(range(K))
    Z.extend(int(pflip(weights, rng=rng)) for _ in range(N-K))
    rng.shuffle(Z)
    return Z

_gen_data = {
    'bernoulli'         : _gen_bernoulli_data,
    'beta'              : _gen_beta_data,
    'categorical'       : _gen_categorical_data,
    'exponential'       : _gen_exponential_data,
    'geometric'         : _gen_geometric_data,
    'lognormal'         : _gen_lognormal_data,
    'normal'            : _gen_normal_data,
    'normal_trunc'      : _gen_normal_trunc_data,
    'poisson'           : _gen_poisson_data,
    'vonmises'          : _gen_vonmises_data,
}
