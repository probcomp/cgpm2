# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict
from math import isnan
from math import lgamma
from math import log
from math import pi
from math import sqrt

import numba
import numpy as np

from scipy.stats import t as student_t

from .distribution import DistributionCGPM

from .utils import get_prng
from .utils import log_linspace
from .utils import simulate_many

LOG2 = log(2)
LOGPI = log(pi)
LOG2PI = LOG2 + LOGPI


class Normal(DistributionCGPM):
    # http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        # Populate default kwargs.
        hypers = hypers or dict()
        params = params or dict()
        distargs = distargs or dict()
        # From constructor.
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.params = params
        self.rng = rng or get_prng(1)
        # Internal attributes.
        self.data = OrderedDict()
        self.N = 0
        self.sum_x = 0
        self.sum_x_sq = 0
        self.m = hypers.get('m', 0.)
        self.r = hypers.get('r', 1.)
        self.s = hypers.get('s', 1.)
        self.nu = hypers.get('nu', 1.)
        assert self.s > 0.
        assert self.r > 0.
        assert self.nu > 0.

    def observe(self, rowid, observation, inputs=None):
        DistributionCGPM.observe(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if not isnan(x):
            self.N += 1.
            self.sum_x += x
            self.sum_x_sq += x*x
        self.data[rowid] = x

    def unobserve(self, rowid):
        DistributionCGPM.unobserve(self, rowid)
        x = self.data.pop(rowid)
        if not isnan(x):
            self.N -= 1
            self.sum_x -= x
            self.sum_x_sq -= x*x
        return {self.outputs[0]: x}, {}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionCGPM.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if isnan(x):
            return 0.
        return calc_predictive_logp(x, self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data and not isnan(self.data[rowid]):
            return {self.outputs[0]: self.data[rowid]}
        mn, rn, sn, nun = posterior_hypers(self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)
        mu, rho = sample_parameters(mn, rn, sn, nun, self.rng)
        x = self.rng.normal(loc=mu, scale=rho**-.5)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.sum_x, self.sum_x_sq,
            self.m, self.r, self.s, self.nu)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['sum_x'] = self.sum_x
        metadata['sum_x_sq'] = self.sum_x_sq
        metadata['m'] = self.m
        metadata['r'] = self.r
        metadata['s'] = self.s
        metadata['nu'] = self.nu
        metadata['factory'] = ('cgpm2.normal', 'Normal')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'], rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.N = metadata['N']
        model.sum_x = metadata['sum_x']
        model.sum_x_sq = metadata['sum_x_sq']
        model.m = metadata['m']
        model.r = metadata['r']
        model.s = metadata['s']
        model.nu = metadata['nu']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['r'] > 0.
        assert hypers['s'] > 0.
        assert hypers['nu'] > 0.
        self.m = hypers['m']
        self.r = hypers['r']
        self.s = hypers['s']
        self.nu = hypers['nu']

    def get_hypers(self):
        return {'m': self.m, 'r': self.r, 's': self.s, 'nu': self.nu}

    def get_params(self):
        return self.params

    def get_suffstats(self):
        return {'N': self.N, 'sum_x': self.sum_x, 'sum_x_sq': self.sum_x_sq}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        # Plus 1 for single observation case.
        N = len(X) if len(X) > 0 else 5
        minX = min(X) if len(X) > 0 else 0
        maxX = max(X) if len(X) > 0 else 1
        ssqdev = np.var(X) * N + 1 if len(X) > 0 else 1
        # Data dependent heuristics.
        grids['m'] = np.linspace(minX, maxX + 5, n_grid)
        grids['r'] = log_linspace(1. / N, N, n_grid)
        grids['s'] = log_linspace(ssqdev / 100., ssqdev, n_grid)
        grids['nu'] = log_linspace(1., N, n_grid) # df >= 1
        return grids

    @staticmethod
    def name():
        return 'normal'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True


@numba.jit
def calc_log_Z(r, s, nu):
    return (
        ((nu + 1.) / 2.) * LOG2
        + .5 * LOGPI
        - .5 * log(r)
        - (nu/2.) * log(s)
        + lgamma(nu/2.))

@numba.jit
def posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu):
    rn = r + float(N)
    nun = nu + float(N)
    mn = (r*m + sum_x)/rn
    sn = s + sum_x_sq + r*m*m - rn*mn*mn
    if sn == 0:
        sn = s
    return mn, rn, sn, nun

@numba.jit
def calc_predictive_logp(x, N, sum_x, sum_x_sq, m, r, s, nu):
    _mn, rn, sn, nun = posterior_hypers(N, sum_x, sum_x_sq, m, r, s, nu)
    _mm, rm, sm, num = posterior_hypers(N+1, sum_x+x, sum_x_sq+x*x, m, r, s, nu)
    ZN = calc_log_Z(rn, sn, nun)
    ZM = calc_log_Z(rm, sm, num)
    return -.5 * LOG2PI + ZM - ZN

@numba.jit
def calc_logpdf_marginal(N, sum_x, sum_x_sq, m, r, s, nu):
    _mn, rn, sn, nun = posterior_hypers(
        N, sum_x, sum_x_sq, m, r, s, nu)
    Z0 = calc_log_Z(r, s, nu)
    ZN = calc_log_Z(rn, sn, nun)
    return -(N/2.) * LOG2PI + ZN - Z0

def sample_parameters(m, r, s, nu, rng):
    rho = rng.gamma(nu/2., scale=2./s)
    mu = rng.normal(loc=m, scale=1./(rho*r)**.5)
    return mu, rho
