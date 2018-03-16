# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict
from math import isnan
from math import log

import numpy as np

from scipy.special import gammaln

from cgpm.utils.general import get_prng
from cgpm.utils.general import log_linspace
from cgpm.utils.general import simulate_many

from .distribution import DistributionCGPM


class Poisson(DistributionCGPM):

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
        self.rng = rng or get_prng()
        # Internal attributes.
        self.data = OrderedDict()
        self.N = 0
        self.sum_x = 0
        self.sum_log_fact_x = 0
        self.a = hypers.get('a', 1.)
        self.b = hypers.get('b', 1.)
        assert self.a > 0
        assert self.b > 0

    def observe(self, rowid, observation, inputs=None):
        DistributionCGPM.observe(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if not isnan(x):
            if not (x % 1 == 0 and x >= 0):
                raise ValueError('Invalid Poisson: %s' % str(x))
            self.N += 1
            self.sum_x += x
            self.sum_log_fact_x += gammaln(x+1)
        self.data[rowid] = x

    def unobserve(self, rowid):
        DistributionCGPM.unobserve(self, rowid)
        x = self.data.pop(rowid)
        if not isnan(x):
            self.N -= 1
            self.sum_x -= x
            self.sum_log_fact_x -= gammaln(x+1)
        return {self.outputs[0]: x}, {}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionCGPM.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if not (x % 1 == 0 and x >= 0):
            return -float('inf')
        if isnan(x):
            return 0.
        return calc_predictive_logp(x, self.N, self.sum_x, self.a, self.b)

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data and not isnan(self.data[rowid]):
            return {self.outputs[0]: self.data[rowid]}
        an, bn = posterior_hypers(self.N, self.sum_x, self.a, self.b)
        x = self.rng.negative_binomial(an, bn/(bn+1.))
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.sum_x, self.sum_log_fact_x,
            self.a, self.b)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['sum_x'] = self.sum_x
        metadata['sum_log_fact_x'] = self.sum_log_fact_x
        metadata['a'] = self.a
        metadata['b'] = self.b
        metadata['factory'] = ('cgpm2.poisson', 'Poisson')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'], rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.N = metadata['N']
        model.sum_x = metadata['sum_x']
        model.sum_log_fact_x = metadata['sum_log_fact_x']
        model.a = metadata['a']
        model.b = metadata['b']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['a'] > 0
        assert hypers['b'] > 0
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {'a': self.a, 'b': self.b}

    def get_params(self):
        return self.params

    def get_suffstats(self):
        return {'N': self.N, 'sum_x' : self.sum_x,
            'sum_log_fact_x': self.sum_log_fact_x}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = len(X) if len(X) > 0 else 2
        grids['a'] = np.unique(np.round(np.linspace(1, N, n_grid)))
        grids['b'] = log_linspace(.1, float(N), n_grid)
        return grids

    @staticmethod
    def name():
        return 'poisson'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True


def calc_log_Z(a, b):
    Z =  gammaln(a) - a*log(b)
    return Z

def posterior_hypers(N, sum_x, a, b):
    an = a + sum_x
    bn = b + N
    return an, bn

def calc_predictive_logp(x, N, sum_x, a, b):
    an, bn = posterior_hypers(N, sum_x, a, b)
    am, bm = posterior_hypers(N+1, sum_x+x, a, b)
    ZN = calc_log_Z(an, bn)
    ZM = calc_log_Z(am, bm)
    return  ZM - ZN - gammaln(x+1)

def calc_logpdf_marginal(N, sum_x, sum_log_fact_x, a, b):
    an, bn = posterior_hypers(N, sum_x, a, b)
    Z0 = calc_log_Z(a, b)
    ZN = calc_log_Z(an, bn)
    return ZN - Z0 - sum_log_fact_x
