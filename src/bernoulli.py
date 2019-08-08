# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict
from math import isnan
from math import log

import numpy as np

from scipy.special import betaln

from .distribution import DistributionCGPM

from .utils import get_prng
from .utils import log_linspace
from .utils import log_pflip
from .utils import simulate_many

class Bernoulli(DistributionCGPM):

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
        self.x_sum = 0
        self.alpha = hypers.get('alpha', 1)
        self.beta = hypers.get('beta', 1)

    def observe(self, rowid, observation, inputs=None):
        DistributionCGPM.observe(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if isnan(x):
            self.data[rowid] = x
        else:
            assert x in [0, 1]
            x_int = int(x)
            self.N += 1
            self.x_sum += x_int
            self.data[rowid] = x_int

    def unobserve(self, rowid):
        DistributionCGPM.unobserve(self, rowid)
        x = self.data.pop(rowid)
        if not isnan(x):
            self.N -= 1
            self.x_sum -= x
        return {self.outputs[0]: x}, {}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionCGPM.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if x not in [0, 1]:
            return -float('inf')
        return calc_predictive_logp(x, self.N, self.x_sum, self.alpha, self.beta)

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data and not isnan(self.data[rowid]):
            return {self.outputs[0]: self.data[rowid]}
        p0 = calc_predictive_logp(0, self.N, self.x_sum, self.alpha, self.beta)
        p1 = calc_predictive_logp(1, self.N, self.x_sum, self.alpha, self.beta)
        x = log_pflip([p0, p1], rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.x_sum, self.alpha, self.beta)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['data'] = self.data.items()
        metadata['N'] = self.N
        metadata['x_sum'] = self.x_sum
        metadata['alpha'] = self.alpha
        metadata['beta'] = self.beta
        metadata['factory'] = ('cgpm2.bernoulli', 'Bernoulli')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'], rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.N = metadata['N']
        model.x_sum = np.array(metadata['x_sum'])
        model.alpha = metadata['alpha']
        model.beta = metadata['beta']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        assert hypers['beta'] > 0
        self.alpha = hypers['alpha']
        self.beta = hypers['beta']

    def get_hypers(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    def get_params(self):
        return self.params

    def get_suffstats(self):
        return {'N':self.N, 'x_sum':self.x_sum}

    def get_distargs(self):
        return {}

    def support(self):
        return [0, 1]

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        N = len(X) if len(X) > 0 else 2
        grids['alpha'] = log_linspace(1./N, N, n_grid)
        grids['beta'] = log_linspace(1./N, N, n_grid)
        return grids

    @staticmethod
    def name():
        return 'bernoulli'

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
        return False

    # Helpers.

def calc_predictive_logp(x, N, x_sum, alpha, beta):
    log_denom = log(N + alpha + beta)
    if x == 1:
        return log(x_sum + alpha) - log_denom
    else:
        return log(N - x_sum + beta) - log_denom

def calc_logpdf_marginal(N, x_sum, alpha, beta):
    return betaln(x_sum + alpha, N - x_sum + beta) - betaln(alpha, beta)
