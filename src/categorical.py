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
from cgpm.utils.general import pflip
from cgpm.utils.general import simulate_many

from .distribution import DistributionCGPM


class Categorical(DistributionCGPM):

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
        self.k = int(distargs['k'])
        self.rng = rng or get_prng()
        # Internal attributes.
        self.data = OrderedDict()
        self.N = 0
        self.counts = np.zeros(self.k)
        self.alpha = hypers.get('alpha', 1)

    def incorporate(self, rowid, observation, inputs=None):
        DistributionCGPM.incorporate(self, rowid, observation, inputs)
        x = observation[self.outputs[0]]
        if not isnan(x):
            if not (x % 1 == 0 and 0 <= x < self.k):
                raise ValueError('Invalid Categorical(%d): %s' % (self.k, x))
            x = int(x)
            self.N += 1
            self.counts[x] += 1
        self.data[rowid] = x

    def unincorporate(self, rowid):
        DistributionCGPM.unincorporate(self, rowid)
        x = self.data.pop(rowid)
        if not isnan(x):
            self.N -= 1
            self.counts[x] -= 1
        return {self.outputs[0]: x}, {}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        DistributionCGPM.logpdf(self, rowid, targets, constraints, inputs)
        x = targets[self.outputs[0]]
        if isnan(x):
            return 0.
        if not (x % 1 == 0 and 0 <= x < self.k):
            return -float('inf')
        return calc_predictive_logp(int(x), self.N, self.counts, self.alpha)

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        x = pflip(self.counts + self.alpha, rng=self.rng)
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.counts, self.alpha)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['k'] = self.k
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['counts'] = list(self.counts)
        metadata['alpha'] = self.alpha
        metadata['factory'] = ('cgpm2.categorical', 'Categorical')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'],
            distargs={'k': metadata['k']}, rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.N = metadata['N']
        model.counts = np.array(metadata['counts'])
        model.alpha = metadata['alpha']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {'alpha': self.alpha}

    def get_params(self):
        return self.params

    def get_suffstats(self):
        return {'N' : self.N, 'counts' : list(self.counts)}

    def get_distargs(self):
        return {'k': self.k}

    def support(self):
        return range(self.k)

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = log_linspace(1., float(len(X)), n_grid)
        return grids

    @staticmethod
    def name():
        return 'categorical'

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

    @staticmethod
    def validate(x, K):
        return int(x) == float(x) and 0 <= x < K


def calc_predictive_logp(x, N, counts, alpha):
    numer = log(alpha + counts[x])
    denom = log(np.sum(counts) + alpha * len(counts))
    return numer - denom

def calc_logpdf_marginal(N, counts, alpha):
    K = len(counts)
    A = K * alpha
    lg = sum(gammaln(counts[k] + alpha) for k in xrange(K))
    return gammaln(A) - gammaln(A+N) + lg - K * gammaln(alpha)
