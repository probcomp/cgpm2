# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict
from math import isnan
from math import log
from math import pi
from math import sqrt

from scipy.stats import norm

from .distribution import DistributionCGPM
from .utils import get_prng

LOG2 = log(2)
LOGPI = log(pi)
LOG2PI = LOG2 + LOGPI

class NormalUC(DistributionCGPM):

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        # Populate default kwargs.
        hypers = hypers or dict()
        params = params or dict()
        distargs = distargs or dict()
        # From constructor.
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.hypers = hypers
        self.params = params
        self.rng = rng or get_prng(1)
        # Internal attributes.
        self.data = OrderedDict()
        self.N = 0
        self.sum_x = 0
        self.sum_x_sq = 0
        self.mu = params.get('mu', 0.)
        self.var = params.get('var', 1.)
        assert self.var > 0

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
        return norm.logpdf(x, loc=self.mu, scale=sqrt(self.var))

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid in self.data and not isnan(self.data[rowid]):
            return {self.outputs[0]: self.data[rowid]}
        x_list = self.rng.normal(loc=self.mu, scale=sqrt(self.var), size=N)
        return {self.outputs[0]: x_list} if N is None else \
            [{self.outputs[0]: x} for x in x_list]

    def logpdf_score(self):
        # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood#hid4
        term_1 = -(self.N/2.) * LOG2PI
        term_2 = -(self.N/2.) * log(self.var)
        term_3_prefactor = -1. / (2*self.var)
        term_3_sum = self.sum_x_sq - 2*self.mu * self.sum_x + self.N*self.mu**2
        return term_1 + term_2 + term_3_prefactor * term_3_sum

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['data'] = self.data.items()
        metadata['hypers'] = self.hypers
        metadata['N'] = self.N
        metadata['sum_x'] = self.sum_x
        metadata['sum_x_sq'] = self.sum_x_sq
        metadata['mu'] = self.mu
        metadata['var'] = self.var
        metadata['factory'] = ('cgpm2.normal_uc', 'NormalUC')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'], rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.hypers = metadata['hypers']
        model.N = metadata['N']
        model.sum_x = metadata['sum_x']
        model.sum_x_sq = metadata['sum_x_sq']
        model.mu = metadata['mu']
        model.var = metadata['var']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['r'] > 0.
        assert hypers['s'] > 0.
        assert hypers['nu'] > 0.
        self.hypers['m'] = hypers['m']
        self.hypers['r'] = hypers['r']
        self.hypers['s'] = hypers['s']
        self.hypers['nu'] = hypers['nu']

    def get_hypers(self):
        return dict(self.hypers)

    def set_params(self, params):
        assert params['var'] > 0.
        self.mu = params['mu']
        self.var = params['var']

    def get_params(self):
        return {'mu': self.mu, 'var': self.var}

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return {}

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        return dict()

    @staticmethod
    def name():
        return 'normal_uc'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return True
