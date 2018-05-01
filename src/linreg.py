# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

from math import log
from math import pi
from math import sqrt

from collections import OrderedDict
from collections import namedtuple

import numpy as np

from numpy.linalg import det

from scipy.special import gammaln

from cgpm.utils.data import dummy_code
from cgpm.utils.general import get_prng
from cgpm.utils.general import log_linspace
from cgpm.utils.general import simulate_many

from .distribution import DistributionCGPM


LOG2PI = log(2*pi)
Data = namedtuple('Data', ['x', 'Y'])


class LinearRegression(DistributionCGPM):
    """Bayesian linear model with normal prior on regression parameters and
    inverse-gamma prior on both observation and regression variance.

    Reference
    http://www.biostat.umn.edu/~ph7440/pubh7440/BayesianLinearModelGoryDetails.pdf


    Y_i = w' X_i + \sigma^2
        Response data                   Y_i \in R
        Covariate vector                X_i \in R^p
        Regression coefficients         w \in R^p
        Regression variance             \sigma^2 \in R

    Hyperparameters:                    a=1, b=1, V=I, mu=[0], dimension=p

    Parameters:                         \sigma2 ~ Inverse-Gamma(a, b)
                                        w ~ MVNormal(\mu, \sigma2*I)

    Data                                Y_i|x_i ~ Normal(w' x_i, \sigma2)
    """

    def __init__(self, outputs, inputs, hypers=None, params=None, distargs=None,
            rng=None):
        # Populate default kwargs.
        hypers = hypers or dict()
        params = params or dict()
        distargs = distargs or dict()
        # From constructor.
        assert len(outputs) == 1
        assert len(inputs) > 0
        assert outputs[0] not in inputs
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.params = params
        self.rng = rng or get_prng(1)
        # Derived attributes.
        assert len(distargs['levels']) == len(inputs)
        self.distargs = distargs
        self.levels = {i:l for i,l in zip(self.inputs, distargs['levels']) if l}
        self.ps = [l - 1 if l else 1 for l in distargs['levels']]
        self.p = 1 + sum(self.ps)
        # Internal attributes.
        self.N = 0
        self.data = OrderedDict()
        self.data_yraw = OrderedDict()
        self.data_ydum = OrderedDict()
        self.a = hypers.get('a', 1.)
        self.b = hypers.get('b', 1.)
        self.mu = hypers.get('mu', np.zeros(self.p))
        self.V = hypers.get('V', np.eye(self.p))

    def observe(self, rowid, observation, inputs=None):
        assert rowid not in self.data
        assert observation.keys() == self.outputs
        x = observation[self.outputs[0]]
        y_raw = [inputs.get(i) for i in self.inputs]
        y_dum = self.process_inputs(inputs)
        self.N += 1
        self.data[rowid] = x
        self.data_yraw[rowid] = y_raw
        self.data_ydum[rowid] = y_dum

    def unobserve(self, rowid):
        x = self.data.pop(rowid)
        y_raw = self.data_yraw.pop(rowid)
        del self.data_ydum[rowid]
        return {self.outputs[0]: x}, dict(zip(self.inputs), y_raw)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert rowid not in self.data
        assert not constraints
        assert targets.keys() == self.outputs
        x = targets[self.outputs[0]]
        y_dum = self.process_inputs(inputs)
        return calc_predictive_logp(x, y_dum, self.N, self.data_ydum.values(),
            self.data.values(), self.a, self.b, self.mu, self.V)

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert not constraints
        if rowid in self.data:
            return {self.outputs[0]: self.data[rowid]}
        y_dum = self.process_inputs(inputs)
        sigma2, b = self.simulate_params()
        x = self.rng.normal(np.dot(y_dum, b), np.sqrt(sigma2))
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.data_ydum.values(),
            self.data.values(), self.a, self.b, self.mu, self.V)

    def simulate_params(self):
        an, bn, mun, Vn_inv = posterior_hypers(self.N, self.data_ydum.values(),
            self.data.values(), self.a, self.b, self.mu, self.V)
        return sample_parameters(an, bn, mun, np.linalg.inv(Vn_inv), self.rng)


    # DistributionCGPM methods.

    def transition(self, N=None):
        pass

    def transition_hypers(self, N=None):
        pass

    def transition_params(self):
        pass

    def set_hypers(self, hypers):
        assert hypers['a'] > 0.
        assert hypers['b'] > 0.
        self.a = hypers['a']
        self.b = hypers['b']

    def get_hypers(self):
        return {'a': self.a, 'b':self.b}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return self.distargs

    @staticmethod
    def construct_hyper_grids(X, n_grid=300):
        grids = dict()
        N = len(X) + 1.
        ssqdev = np.var(X) * len(X) + 1.
        grids['a'] = log_linspace(1./(10*N), 10*N, n_grid)
        grids['b'] = log_linspace(ssqdev/100., ssqdev, n_grid)
        return grids

    @staticmethod
    def name():
        return 'linear_regression'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return True

    # Helpers.

    def process_inputs(self, inputs):
        y_dum = list(itertools.chain.from_iterable([
            dummy_code_one(self.levels, i, inputs[i]) for i in self.inputs
        ]))
        y_bias = [1] + y_dum
        assert len(y_dum) == self.p - 1
        return y_bias

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['data_yraw'] = self.data_yraw.items()
        metadata['data_ydum'] = self.data_ydum.items()
        metadata['distargs'] = self.get_distargs()
        metadata['hypers'] = self.get_hypers()
        metadata['factory'] = ('cgpm2.linreg', 'LinearRegression')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(
            outputs=metadata['outputs'],
            inputs=metadata['inputs'],
            hypers=metadata['hypers'],
            distargs=metadata['distargs'],
            rng=rng,
        )
        model.N = metadata['N']
        model.data = OrderedDict(metadata['data'])
        model.data_yraw = OrderedDict(metadata['data_yraw'])
        model.data_ydum = OrderedDict(metadata['data_ydum'])
        return model


def calc_log_Z(a, b, V_inv):
    # Equation 19.
    return gammaln(a) + log(sqrt(1./det(V_inv))) - a * np.log(b)

def posterior_hypers(N, Y, x, a, b, mu, V):
    if N == 0:
        assert len(x) == len(Y) == 0
        return a, b, mu, np.linalg.inv(V)
    # Equation 6.
    X, y = np.asarray(Y), np.asarray(x)
    assert X.shape == (N,len(mu))
    assert y.shape == (N,)
    V_inv = np.linalg.inv(V)
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    mun = np.dot(
        np.linalg.inv(V_inv + XTX),
        np.dot(V_inv, mu) + np.dot(XT, y))
    Vn_inv = V_inv + XTX
    an = a + N/2.
    bn = b + .5 * (
        np.dot(np.transpose(mu), np.dot(V_inv, mu))
        + np.dot(np.transpose(x), x)
        - np.dot(
            np.transpose(mun),
            np.dot(Vn_inv, mun)))
    return an, bn, mun, Vn_inv

def calc_predictive_logp(xs, ys, N, Y, x, a, b, mu, V):
    # Equation 19.
    an, bn, _mun, Vn_inv = posterior_hypers(N, Y, x, a, b, mu, V)
    am, bm, _mum, Vm_inv = posterior_hypers(N+1, Y+[ys], x+[xs], a, b, mu, V)
    ZN = calc_log_Z(an, bn, Vn_inv)
    ZM = calc_log_Z(am, bm, Vm_inv)
    return (-1/2.)*LOG2PI + ZM - ZN

def calc_logpdf_marginal(N, Y, x, a, b, mu, V):
    # Equation 19.
    an, bn, _mun, Vn_inv = posterior_hypers(N, Y, x, a, b, mu, V)
    Z0 = calc_log_Z(a, b, np.linalg.inv(V))
    ZN = calc_log_Z(an, bn, Vn_inv)
    return (-N/2.)*LOG2PI + ZN - Z0

def sample_parameters(a, b, mu, V, rng):
    sigma2 = 1./rng.gamma(a, scale=1./b)
    b = rng.multivariate_normal(mu, sigma2 * V)
    return sigma2, b

def dummy_code_one(codes, i, val):
    if i not in codes:
        return [float(val)]
    levels = codes[i]
    assert 0 <= val < levels
    coded = [0]*(levels-1)
    if val < levels - 1:
        coded[val] = 1
    return coded
