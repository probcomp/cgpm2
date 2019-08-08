# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import base64
import cPickle
import itertools

from math import isnan
from math import log
from math import pi
from math import sqrt

from collections import OrderedDict
from collections import namedtuple

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from .distribution import DistributionCGPM

from .utils import get_prng
from .utils import log_pflip


INF = float('inf')
NAN = float('nan')
LOG2PI = log(2*pi)


class RandomForest(DistributionCGPM):

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
        self.k = int(distargs['k'])
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
        self.class_to_index = OrderedDict()
        self.regressor = None

    def observe(self, rowid, observation, inputs=None):
        assert rowid not in self.data
        assert list(observation) == self.outputs
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
        assert list(targets) == self.outputs
        x = targets[self.outputs[0]]
        if x not in self.class_to_index:
            return -INF
        if not self.regressor:
            return -log(self.k)
        ix = self.class_to_index[x]
        y_dum = self.process_inputs(inputs)
        y_dum_probe = np.reshape(y_dum, (1,-1))
        logps = self.regressor.predict_log_proba(y_dum_probe)
        return logps[0, ix]

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert targets == self.outputs
        assert not constraints
        if rowid in self.data:
            samples = [self.data[rowid]]*(N or 1)
        elif not self.regressor:
            samples = self.rng.choice(range(self.k), size=(N or 1))
        else:
            y_dum = self.process_inputs(inputs)
            y_dum_probe = np.reshape(y_dum, (1,-1))
            logps = self.regressor.predict_log_proba(y_dum_probe)
            samples = log_pflip(logps[0], array=list(self.class_to_index),
                size=(N or 1), rng=self.rng)
        return dictify_samples(self.outputs[0], samples, N)

    def logpdf_score(self):
        pass

    # DistributionCGPM methods.

    def transition(self):
        Y, x = get_training_data(self.data_ydum.values(), self.data.values())
        self.regressor = RandomForestClassifier(random_state=self.rng)
        self.regressor.fit(Y, x)
        self.class_to_index = OrderedDict([
            (l,i) for i,l in enumerate(self.regressor.classes_)
        ])

    def transition_hypers(self, N=None):
        pass

    def transition_params(self):
        pass

    def set_hypers(self, hypers):
        pass

    def get_hypers(self):
        return {}

    def get_params(self):
        return {}

    def get_suffstats(self):
        return {}

    def get_distargs(self):
        return self.distargs

    @staticmethod
    def construct_hyper_grids(X, n_grid=300):
        return {}

    @staticmethod
    def name():
        return 'random_forest'

    @staticmethod
    def is_collapsed():
        return False

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return True

    @staticmethod
    def is_numeric():
        return False

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
        metadata['distargs'] = self.get_distargs()
        metadata['hypers'] = self.get_hypers()
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['data_yraw'] = self.data_yraw.items()
        metadata['data_ydum'] = self.data_ydum.items()
        metadata['class_to_index'] = self.class_to_index.items()
        metadata['regressor'] = None if self.regressor is None \
            else base64.b64encode(cPickle.dumps(self.regressor))
        metadata['factory'] = ('cgpm2.forest', 'RandomForest')
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
        model.class_to_index = OrderedDict(metadata['class_to_index'])
        if metadata['regressor'] is not None:
            model.regressor = \
                cPickle.loads(base64.b64decode(metadata['regressor']))
        return model


def dummy_code_one(codes, i, val):
    if i not in codes:
        return [float(val)]
    elif isnan(val):
        levels = codes[i]
        return [NAN]*(levels-1)
    else:
        levels = codes[i]
        coded = [0]*(levels-1)
        assert 0 <= val < levels
        assert int(val) == val
        if val < levels - 1:
            coded[int(val)] = 1
        return coded

def get_training_data(data_Y, data_x):
    assert len(data_Y) == len(data_x)
    Y = np.asarray(data_Y)
    x = np.asarray(data_x)
    y_nan = np.any(np.isnan(Y), axis=1)
    x_nan = np.isnan(x)
    nans = np.logical_or(y_nan, x_nan)
    return (Y[~nans], x[~nans])

def dictify_samples(output, samples, N):
    if N is None:
        assert len(samples) == 1
        return {output: samples[0]}
    else:
        assert len(samples) == N
        return [{output: sample} for sample in samples]
