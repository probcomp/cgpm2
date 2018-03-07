# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

from math import isinf

from cgpm.network.helpers import retrieve_adjacency_list
from cgpm.network.helpers import retrieve_extraneous_inputs
from cgpm.network.helpers import retrieve_required_inputs
from cgpm.network.helpers import retrieve_variable_to_cgpm
from cgpm.network.helpers import topological_sort
from cgpm.network.helpers import validate_cgpms

from cgpm.utils.general import build_cgpm
from cgpm.utils.general import flatten_cgpms
from cgpm.utils.general import get_intersection
from cgpm.utils.general import get_prng
from cgpm.utils.general import lchain
from cgpm.utils.general import log_pflip
from cgpm.utils.general import logmeanexp
from cgpm.utils.general import merged
from cgpm.utils.general import mergedl
from cgpm.utils.general import simulate_many

from .icgpm import CGPM


class Chain(CGPM):
    """Querier for a Composite CGpm."""

    def __init__(self, cgpms, accuracy=None, rng=None):
        # Validate inputs.
        cgpms_valid = validate_cgpms(cgpms)
        # From constructor
        self.cgpms = flatten_cgpms(cgpms_valid, Chain)
        self.accuracy = accuracy or 1
        self.rng = rng if rng else get_prng(1)
        # Derived attributes.
        self.outputs = lchain(*[cgpm.outputs for cgpm in self.cgpms])
        self.inputs = lchain(*[cgpm.inputs for cgpm in self.cgpms])
        self.v_to_c = retrieve_variable_to_cgpm(self.cgpms)
        self.adjacency = retrieve_adjacency_list(self.cgpms, self.v_to_c)
        self.extraneous = retrieve_extraneous_inputs(self.cgpms, self.v_to_c)
        self.topo = topological_sort(self.adjacency)

    def incorporate(self, rowid, observation, inputs=None):
        for cgpm in self.cgpms:
            incorporate_one(cgpm, rowid, observation, inputs)

    def unincorporate(self, rowid):
        observations_list, inputs_list = zip(*[
            unincorporate_one(cgpm, rowid)
            for cgpm in self.cgpms
        ])
        observations_dict = mergedl(observations_list)
        inputs_dict = mergedl(inputs_list)
        return observations_dict, inputs_dict

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        constraints = constraints or {}
        inputs = inputs or {}
        # Generate samples and weights.
        samples, weights = zip(*[
            self.weighted_sample(rowid, targets, constraints, inputs)
            for _i in xrange(self.accuracy)
        ])
        # Sample importance resample.
        if all(isinf(l) for l in weights):
            raise ValueError('Zero density constraints: %s' % (constraints,))
        index = 0 if self.accuracy == 1 else log_pflip(weights, rng=self.rng)
        return {q: samples[index][q] for q in targets}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        constraints = constraints or {}
        inputs = inputs or {}
        # Compute joint probability.
        _samples_joint, weights_joint = zip(*[
            self.weighted_sample(
                rowid, [], merged(targets, constraints), inputs)
            for _i in xrange(self.accuracy)
        ])
        logp_joint = logmeanexp(weights_joint)
        # Compute marginal probability.
        _samples_marginal, weights_marginal = zip(*[
            self.weighted_sample(rowid, [], constraints, inputs)
            for _i in xrange(self.accuracy)
        ]) if constraints else ({}, [0.])
        if all(isinf(l) for l in weights_marginal):
            raise ValueError('Zero density constraints: %s' % (constraints,))
        logp_constraints = logmeanexp(weights_marginal)
        # Return log ratio.
        return logp_joint - logp_constraints

    def logpdf_score(self):
        return sum(cgpm.logpdf_score() for cgpm in self.cgpms)

    def to_metadata(self):
        metadata = dict()
        metadata['cgpms'] = [cgpm.to_metadata() for cgpm in self.cgpms]
        metadata['accuracy'] = self.accuracy
        metadata['factory'] = ('cgpm2.chain', 'Chain')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        cgpms = [build_cgpm(blob, rng) for blob in metadata['cgpms']]
        model = cls(cgpms, accuracy=metadata['accuracy'], rng=rng)
        return model

    def render(self):
        return [
            'Chain',
            ['cgpms=', [cgpm.render() for cgpm in self.cgpms]],
        ]

    # Helpers.

    def weighted_sample(self, rowid, targets, constraints, inputs):
        targets_required = retrieve_required_inputs(
            self.cgpms, self.topo, targets, constraints, self.extraneous)
        targets_all = list(itertools.chain(targets, targets_required))
        sample = dict(constraints)
        weight = 0
        for level in self.topo:
            sample_level, weight_level = self.invoke_cgpm(
                rowid, self.cgpms[level], targets_all, sample, inputs)
            sample.update(sample_level)
            weight += weight_level
        assert set(sample) == set.union(set(constraints), set(targets_all))
        return sample, weight

    def invoke_cgpm(self, rowid, cgpm, targets, constraints, inputs):
        cgpm_inputs = {e:x for e,x in
            itertools.chain(inputs.iteritems(), constraints.iteritems())
            if e in cgpm.inputs
        }
        cgpm_constraints = {e:x for e, x in constraints.iteritems()
            if e in cgpm.outputs
        }
        cgpm_targets = [q for q in targets if q in cgpm.outputs]
        if cgpm_constraints or cgpm_targets:
            assert all(i in cgpm_inputs for i in cgpm.inputs)
        weight = cgpm.logpdf(
            rowid,
            cgpm_constraints,
            constraints=None,
            inputs=cgpm_inputs,
        ) if cgpm_constraints else 0
        sample = cgpm.simulate(
            rowid,
            cgpm_targets,
            constraints=cgpm_constraints,
            inputs=cgpm_inputs,
        ) if cgpm_targets else {}
        return sample, weight


def incorporate_one(cgpm, rowid, observation, inputs):
    observation_cgpm = get_intersection(cgpm.outputs, observation)
    if observation_cgpm:
        inputs_cgpm = get_intersection(cgpm.inputs, inputs)
        cgpm.incorporate(rowid, observation_cgpm, inputs_cgpm)

def unincorporate_one(cgpm, rowid):
    try:
        return cgpm.unincorporate(rowid)
    except Exception:
        return {}, {}
