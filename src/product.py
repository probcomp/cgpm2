# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools as it

from cgpm.utils.general import build_cgpm
from cgpm.utils.general import flatten_cgpms
from cgpm.utils.general import get_intersection
from cgpm.utils.general import get_prng
from cgpm.utils.general import lchain
from cgpm.utils.general import mergedl

from .chain import Chain


class Product(Chain):

    def __init__(self, cgpms, rng=None):
        # Assertions.
        validate_cgpms_product(cgpms)
        # From constructor.
        self.cgpms = flatten_cgpms(cgpms, Product)
        self.rng = rng or get_prng(1)
        # Derived attributes.
        self.outputs = lchain(*[cgpm.outputs for cgpm in self.cgpms])
        self.inputs = lchain(*[cgpm.inputs for cgpm in self.cgpms])
        self.output_to_index = {output:i for i, cgpm in enumerate(self.cgpms)
            for output in cgpm.outputs}

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        samples = [simulate_one(cgpm, rowid, targets, constraints, inputs, N)
            for cgpm in self.cgpms]
        return merge_samples(samples)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        logps = [logpdf_one(cgpm, rowid, targets, constraints, inputs)
            for cgpm in self.cgpms]
        return sum(logps)

    def transition(self, **kwargs):
        return

    def to_metadata(self):
        metadata = dict()
        metadata['cgpms'] = [cgpm.to_metadata() for cgpm in self.cgpms]
        metadata['factory'] = ('cgpm2.product', 'Product')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        cgpms = [build_cgpm(blob, rng) for blob in metadata['cgpms']]
        model = cls(cgpms, rng)
        return model

    def render(self):
        return [
            'Product',
            ['cgpms=', [cgpm.render() for cgpm in self.cgpms]]
        ]


def validate_cgpms_product(cgpms):
    # Check all outputs are disjoint.
    outputs_list = list(it.chain(*(cgpm.outputs for cgpm in cgpms)))
    outputs_set = set(outputs_list)
    assert len(outputs_list) == len(outputs_set)
    # Check no input is an output of another cgpm.
    inputs_list = list(it.chain(*(cgpm.inputs for cgpm in cgpms)))
    assert all([c not in outputs_set for c in inputs_list])

def merge_samples(samples):
    if isinstance(samples[0], dict):
        return mergedl(samples)
    elif isinstance(samples[0], list):
        samples_list = zip(*samples)
        return [mergedl(sample) for sample in samples_list]
    else:
        assert False, 'Unknown samples return type'

def simulate_one(cgpm, rowid, targets, constraints, inputs, N=None):
    targets_cgpm = get_intersection(cgpm.outputs, targets)
    if not targets_cgpm:
        return {} if N is None else [{}]*N
    constraints_cgpm = get_intersection(cgpm.outputs, constraints)
    inputs_cgpm = get_intersection(cgpm.inputs, inputs)
    return cgpm.simulate(rowid, targets_cgpm, constraints_cgpm, inputs_cgpm, N)

def logpdf_one(cgpm, rowid, targets, constraints, inputs):
    targets_cgpm = get_intersection(cgpm.outputs, targets)
    if not targets_cgpm:
        return 0
    constraints_cgpm = get_intersection(cgpm.outputs, constraints)
    inputs_cgpm = get_intersection(cgpm.inputs, inputs)
    return cgpm.logpdf(rowid, targets_cgpm, constraints_cgpm, inputs_cgpm)
