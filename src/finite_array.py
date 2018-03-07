# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import build_cgpm

from .icgpm import CGPM


class FiniteArray(CGPM):

    def __init__(self, cgpms, indexer, rng):
        # From constructor.
        self.cgpms = cgpms
        self.rng = rng
        # Derived attributes.
        self.outputs = self.cgpms[0].outputs
        self.inputs = [indexer] + self.cgpms[0].inputs
        self.indexer = indexer
        # Internal attributes.
        self.rowid_to_index = {}

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.cgpms[i_select]
        return cgpm.simulate(rowid, targets, constraints, inputs, N)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.cgpms[i_select]
        return cgpm.logpdf(rowid, targets, constraints, inputs)

    def logpdf_score(self):
        return sum(cgpm.logpdf_score for cgpm in self.cgpms)

    def incorporate(self, rowid, observation, inputs=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.cgpms[i_select]
        cgpm.incorporate(rowid, observation, inputs)
        self.rowid_to_index[rowid] = i_select

    def unincorporate(self, rowid):
        i_select = self.rowid_to_index[rowid]
        del self.rowid_to_index[rowid]
        cgpm = self.cgpms[i_select]
        return cgpm.unincorporate(rowid)

    def transition(self, **kwargs):
        return

    def to_metadata(self):
        metadata = dict()
        metadata['cgpms'] = [cgpm.to_metadata() for cgpm in self.cgpms]
        metadata['indexer'] = self.inputs[0]
        metadata['rowid_to_index'] = self.rowid_to_index.items()
        metadata['factory'] = ('cgpm2.finite_array', 'FiniteArray')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        cgpms = [build_cgpm(blob, rng) for blob in metadata['cgpms']]
        model = cls(cgpms, metadata['indexer'], rng)
        model.rowid_to_index = dict(metadata['rowid_to_index'])
        return model

    def render(self):
        return [
            'FiniteArray',
            ['cgpms=', [cgpm.render() for cgpm in self.cgpms]]
        ]
