# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import importlib

from .icgpm import CGPM


class FlexibleArray(CGPM):

    def __init__(self, cgpm_base, indexer, rng):
        # From constructor.
        self.cgpm_base = cgpm_base
        self.rng = rng
        # Derived attributes.
        self.outputs = self.cgpm_base.outputs
        self.inputs = [indexer] + self.cgpm_base.inputs
        self.indexer = indexer
        self.cgpm_base_metadata = self.cgpm_base.to_metadata()
        # Internal attributes.
        self.cgpms = {}
        self.rowid_to_index = {}

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.get_cgpm(i_select)
        return cgpm.simulate(rowid, targets, constraints, inputs, N)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.get_cgpm(i_select)
        return cgpm.logpdf(rowid, targets, constraints, inputs)

    def incorporate(self, rowid, observation, inputs=None):
        i_select = inputs.pop(self.indexer)
        cgpm = self.get_cgpm(i_select)
        cgpm.incorporate(rowid, observation, inputs)
        self.rowid_to_index[rowid] = i_select

    def unincorporate(self, rowid):
        i_select = self.rowid_to_index[rowid]
        del self.rowid_to_index[rowid]
        cgpm = self.get_cgpm(i_select)
        cgpm.unincorporate(rowid)

    def transition(self, **kwargs):
        return

    def to_metadata(self):
        metadata = dict()
        metadata['cgpm_base_metadata'] = self.cgpm_base_metadata
        metadata['indexer'] = self.inputs[0]
        metadata['rowid_to_index'] = self.rowid_to_index.items()
        metadata['cgpms'] = [(i, cgpm.to_metadata())
            for i, cgpm in self.cgpms.iteritems()]
        metadata['factory'] = ('cgpm2.flexible_array', 'FlexibleArray')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        def build_cgpm(blob):
            modname, attrname = blob['factory']
            module = importlib.import_module(modname)
            builder = getattr(module, attrname)
            return builder.from_metadata(blob, rng)
        cgpm_base = build_cgpm(metadata['cgpm_base_metadata'])
        model = cls(cgpm_base, metadata['indexer'], rng)
        model.cgpms = {i: build_cgpm(blob) for i, blob in metadata['cgpms']}
        model.rowid_to_index = dict(metadata['rowid_to_index'])
        return model

    # Internal

    def get_cgpm(self, i_select):
        if i_select not in self.cgpms:
            self.cgpms[i_select] = self.create_new_cgpm()
        return self.cgpms[i_select]

    def create_new_cgpm(self):
        modname, attrname = self.cgpm_base_metadata['factory']
        module = importlib.import_module(modname)
        builder = getattr(module, attrname)
        return builder.from_metadata(self.cgpm_base_metadata, self.rng)
