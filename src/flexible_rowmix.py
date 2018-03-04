# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import importlib

from cgpm.utils.general import get_prng
from cgpm.utils.general import lchain

from .finite_rowmix import FiniteRowMixture
from .flexible_array import FlexibleArray


class FlexibleRowMixture(FiniteRowMixture):

    def __init__(self, cgpm_row_divide, cgpm_components_base, rng=None):
        # Assertions.
        assert len(cgpm_row_divide.outputs) == 1
        # From constructor.
        self.cgpm_row_divide = cgpm_row_divide
        self.cgpm_components_base = cgpm_components_base
        self.rng = rng or get_prng()
        # Derived attributes.
        self.outputs_z = self.cgpm_row_divide.outputs
        self.outputs_x = self.cgpm_components_base.outputs
        self.inputs_z = self.cgpm_row_divide.inputs
        self.inputs_x = self.cgpm_components_base.inputs
        self.outputs = lchain(self.outputs_z, self.outputs_x)
        self.inputs = lchain(self.inputs_z, self.inputs_x)
        self.indexer = self.outputs[0]
        # Internal attributes.
        self.indexer = self.outputs[0]
        self.rowid_to_component = {}
        self.cgpm_components_array = FlexibleArray(
            self.cgpm_components_base, self.indexer, self.rng)

    def to_metadata(self):
        metadata = dict()
        metadata['cgpm_row_divide'] = self.cgpm_row_divide.to_metadata()
        metadata['cgpm_components_array'] = \
            self.cgpm_components_array.to_metadata()
        metadata['rowid_to_cgpm'] = self.rowid_to_cgpm.items()
        metadata['factory'] = ('cgpm.compositors.flexible_rowmix',
            'FlexibleRowMixture')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        def build_cgpm(blob):
            modname, attrname = blob['factory']
            module = importlib.import_module(modname)
            builder = getattr(module, attrname)
            return builder.from_metadata(blob, rng)
        cgpm_row_divide = build_cgpm(metadata['cgpm_row_divide'])
        cgpm_components_array = build_cgpm(['cgpm_components_array'])
        model = cls(cgpm_row_divide, cgpm_components_array.cgpm_base, rng)
        model.rowid_to_component = dict(metadata['rowid_to_component'])
        model.cgpm_components_array = cgpm_components_array
        return model

    def render(self):
        return [
            'FlexibleRowMixture',
            ['cgpm_row_divide=', self.cgpm_row_divide.render()],
            ['cgpm_base=', self.cgpm_components_base.render()],
            ['cgpm_components=', self.cgpm_components_array.render()],
        ]
