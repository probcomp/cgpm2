# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import build_cgpm
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
        self.rng = rng or get_prng(1)
        # Derived attributes.
        self.outputs_z = cgpm_row_divide.outputs
        self.inputs_z = cgpm_row_divide.inputs
        self.outputs_x = cgpm_components_base.outputs
        self.inputs_x = cgpm_components_base.inputs
        self.outputs = lchain(self.outputs_z, self.outputs_x)
        self.inputs = lchain(self.inputs_z, self.inputs_x)
        self.indexer = self.outputs[0]
        # Internal attributes.
        self.indexer = self.outputs[0]
        self.rowid_to_component = {}
        self.cgpm_components_array = FlexibleArray(
            cgpm_components_base, self.indexer, self.rng)

    def to_metadata(self):
        metadata = dict()
        metadata['cgpm_row_divide'] = self.cgpm_row_divide.to_metadata()
        metadata['cgpm_components_array'] = \
            self.cgpm_components_array.to_metadata()
        metadata['rowid_to_component'] = self.rowid_to_component.items()
        metadata['factory'] = ('cgpm2.flexible_rowmix',
            'FlexibleRowMixture')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        cgpm_row_divide = build_cgpm(metadata['cgpm_row_divide'], rng)
        cgpm_components_array = \
            build_cgpm(metadata['cgpm_components_array'], rng)
        model = cls(cgpm_row_divide, cgpm_components_array.cgpm_base, rng)
        model.rowid_to_component = dict(metadata['rowid_to_component'])
        model.cgpm_components_array = cgpm_components_array
        return model

    def render(self):
        return [
            'FlexibleRowMixture',
            ['cgpm_row_divide=', self.cgpm_row_divide.render()],
            ['cgpm_components=', self.cgpm_components_array.render()],
        ]
