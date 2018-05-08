# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

from cgpm.utils.validation import validate_dependency_constraints
from cgpm.utils.validation import validate_crp_constrained_partition

from .crp import CRP
from .distribution import DistributionCGPM
from .flexible_rowmix import FlexibleRowMixture
from .product import Product

from .progress import transition_generic

from .transition_crosscat_cpp import transition_cpp

from .transition_hypers import set_hypers
from .transition_hypers import transition_hyper_grids
from .transition_hypers import transition_hypers

from .transition_rows import get_rowids
from .transition_rows import set_rowid_component
from .transition_rows import transition_rows

from .transition_views import get_cgpm_current_view_index
from .transition_views import set_cgpm_view_assignment
from .transition_views import transition_cgpm_view_assigments

from .walks import get_cgpms_by_output_index


def get_distribution_outputs(crosscat):
    return list(itertools.chain.from_iterable(
        [view.outputs[1:] for view in crosscat.cgpms]
    ))

def get_row_mixture_cgpms(crosscat, outputs=None):
    outputs = outputs or get_distribution_outputs(crosscat)
    view_indexes = set([
        get_cgpm_current_view_index(crosscat, [output])
        for output in outputs
    ])
    return [crosscat.cgpms[i] for i in view_indexes]

def get_row_divide_outputs(crosscat, outputs=None):
    views = get_row_mixture_cgpms(crosscat, outputs)
    return [view.outputs[0] for view in views]

def get_distribution_cgpms(crosscat, outputs=None):
    outputs = outputs or get_distribution_outputs(crosscat)
    return {
        output: get_cgpms_by_output_index(crosscat, output)
        for output in outputs
    }

def get_row_divide_cgpms(crosscat, outputs=None):
    crp_outputs = get_row_divide_outputs(crosscat, outputs)
    return {
        output: get_cgpms_by_output_index(crosscat, output)
        for output in crp_outputs
    }

def validate_crosscat(crosscat):
    assert isinstance(crosscat, Product)
    for view in crosscat.cgpms:
        assert isinstance(view, (FlexibleRowMixture))
        assert isinstance(view.cgpm_row_divide, CRP)
        array = view.cgpm_components_array
        assert isinstance(array.cgpm_base, Product)
        for component in array.cgpm_base.cgpms:
            assert isinstance(component, DistributionCGPM)

def validate_crosscat_dependencies(crosscat, Cd, Ci):
    if not (Cd or Ci):
        return True
    outputs = sorted(get_distribution_outputs(crosscat))
    N = len(outputs)
    assert outputs == range(N)
    validate_dependency_constraints(N, Cd, Ci)
    Zv = {c: i for i, cgpm in enumerate(crosscat.cgpms) for c in cgpm.outputs}
    assert validate_crp_constrained_partition(Zv, Cd, Ci, [], [])

def get_rowids_transition(view, rowids):
    rowids_view = get_rowids(view)
    return rowids_view if not rowids \
        else [r for r in rowids_view if r in rowids]

class GibbsCrossCat(object):

    def __init__(self, crosscat, Cd=None, Ci=None):
        # From constructor.
        self.crosscat = crosscat
        self.Cd = tuple(Cd or [])
        self.Ci = tuple(Ci or [])
        # Confirm CrossCat is well-formed.
        validate_crosscat(crosscat)
        validate_crosscat_dependencies(crosscat, self.Cd, self.Ci)
        # Derived attributes.
        self.grids = dict()
        self.transition_hyper_grids_row_divide()
        self.transition_hyper_grids_distribution()

    # Stochastic mutation (python synthesizer).

    def transition(self, N=None, S=None, kernels=None, outputs=None,
            rowids=None, progress=None):
        kernel_lookup = {
            'hypers_distributions':
                lambda : self.transition_hypers_distributions(outputs=outputs),
            'hypers_row_divide':
                lambda : self.transition_hypers_row_divide(outputs=outputs),
            'hypers_column_divide':
                self.transition_hypers_column_divide,
            'row_assignments':
                lambda : self.transition_row_assignments(
                    outputs=outputs, rowids=rowids),
            'view_assignments':
                lambda : self.transition_view_assignments(outputs=outputs),
        }
        if kernels is None:
            kernels = kernel_lookup.keys()
        kernel_funcs = [kernel_lookup[k] for k in kernels]
        transition_generic(kernel_funcs, N, S, progress)

    def transition_hypers_distributions(self, outputs=None):
        distribution_cgpms = get_distribution_cgpms(self.crosscat, outputs)
        for output, cgpms in distribution_cgpms.iteritems():
            transition_hypers(cgpms, self.grids[output], self.crosscat.rng)

    def transition_hypers_row_divide(self, outputs=None):
        crp_cgpms = get_row_divide_cgpms(self.crosscat, outputs)
        for _output, cgpms in crp_cgpms.iteritems():
            transition_hypers(cgpms, self.grids['row_divide'], self.crosscat.rng)

    def transition_hypers_column_divide(self):
        pass

    def transition_hyper_grids_distribution(self, outputs=None):
        distribution_cgpms = get_distribution_cgpms(self.crosscat, outputs)
        for output, cgpms in distribution_cgpms.iteritems():
            self.grids[output] = transition_hyper_grids(cgpms)

    def transition_hyper_grids_row_divide(self):
        crp_cgpm = self.crosscat.cgpms[0].cgpm_row_divide
        self.grids['row_divide'] = transition_hyper_grids([crp_cgpm])

    def transition_hyper_grids_column_divide(self):
        pass

    def transition_row_assignments(self, outputs=None, rowids=None):
        rowids = set(rowids) if rowids is not None else set()
        views = get_row_mixture_cgpms(self.crosscat, outputs)
        for view in views:
            rowids_view = get_rowids_transition(view, rowids)
            for rowid in self.crosscat.rng.permutation(rowids_view):
                transition_rows(view, rowid, self.crosscat.rng)

    def transition_view_assignments(self, outputs=None):
        assert not (self.Cd or self.Ci)
        outputs = outputs or get_distribution_outputs(self.crosscat)
        for output in outputs:
            self.crosscat = \
                transition_cgpm_view_assigments(self.crosscat, [output])

    # Stochastic mutation (cpp synthesizer).

    def transition_row_assignments_cpp(self, N=None, S=None, rowids=None):
        seed = self.crosscat.rng.randint(2**32-1)
        kernels = [
            'row_partition_hyperparameters',
            'column_hyperparameters',
            'row_partition_hyperparameters',
            'row_partition_assignments',
        ]
        self.crosscat = transition_cpp(self.crosscat, N=N, S=S,
            Cd=self.Cd, Ci=self.Ci, kernels=kernels, rowids=rowids, seed=seed)
        self.transition_hypers_distributions()
        self.transition_hypers_row_divide()

    def transition_view_assignments_cpp(self, N=None, S=None, outputs=None):
        seed = self.crosscat.rng.randint(2**32-1)
        kernels = [
            'column_partition_assignments',
            'column_partition_hyperparameter',
        ]
        self.crosscat = transition_cpp(self.crosscat, N=N, S=S,
            Cd=self.Cd, Ci=self.Ci, kernels=kernels, cols=outputs, seed=seed)
        self.transition_hypers_distributions()
        self.transition_hypers_row_divide()

    def transition_structure_cpp(self, N=None, S=None, outputs=None,
            progress=None):
        seed = self.crosscat.rng.randint(2**32-1)
        self.crosscat = transition_cpp(self.crosscat, N=N, S=S, cols=outputs,
            Cd=self.Cd, Ci=self.Ci, seed=seed, progress=progress)
        self.transition_hypers_distributions()
        self.transition_hypers_row_divide()

    # Deterministic mutation.

    def set_hypers_distribution(self, output, hypers):
        distribution_cgpms = get_distribution_cgpms(self.crosscat, [output])
        set_hypers(distribution_cgpms[output], hypers)

    def set_hypers_row_divide(self, output, hypers):
        crp_cgpm = get_row_divide_cgpms(self.crosscat, [output])
        set_hypers(crp_cgpm.values()[0], hypers)

    def set_rowid_component(self, outputs, rowid0, rowid1):
        views = get_row_mixture_cgpms(self.crosscat, outputs)
        for view in views:
            set_rowid_component(view, rowid0, rowid1)

    def set_view_assignment(self, output0, output1):
        assert not (self.Cd or self.Ci)
        self.crosscat = set_cgpm_view_assignment(self.crosscat, output0, output1)
