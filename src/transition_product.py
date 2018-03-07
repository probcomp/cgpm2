# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools
import numpy as np

from cgpm.utils.general import log_pflip

from .crp import CRP
from .distribution import DistributionCGPM
from .flexible_rowmix import FlexibleRowMixture
from .product import Product

from .walks import add_cgpm
from .walks import get_cgpms_by_output_index
from .walks import remove_cgpm


def get_dataset(cgpm, output):
    cgpms = get_cgpms_by_output_index(cgpm, output)
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    return [
        (rowid, {c:v for c, v in zip(cgpm.outputs, np.atleast_1d(row))})
        for cgpm in cgpms
        for rowid, row in cgpm.data.iteritems()
    ]

def get_cgpm_base(cgpm, output):
    cgpms = get_cgpms_by_output_index(cgpm, output)
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    empty_cgpms = [cgpm for cgpm in cgpms if cgpm.N == 0]
    return empty_cgpms[0]

def get_cgpm_data_logp(view, output):
    cgpms = get_cgpms_by_output_index(view, output)
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    return sum(c.logpdf_score() for c in cgpms)

def get_cgpm_current_view_index(crosscat, outputs):
    views = [i for output in outputs for i, view in enumerate(crosscat.cgpms)
        if output in view.outputs]
    assert all(view == views[0] for view in views)
    return views[0]

def get_cgpm_view_proposal_existing_one(view, cgpm_base_list, dataset_list):
    proposal = reduce(add_cgpm, cgpm_base_list, view)
    for dataset in dataset_list:
        for rowid, observation in dataset:
            proposal.incorporate(rowid, observation)
    return proposal

def get_cgpm_view_proposal_singleton_one(crosscat, cgpm_base_list, dataset_list):
    crp_output = crosscat.rng.randint(2**32-1)
    cgpm_row_divide = CRP([crp_output], [], rng=crosscat.rng)
    view = FlexibleRowMixture(cgpm_row_divide, Product(cgpm_base_list),
        rng=crosscat.rng)
    for dataset in dataset_list:
        for rowid, observation in dataset:
            view.incorporate(rowid, observation)
    return view

def get_cgpm_view_proposals_existing(crosscat, outputs):
    view_current = get_cgpm_current_view_index(crosscat, outputs)
    cgpm_base_list = [get_cgpm_base(crosscat, output) for output in outputs]
    dataset_list = [get_dataset(crosscat, output) for output in outputs]
    return [
        get_cgpm_view_proposal_existing_one(view, cgpm_base_list, dataset_list)
            if (i is not view_current) else view
        for i, view in enumerate(crosscat.cgpms)
    ]

def get_cgpm_view_proposals_singleton(crosscat, outputs, aux):
    cgpm_base_list = [get_cgpm_base(crosscat, output) for output in outputs]
    dataset_list = [get_dataset(crosscat, output) for output in outputs]
    return [
        get_cgpm_view_proposal_singleton_one(
            crosscat, cgpm_base_list, dataset_list)
        for _m in xrange(aux)
    ]

def get_cgpm_view_proposals(crosscat, outputs, aux):
    proposals_existing = get_cgpm_view_proposals_existing(crosscat, outputs)
    proposals_singleton = \
        get_cgpm_view_proposals_singleton(crosscat, outputs, aux)
    return list(itertools.chain(proposals_existing, proposals_singleton))

def transition_cgpm_view_assigments(crosscat, outputs, aux=1):
    view_proposals = get_cgpm_view_proposals(crosscat, outputs, aux)
    view_logps = [
        sum(get_cgpm_data_logp(view, output) for output in outputs)
        for view in view_proposals
    ]
    num_views = len(crosscat.cgpms)
    view_idx_current = get_cgpm_current_view_index(crosscat, outputs)
    view_idx_sampled = log_pflip(view_logps, rng=crosscat.rng)
    view_current = crosscat.cgpms[view_idx_current]
    view_sampled = view_proposals[view_idx_sampled]
    view_sampled_original = crosscat.cgpms[view_idx_sampled] \
        if view_idx_sampled < num_views else None
    # If the current and sampled views are identical then exit.
    if view_idx_current == view_idx_sampled:
        return crosscat
    # Remove outputs from current view, and current view from CrossCat.
    # Only restore current view if it still has CGPMs.
    view_current_prime = reduce(remove_cgpm, outputs, view_current)
    crosscat = remove_cgpm(crosscat, view_current.outputs[0])
    if len(view_current_prime.outputs) > 1:
        crosscat = add_cgpm(crosscat, view_current_prime)
    # If the sampled view is a singleton, add it directly to CrossCat.
    # Otherwise, remove the original sampled view and add the new sampled view
    # that came from view_proposals.
    if num_views <= view_idx_sampled:
        crosscat = add_cgpm(crosscat, view_sampled)
    else:
        assert view_sampled_original.outputs[0] == view_sampled.outputs[0]
        crosscat = remove_cgpm(crosscat, view_sampled_original.outputs[0])
        crosscat = add_cgpm(crosscat, view_sampled)
    return crosscat
