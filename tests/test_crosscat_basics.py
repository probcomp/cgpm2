# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pytest

from cgpm2.categorical import Categorical
from cgpm2.crp import CRP
from cgpm2.normal import Normal
from cgpm2.poisson import Poisson

from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.walks import add_cgpm
from cgpm2.walks import remove_cgpm

from cgpm2.transition_views import get_cgpm_view_proposals_existing
from cgpm2.transition_views import get_cgpm_view_proposals_singleton
from cgpm2.transition_views import get_dataset

from cgpm2.tests.utils import gen_data_table
from cgpm2.utils import get_prng

def get_crosscat(prng):
    view0 = FlexibleRowMixture(
        cgpm_row_divide=CRP([-1], [], rng=prng),
        cgpm_components_base=Product([
            Normal([0], [], rng=prng),
            Normal([1], [], rng=prng),
        ], rng=prng),
        rng=prng)
    view1 = FlexibleRowMixture(
        cgpm_row_divide=CRP([-2], [], rng=prng),
        cgpm_components_base=Product([
            Poisson([2], [], rng=prng),
            Normal([3], [], rng=prng),
            Normal([4], [], rng=prng),
        ], rng=prng),
        rng=prng)
    view2 = FlexibleRowMixture(
        cgpm_row_divide=CRP([-3], [], rng=prng),
        cgpm_components_base=Product([
            Categorical([5], [], distargs={'k':4}, rng=prng),
        ], rng=prng),
        rng=prng)
    return Product([view0, view1, view2], rng=prng)

def populate_crosscat(crosscat, prng):
    X, Zv, Zrv = gen_data_table(
        n_rows=10,
        view_weights=[.4, .6],
        cluster_weights=[[.3,.4,.3],[.5,.5]],
        cctypes=['normal','normal','poisson','normal','normal','categorical'],
        distargs=[None, None, None, None, None, {'k':4}],
        separation=[0.99]*6,
        rng=prng)
    X[0,1] = X[3,1] = float('nan')
    dataset = np.transpose(X)
    for rowid, row in enumerate(dataset):
        observation = {c:v for c,v in enumerate(row)}
        crosscat.observe(rowid, observation)
    return crosscat

def test_crosscat_add_remove():
    prng = get_prng(2)
    crosscat =  get_crosscat(prng)
    infinite_mixture4 = FlexibleRowMixture(
        cgpm_row_divide=CRP([-4], [], rng=prng),
        cgpm_components_base=Product([
            Categorical([6], [], distargs={'k':4}, rng=prng),
        ], rng=prng),
        rng=prng)
    crosscat = add_cgpm(crosscat, infinite_mixture4)
    assert crosscat.outputs == [-1, 0, 1, -2, 2, 3, 4, -3, 5, -4, 6]
    crosscat = remove_cgpm(crosscat, -1)
    assert crosscat.outputs == [-2, 2, 3, 4, -3, 5, -4, 6]
    crosscat = remove_cgpm(crosscat, 5)
    assert crosscat.outputs == [-2, 2, 3, 4, -4, 6]

def test_get_view_proposals():
    prng = get_prng(2)
    crosscat = get_crosscat(prng)
    # Get block proposals of outputs [0,1] into all three views.
    proposals = get_cgpm_view_proposals_existing(crosscat, [0,1])
    assert len(proposals) == 3
    assert proposals[0].outputs == crosscat.cgpms[0].outputs
    assert proposals[1].outputs == crosscat.cgpms[1].outputs + [0, 1]
    assert proposals[2].outputs == crosscat.cgpms[2].outputs + [0, 1]
    # Get proposals of outputs [0,1] into 2 singleton views.
    proposals = get_cgpm_view_proposals_singleton(crosscat, [0,1], 2)
    assert len(proposals) == 2
    assert proposals[0].outputs[1:] == [0,1]
    assert proposals[1].outputs[1:] == [0,1]
    # Fail to get proposals for outputs in different views.
    with pytest.raises(Exception):
        proposals = get_cgpm_view_proposals_existing(crosscat, [0,2])

def test_logpdf_basic():
    prng = get_prng(2)
    crosscat = get_crosscat(prng)
    crosscat = populate_crosscat(crosscat, prng)
    for _rowid, row in get_dataset(crosscat, 0):
        logp = crosscat.logpdf(None, row)
        if np.isnan(row.values()[0]):
            assert np.allclose(logp, 0)
        else:
            assert logp < 0
