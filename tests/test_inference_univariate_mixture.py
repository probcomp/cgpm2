# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import pytest
import numpy as np

from cgpm2.categorical import Categorical
from cgpm2.crp import CRP
from cgpm2.normal import Normal

from cgpm2.finite_rowmix import FiniteRowMixture
from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.transition_crosscat import GibbsCrossCat
from cgpm2.transition_hypers import transition_hyper_grids
from cgpm2.transition_hypers import transition_hypers
from cgpm2.transition_rows import transition_rows

from cgpm2.utils import get_prng
from cgpm2.walks import get_cgpms_by_output_index

# Test utilities.

def make_univariate_three_clusters(prng):
    data0 = prng.normal(loc=0, scale=2, size=20)
    data1 = prng.normal(loc=30, scale=1, size=20)
    data2 = prng.normal(loc=-30, scale=1, size=20)
    return np.concatenate((data0, data1, data2))

def plot_clustered_data(data, assignments):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for cluster, _color in enumerate(['r','k','g','y','b']):
        rowids = [rowid for rowid, z in assignments.iteritems() if z == cluster]
        ax.hist(data[rowids])
    return fig, ax

def plot_sampled_data(samples):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    clusters = set([s[1] for s in samples])
    for cluster, _color in zip(clusters, ['r','k','b','g']):
        xs = [sample[0] for sample in samples if sample[1] == cluster]
        ax.hist(xs)
    return fig, ax

def get_table_counts(assignments):
    tables = set(assignments.values())
    return {t: sum(a==t for a in assignments.itervalues()) for t in tables}

def check_clustered_data(assignments):
    counts = get_table_counts(assignments)
    assert len(counts) == 3
    assert all(c==20 for c in counts.itervalues())
    assert all(assignments[i]==assignments[i-1] for i in xrange(1, 20))
    assert all(assignments[i]==assignments[i-1] for i in xrange(21, 40))
    assert all(assignments[i]==assignments[i-1] for i in xrange(41, 60))

def check_sampled_data(samples, id_x=0, id_z=1):
    for mean in [0, 30, -30]:
        subsample = [s for s in samples if mean-10<=s[id_x]<=mean+10]
        assert 40 <= len(subsample)
        assert all(s[id_z]==subsample[0][id_z] for s in subsample)

# Tests for mixture models.

def run_mixture_test(cgpm_mixture, integration, prng):
    data = make_univariate_three_clusters(prng)
    for rowid, value in enumerate(data):
        cgpm_mixture.observe(rowid, {0: value})
    # Run inference.
    cgpms = {
        0  : get_cgpms_by_output_index(cgpm_mixture, 0),
        1  : get_cgpms_by_output_index(cgpm_mixture, 1),
    }
    grids = {
        0 : transition_hyper_grids(cgpms[0], 30),
        1 : transition_hyper_grids(cgpms[1], 30)
    }
    n_steps = 500 if integration else 1
    for _step in xrange(n_steps):
        rowids = prng.permutation(range(len(data)))
        for rowid in rowids:
            transition_rows(cgpm_mixture, rowid, prng)
        for output in cgpm_mixture.outputs:
            transition_hypers(cgpms[output], grids[output], prng)
    # Test clustered data.
    assignments = cgpm_mixture.cgpm_row_divide.data
    not integration or check_clustered_data(assignments)
    # Test simulated data.
    samples = cgpm_mixture.simulate(None, [0,1], N=150)
    not integration or check_sampled_data(samples)


def test_finite_mixture_three_component__ci_():
    prng = get_prng(2)
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=Categorical([1], [], distargs={'k':3}, rng=prng),
        cgpm_components=[
            Normal([0], [], rng=prng),
            Normal([0], [], rng=prng),
            Normal([0], [], rng=prng)
        ],
        rng=prng,
    )
    integration = pytest.config.getoption('--integration')
    run_mixture_test(finite_mixture, integration, prng)

def test_flexible_mixture_three_component__ci_():
    prng = get_prng(2)
    flexible_mixture = FlexibleRowMixture(
        cgpm_row_divide=CRP([1], [], rng=prng),
        cgpm_components_base=Normal([0], [], rng=prng),
        rng=prng
    )
    integration = pytest.config.getoption('--integration')
    run_mixture_test(flexible_mixture, integration, prng)

# Tests for CrossCat.

def run_crosscat_test(crosscat, func_inference, integration, prng):
    data = make_univariate_three_clusters(prng)
    for rowid, value in enumerate(data):
        crosscat.observe(rowid, {0: value})
    # Run inference.
    synthesizer = func_inference(crosscat)
    # Test clustered data.
    assignments = synthesizer.crosscat.cgpms[0].cgpm_row_divide.data
    not integration or check_clustered_data(assignments)
    # Test simulated data.
    id_z = synthesizer.crosscat.outputs[0]
    samples = synthesizer.crosscat.simulate(None, [0, id_z], N=150)
    not integration or check_sampled_data(samples, id_z=id_z)

def test_crosscat_three_component__ci_():
    prng = get_prng(10)
    integration = pytest.config.getoption('--integration')
    view = FlexibleRowMixture(
        cgpm_row_divide=CRP([1], [], rng=prng),
        cgpm_components_base=Product(cgpms=[Normal([0], [], rng=prng)], rng=prng),
        rng=prng)
    crosscat = Product(cgpms=[view], rng=prng)
    def func_inference(crosscat):
        synthesizer = GibbsCrossCat(crosscat)
        n_step = 500 if integration else 1
        for _step in xrange(n_step):
            synthesizer.transition_row_assignments()
            synthesizer.transition_hypers_distributions()
            synthesizer.transition_hypers_row_divide()
        return synthesizer
    run_crosscat_test(crosscat, func_inference, integration, prng)

def test_crosscat_three_component_cpp__ci_():
    prng = get_prng(12)
    integration = pytest.config.getoption('--integration')
    view = FlexibleRowMixture(
        cgpm_row_divide=CRP([1], [], rng=prng),
        cgpm_components_base=Product(cgpms=[Normal([0], [], rng=prng)], rng=prng),
        rng=prng)
    crosscat = Product(cgpms=[view], rng=prng)
    def func_inference(crosscat):
        n_step = 1000 if integration else 1
        synthesizer = GibbsCrossCat(crosscat)
        synthesizer.transition_structure_cpp(N=n_step)
        synthesizer.transition_hypers_distributions()
        synthesizer.transition_hypers_row_divide()
        return synthesizer
    run_crosscat_test(crosscat, func_inference, integration, prng)
