# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pytest

from cgpm.utils.general import get_prng

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

from cgpm2.walks import get_cgpms_by_output_index

# Test utilities.

def plot_clustered_data(data, assignments):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for cluster, color in enumerate(['r','k','g','y','b']):
        rowids = [rowid for rowid, z in assignments.iteritems() if z == cluster]
        ax.scatter(data[rowids,0], data[rowids,1], color=color)
    return fig, ax

def plot_sampled_data(samples):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    clusters = set([s[2] for s in samples])
    for cluster, color in zip(clusters, ['r','k','b','g']):
        xs = [sample[0] for sample in samples if sample[2] == cluster]
        ys = [sample[1] for sample in samples if sample[2] == cluster]
        ax.scatter(xs, ys, color=color)
    return fig, ax

def make_bivariate_two_clusters(prng):
    data0 = prng.normal(loc=0, scale=.5, size=60)
    data1 = prng.normal(loc=7, scale=.5, size=60)
    data = np.row_stack((
        np.reshape(data0, (30, 2)),
        np.reshape(data1, (30, 2)),
    ))
    data[0,0] = np.nan
    data[50,1] = np.nan
    data[10,0] = np.nan
    data[30,1] = np.nan
    data[33,0] = np.nan
    data[33,1] = np.nan
    return data

def get_table_counts(assignments):
    tables = set(assignments.values())
    return {t: sum(a==t for a in assignments.itervalues()) for t in tables}

def check_clustered_data(assignments):
    counts = get_table_counts(assignments)
    assert len(counts) == 2
    assert all(28<c<32 for c in counts.itervalues())
    assert sum(assignments[i]==assignments[i-1] for i in xrange(1, 30)) > 26
    assert sum(assignments[i]==assignments[i-1] for i in xrange(31, 60)) > 26

def check_sampled_data(samples, means, tolerance, atleast):
    for mean in means:
        in_range = lambda s: mean-tolerance <= s <= mean+tolerance
        subsample = [s for s in samples if in_range(s[0]) and in_range(s[1])]
        assert atleast <= len(subsample)

def observe_data(cgpm, data):
    for rowid, row in enumerate(data[:10]):
        cgpm.observe(rowid, {0: row[0]})
        cgpm.observe(rowid, {1: row[1]})
    for rowid, row in enumerate(data[10:]):
        with pytest.raises(ValueError):
            cgpm.observe(rowid, {0: row[0], 1: row[1]})
        cgpm.observe(10 + rowid, {0: row[0], 1: row[1]})
    return cgpm

# Tests for mixture models.

def run_mixture_test(cgpm_mixture, prng):
    data = make_bivariate_two_clusters(prng)
    # Observe data.
    cgpm_mixture = observe_data(cgpm_mixture, data)
    # Run inference.
    cgpms = {
        0  : get_cgpms_by_output_index(cgpm_mixture, 0),
        1  : get_cgpms_by_output_index(cgpm_mixture, 1),
        2  : get_cgpms_by_output_index(cgpm_mixture, 2),
    }
    grids = {
        0 : transition_hyper_grids(cgpms[0], 30),
        1 : transition_hyper_grids(cgpms[1], 30),
        2 : transition_hyper_grids(cgpms[2], 30),
    }
    for _step in xrange(500):
        rowids = prng.permutation(range(len(data)))
        for rowid in rowids:
            transition_rows(cgpm_mixture, rowid, prng)
        for output in cgpm_mixture.outputs:
            transition_hypers(cgpms[output], grids[output], prng)
    # Test clustered data.
    assignments = cgpm_mixture.cgpm_row_divide.data
    check_clustered_data(assignments)
    # Test simulated data.
    samples = cgpm_mixture.simulate(None, [0,1,2], N=150)
    check_sampled_data(samples, [0, 7], 3, 60)

def test_finite_mixture_two_component__ci_():
    prng = get_prng(2)
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=Categorical([2], [], distargs={'k':2}, rng=prng),
        cgpm_components=[
            Product([Normal([0], [], rng=prng), Normal([1], [], rng=prng)],
                rng=prng),
            Product([Normal([0], [], rng=prng), Normal([1], [], rng=prng)],
                rng=prng),
        ],
        rng=prng)
    run_mixture_test(finite_mixture, prng)

def test_flexible_mixture_two_component__ci_():
    prng = get_prng(2)
    flexible_mixture = FlexibleRowMixture(
        cgpm_row_divide=CRP([2], [], rng=prng),
        cgpm_components_base=Product([
            Normal([0], [], rng=prng),
            Normal([1], [], rng=prng),
        ], rng=prng),
        rng=prng)
    run_mixture_test(flexible_mixture, prng)

# Tests for CrossCat.

def get_crosscat(prng):
    view = FlexibleRowMixture(
        cgpm_row_divide=CRP([2], [], rng=prng),
        cgpm_components_base=Product([
            Normal([0], [], rng=prng),
            Normal([1], [], rng=prng),
        ], rng=prng),
        rng=prng)
    return Product(cgpms=[view], rng=prng)

def run_crosscat_test(crosscat, func_inference, prng):
    data = make_bivariate_two_clusters(prng)
    crosscat = observe_data(crosscat, data)
    # Run inference.
    synthesizer = func_inference(crosscat)
    # Test one view.
    assert len(synthesizer.crosscat.cgpms) == 1
    # Test clustered data.
    assignments = synthesizer.crosscat.cgpms[0].cgpm_row_divide.data
    check_clustered_data(assignments)
    # Test simulated data.
    samples = synthesizer.crosscat.simulate(None, [0, 1], N=150)
    check_sampled_data(samples, [0, 7], 60, 3)

def test_crosscat_two_component_no_view__ci_():
    prng = get_prng(10)
    crosscat = get_crosscat(prng)
    def func_inference(crosscat):
        synthesizer = GibbsCrossCat(crosscat)
        for _step in xrange(500):
            synthesizer.transition_row_assignments()
            synthesizer.transition_hypers_distributions()
            synthesizer.transition_hypers_row_divide()
        return synthesizer
    run_crosscat_test(crosscat, func_inference, prng)

def test_crosscat_two_component_view__ci_():
    prng = get_prng(10)
    crosscat = get_crosscat(prng)
    def func_inference(crosscat):
        synthesizer = GibbsCrossCat(crosscat)
        for _step in xrange(540):
            synthesizer.transition_row_assignments()
            synthesizer.transition_hypers_row_divide()
            synthesizer.transition_hypers_distributions()
            synthesizer.transition_view_assignments()
        return synthesizer
    run_crosscat_test(crosscat, func_inference, prng)

def test_crosscat_two_component_cpp__ci_():
    prng = get_prng(10)
    crosscat = get_crosscat(prng)
    def func_inference(crosscat):
        synthesizer = GibbsCrossCat(crosscat)
        for _step in xrange(540):
            synthesizer = GibbsCrossCat(crosscat)
            synthesizer.transition_structure_cpp(N=1000)
            synthesizer.transition_hypers_distributions()
            synthesizer.transition_hypers_row_divide()
            return synthesizer
    run_crosscat_test(crosscat, func_inference, prng)

# Test for crosscat with a nominal variable.

def test_crosscat_two_component_nominal__ci_():
    prng = get_prng(10)
    # Build CGPM with adversarial initialization.
    crosscat = Product([
        FlexibleRowMixture(
            cgpm_row_divide=CRP([-1], [], rng=prng),
            cgpm_components_base=Product([
                Normal([0], [], rng=prng),
            ], rng=prng),
            rng=prng),
        FlexibleRowMixture(
            cgpm_row_divide=CRP([-2], [], rng=prng),
            cgpm_components_base=Product([
                Normal([1], [], rng=prng),
                Categorical([50], [], distargs={'k':4}, rng=prng),
            ], rng=prng),
            rng=prng),
    ], rng=prng,)
    # Fetch data and add a nominal variable.
    data_xy = make_bivariate_two_clusters(prng)
    data_z = np.zeros(len(data_xy))
    data_z[:15] = 0
    data_z[15:30] = 1
    data_z[30:45] = 2
    data_z[45:60] = 3
    data = np.column_stack((data_xy, data_z))
    # Observe.
    for rowid, row in enumerate(data):
        crosscat.observe(rowid, {0: row[0], 1: row[1], 50:row[2]})
    # Run inference.
    synthesizer = GibbsCrossCat(crosscat)
    synthesizer.transition(N=50, progress=False)
    synthesizer.transition(N=100,
            kernels=['hypers_distributions','hypers_row_divide'],
            progress=False)

    # Assert views are merged into one.
    assert len(synthesizer.crosscat.cgpms) == 1
    crp_output = synthesizer.crosscat.cgpms[0].cgpm_row_divide.outputs[0]

    # Check joint samples for all nominals.
    samples = synthesizer.crosscat.simulate(None, [crp_output,0,1,50], N=250)
    check_sampled_data(samples, [0, 7], 3, 110)
    # Check joint samples for nominals [0, 2].
    samples_a = [s for s in samples if s[50] in [0,2]]
    check_sampled_data(samples_a, [0, 7], 3, 45)
    # Check joint samples for nominals [1, 3].
    samples_b = [s for s in samples if s[50] in [1,3]]
    check_sampled_data(samples_b, [0, 7], 3, 45)

    # Check conditional samples in correct quadrants.
    means = {0:0, 1:0, 2:7, 3:7}
    for z in [0, 1, 2, 3]:
        samples = synthesizer.crosscat.simulate(None, [0, 1], {50:z}, N=100)
        check_sampled_data(samples, [means[z]], 3, 90)
