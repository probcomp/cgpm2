# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pytest

from cgpm.utils.general import get_prng

from cgpm2.categorical import Categorical
from cgpm2.crp import CRP
from cgpm2.normal import Normal
from cgpm2.poisson import Poisson

from cgpm2.chain import Chain
from cgpm2.finite_array import FiniteArray
from cgpm2.finite_rowmix import FiniteRowMixture
from cgpm2.flexible_array import FlexibleArray
from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product


def test_simple_product():
    prng = get_prng(2)
    column0 = Normal([0], [], rng=prng)
    column1 = Normal([1], [], rng=prng)
    column2 = Categorical([2], [], distargs={'k':4}, rng=prng)
    product = Product([column0, column1, column2], prng)
    assert product.outputs == [0,1,2]
    assert product.inputs == []
    sample = product.simulate(None, [1,2,0])
    assert set(sample.keys()) == set([1, 2, 0])
    logp = product.logpdf(None, sample)
    assert logp < 0

def test_simple_product_finite_array():
    prng = get_prng(2)
    array0_component0 = Normal([0], [], hypers={'m':100}, rng=prng)
    array0_component1 = Normal([0], [], hypers={'m':-100}, rng=prng)
    array0_component2 = Normal([0], [], hypers={'m':0}, rng=prng)
    indexer_0 = 128
    cgpm_array_0 = FiniteArray(
        cgpms=[array0_component0, array0_component1, array0_component2],
        indexer=indexer_0,
        rng=prng)
    with pytest.raises(Exception):
        # Missing indexer_0 as a required input.
        cgpm_array_0.simulate(None, [0])
    array1_component0 = Normal([1], [], hypers={'m':1000}, rng=prng)
    array1_component1 = Normal([1], [], hypers={'m':-1000}, rng=prng)
    array1_component2 = Normal([1], [], hypers={'m':50}, rng=prng)
    indexer_1 = 129
    cgpm_array_1 = FiniteArray(
        cgpms=[array1_component0, array1_component1, array1_component2],
        indexer=indexer_1,
        rng=prng)
    product = Product([cgpm_array_0, cgpm_array_1], prng)
    assert product.outputs == [0, 1]
    assert product.inputs == [indexer_0, indexer_1]
    with pytest.raises(Exception):
        # Missing indexer_0.
        product.simulate(None, [0,1], inputs={indexer_1: 0})
    with pytest.raises(Exception):
        # Missing indexer_1.
        product.simulate(None, [0,1], inputs={indexer_0: 1})
    # Should work, since output 1 is not being queried.
    product.simulate(None, [0], inputs={indexer_0: 1})
    # Sampling from correct components.
    sample = product.simulate(None, [0,1], inputs={indexer_0:1, indexer_1:0})
    assert abs(-100 - sample[0]) < 10
    assert abs(1000 - sample[1]) < 10
    logp = product.logpdf(None, sample, inputs={indexer_0:1, indexer_1:0})
    assert np.allclose(logp,
        array0_component1.logpdf(None, {0: sample[0]})
            + array1_component0.logpdf(None, {1: sample[1]}))

def test_flexible_array_observe():
    prng = get_prng(2)
    component0 = Normal([0], [], hypers={'m':0}, rng=prng)
    indexer_0 = 128
    cgpm_array_1 = FlexibleArray(
        cgpm_base=component0,
        indexer=indexer_0,
        rng=prng)
    # Make observations into cell 10.
    cgpm_array_1.observe(100, {0:10000}, inputs={indexer_0:10})
    cgpm_array_1.observe(101, {0:10000}, inputs={indexer_0:10})
    cgpm_array_1.observe(102, {0:10000}, inputs={indexer_0:10})
    cgpm_array_1.observe(103, {0:10000}, inputs={indexer_0:10})
    cgpm_array_1.observe(104, {0:10000}, inputs={indexer_0:10})
    cgpm_array_1.observe(105, {0:10000}, inputs={indexer_0:10})
    # Simulate from cell 10 (high mean).
    samples = cgpm_array_1.simulate(None, [0], inputs={indexer_0: 10}, N=100)
    assert len([s for s in samples if s[0] > 1000]) > int(.9*len(samples))
    # Simulate from cell 0 (zero mean).
    samples = cgpm_array_1.simulate(None, [0], inputs={indexer_0: 0}, N=100)
    assert len([s for s in samples if s[0] > 1000]) == 0
    assert len([s for s in samples if -10 < s[0] < 10]) > int(.9*len(samples))

def test_finite_mixture_probabilities():
    prng = get_prng(2)
    component0 = Normal([0], [], hypers={'m':100}, rng=prng)
    component1 = Normal([0], [], hypers={'m':-100}, rng=prng)
    component2 = Normal([0], [], hypers={'m':0}, rng=prng)
    cgpm_row_divide = Categorical([1], [], distargs={'k':3}, rng=prng)
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components=[component0, component1, component2],
        rng=prng)
    # Make observations into component0 at prior mean.
    finite_mixture.observe(1, {0:100, 1:0})
    finite_mixture.observe(2, {0:100, 1:0})
    finite_mixture.observe(3, {0:100, 1:0})
    assert finite_mixture.rowid_to_component == {1:0, 2:0, 3:0}
    # Sample of cluster assignments given data is 100.
    samples = finite_mixture.simulate(None, [1], constraints={0:100}, N=20)
    assert len([s for s in samples if s[1]==0]) > int(0.9*len(samples))
    # Compute likelihood of data given cluster assignment.
    lp0 = finite_mixture.logpdf(None, {0:100}, constraints={1:0})
    lp1 = finite_mixture.logpdf(None, {0:100}, constraints={1:1})
    lp2 = finite_mixture.logpdf(None, {0:100}, constraints={1:2})
    assert lp1 < lp0
    assert lp2 < lp0
    assert lp1 < lp2
    # Compute posterior probabilities of cluster assignment
    lp0 = finite_mixture.logpdf(None, {1:0}, constraints={0:100})
    lp1 = finite_mixture.logpdf(None, {1:1}, constraints={0:100})
    lp2 = finite_mixture.logpdf(None, {1:2}, constraints={0:100})
    assert lp1 < lp0
    assert lp2 < lp0
    assert lp1 < lp2
    # Constrained cluster has zero density.
    with pytest.raises(ValueError):
        lp2 = finite_mixture.logpdf(None, {0:100}, constraints={1:-1})
    with pytest.raises(ValueError):
        lp2 = finite_mixture.logpdf(None, {0:100}, constraints={1:20})
    with pytest.raises(ValueError):
        lp2 = finite_mixture.simulate(None, [0], constraints={1:-1})
    with pytest.raises(ValueError):
        lp2 = finite_mixture.simulate(None, [0], constraints={1:20})

def test_product_mixture_constraints():
    prng = get_prng(2)
    component0 = Product([
        Normal([0], [], hypers={'m':1000}, rng=prng),
        Normal([1], [], hypers={'m':1000}, rng=prng)
        ], rng=prng)
    component1 = Product([
        Normal([0], [], hypers={'m':-1000}, rng=prng),
        Normal([1], [], hypers={'m':-1000}, rng=prng)
        ], rng=prng)
    component2 = Product([
        Normal([0], [], hypers={'m':0}, rng=prng),
        Normal([1], [], hypers={'m':0}, rng=prng)
        ], rng=prng)
    cgpm_row_divide = Categorical([2], [], distargs={'k':3}, rng=prng)
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components=[component0, component1, component2],
        rng=prng)
    def run_mixture_tests(mixture):
        N = 100
        # Simulate from component 1.
        samples = mixture.simulate(None, [0], constraints={2:1}, N=N)
        assert len([s for s in samples if -1100 < s[0] < -900]) > int(.9*N)
        # Simulate from random components.
        samples = mixture.simulate(None, [0], N=N)
        assert len([s for s in samples if -900 < s[0] < -1100]) < int(.33*N)
        # Simulate (implicitly) from component 0.
        samples = mixture.simulate(None, [1,2], constraints={0:1000}, N=N)
        assert len([s for s in samples if 900 < s[1] < 1100]) > int(.9*N)
        assert len([s for s in samples if s[2] == 0]) == N
    # Run tests on finite_mixture.
    run_mixture_tests(finite_mixture)
    # Run tests after to/from metadata conversion.
    metadata = finite_mixture.to_metadata()
    finite_mixture2 = FiniteRowMixture.from_metadata(metadata, prng)
    run_mixture_tests(finite_mixture2)

def test_simple_product_as_chain():
    prng = get_prng(2)
    component0 = Chain([
        Poisson([0], [], hypers={'a': 10, 'b': 1}, rng=prng),
        Normal([1], [], hypers={'m':100}, rng=prng)
        ],
        rng=prng)
    cgpm_row_divide = CRP([2], [], rng=prng)
    infinite_mixture = FlexibleRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components_base=component0,
        rng=prng)
    assert infinite_mixture.cgpm_row_divide.support() == [0]
    # Test logpdf identities.
    lp0 = infinite_mixture.logpdf(None, {0:1})
    assert lp0 < 0
    lp1 = infinite_mixture.logpdf(None, {0:1, 2:0})
    assert np.allclose(lp0, lp1)
    lp2 = infinite_mixture.logpdf(None, {0:1, 2:1})
    assert lp2 == -float('inf')
    # Add an observation.
    infinite_mixture.observe(0, {1:100})
    lp0 = infinite_mixture.logpdf(None, {1:100, 2:0}, constraints={0:1})
    lp1 = infinite_mixture.logpdf(None, {1:100, 2:1}, constraints={0:1})
    lp2 = infinite_mixture.logpdf(None, {1:100, 2:2}, constraints={0:1})
    assert lp1 < lp0
    assert lp2 == float('-inf')
    # Remove observation.
    observation = infinite_mixture.unobserve(0)
    assert observation == ({1:100, 2:0}, {})
    # Remove observation again.
    with pytest.raises(Exception):
        infinite_mixture.unobserve(0)
    # Add more observations.
    infinite_mixture.observe(0, {1:100})
    infinite_mixture.observe(1, {1:300})
    infinite_mixture.observe(2, {0:2})
    # Constrained cluster has zero density.
    with pytest.raises(ValueError):
        infinite_mixture.logpdf(None, {0:1}, constraints={2:10})
    with pytest.raises(ValueError):
        infinite_mixture.logpdf(None, {0:1}, constraints={2:10})
    # Convert to/from metadata and assert unobserves return correct data.
    metadata = infinite_mixture.to_metadata()
    infinite_mixture2 = FlexibleRowMixture.from_metadata(metadata, prng)
    assert infinite_mixture2.unobserve(0) == \
        ({1:100, 2: infinite_mixture.cgpm_row_divide.data[0]}, {})
    assert infinite_mixture2.unobserve(1) == \
        ({1:300, 2: infinite_mixture.cgpm_row_divide.data[1]}, {})
    assert infinite_mixture2.unobserve(2) == \
        ({0:2, 2: infinite_mixture.cgpm_row_divide.data[2]}, {})
