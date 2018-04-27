# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import get_prng

from cgpm2.categorical import Categorical
from cgpm2.normal import Normal

from cgpm2.finite_rowmix import FiniteRowMixture
from cgpm2.product import Product

from cgpm2.transition_rows import transition_rows

def test_transition_rows_fixed_mixture():
    prng = get_prng(2)
    component0 = Product([
        Normal([0], [], hypers={'m':1000}, rng=prng),
        Normal([1], [], hypers={'m':0}, rng=prng)
        ], rng=prng)
    component1 = Product([
        Normal([0], [], hypers={'m':-1000}, rng=prng),
        Normal([1], [], hypers={'m':1000}, rng=prng)
        ], rng=prng)
    component2 = Product([
        Normal([0], [], hypers={'m':0}, rng=prng),
        Normal([1], [], hypers={'m':-100}, rng=prng)
        ], rng=prng)
    cgpm_row_divide = Categorical([2], [], distargs={'k':3}, rng=prng)
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components=[component0, component1, component2],
        rng=prng)
    # For component 0.
    finite_mixture.observe(0, {0:1000, 1:0, 2:0})
    finite_mixture.observe(1, {0:990, 1:-10, 2:0})
    # For component 1.
    finite_mixture.observe(2, {0:-1000, 1:1000, 2:0})
    finite_mixture.observe(3, {0:-990, 1:990, 2:0})
    # For component 2.
    finite_mixture.observe(4, {0:0, 1:-1000, 2:0})
    finite_mixture.observe(5, {0:10, 1:-990, 2:0})
    # Confirm all rows in component 0.
    assert finite_mixture.simulate(0, [2]) == {2:0}
    assert finite_mixture.simulate(1, [2]) == {2:0}
    assert finite_mixture.simulate(2, [2]) == {2:0}
    assert finite_mixture.simulate(3, [2]) == {2:0}
    assert finite_mixture.simulate(4, [2]) == {2:0}
    assert finite_mixture.simulate(5, [2]) == {2:0}
    # Run transitions
    for _i in xrange(10):
        for rowid in range(6):
            transition_rows(finite_mixture, rowid, prng)
    # Confirm all rows in correct components.
    assert finite_mixture.simulate(0, [2]) == {2:0}
    assert finite_mixture.simulate(1, [2]) == {2:0}
    assert finite_mixture.simulate(2, [2]) == {2:1}
    assert finite_mixture.simulate(3, [2]) == {2:1}
    assert finite_mixture.simulate(4, [2]) == {2:2}
    assert finite_mixture.simulate(5, [2]) == {2:2}
