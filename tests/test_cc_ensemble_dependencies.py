# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

import numpy as np
import pytest

from cgpm.utils.general import get_prng

from cgpm2.crosscat_ensemble import CrossCatEnsemble
from cgpm2.transition_crosscat import GibbsCrossCat
from cgpm2.transition_crosscat import validate_crosscat_dependencies


@pytest.mark.xfail(strict=True,
    reason='Outputs must be zero based for dependence constraints.')
def test_dependencies_zero_based():
    prng = get_prng(2)
    CrossCatEnsemble(outputs=(1,2), inputs=(), Ci=[(1,2)],
        distributions=[('normal', None)]*2, chains=5, rng=prng)

@pytest.mark.xfail(strict=True,
    reason='CPP backend for view inference with dependence constraints.')
def test_dependencies_no_cpp():
    prng = get_prng(2)
    ensemble = CrossCatEnsemble(outputs=(0,1), inputs=[], Ci=[(0,1)],
        distributions=[('normal', None)]*2, chains=5, rng=prng)
    ensemble.observe(0, {0:0, 1:1})
    synthesizer = GibbsCrossCat(ensemble.cgpms[0], Ci=ensemble.Ci)
    synthesizer.transition_view_assignments()

def incorporate_data(ensemble, T):
    rowids = range(np.shape(T)[0])
    observations = [dict(zip(ensemble.outputs, row)) for row in T]
    ensemble.observe_bulk(rowids, observations)
    return ensemble

Ci_list = [
    list(itertools.combinations(range(10), 2)),   # All independent.
    [(2,8), (0,3)]                                # Custom independences.
]
@pytest.mark.parametrize('Ci', Ci_list)
def test_custom_independence(Ci):
    prng = get_prng(1)
    D = prng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    ensemble = CrossCatEnsemble(outputs=range(10), inputs=[],
        distributions=[('normal', None)]*10, chains=5, Ci=Ci, rng=prng)
    incorporate_data(ensemble, T)
    for crosscat in ensemble.cgpms:
        validate_crosscat_dependencies(crosscat, (), Ci)
    ensemble.transition(ensemble.make_default_inference_program(N=10))
    for crosscat in ensemble.cgpms:
        validate_crosscat_dependencies(crosscat, (), Ci)

CIs = [[], [(2,8), (0,3)]]
@pytest.mark.parametrize('Ci', CIs)
def test_simple_dependence_constraint(Ci):
    prng = get_prng(1)
    D = prng.normal(size=(10,1))
    T = np.repeat(D, 10, axis=1)
    Cd = [(2,0), (8,3)]
    ensemble = CrossCatEnsemble(outputs=range(10), inputs=[],
        distributions=[('normal', None)]*10, chains=5, Ci=Ci, Cd=Cd, rng=prng)
    incorporate_data(ensemble, T)
    for crosscat in ensemble.cgpms:
        validate_crosscat_dependencies(crosscat, (), Ci)
    ensemble.transition(ensemble.make_default_inference_program(N=10))
    for crosscat in ensemble.cgpms:
        validate_crosscat_dependencies(crosscat, Cd, Ci)

def get_independence_inference_data(prng):
    column_view_1 = prng.normal(loc=0, size=(50,1))
    column_view_2 = np.concatenate((
        prng.normal(loc=10, size=(25,1)),
        prng.normal(loc=20, size=(25,1)),
    ))
    data_view_1 = np.repeat(column_view_1, 4, axis=1)
    data_view_2 = np.repeat(column_view_2, 4, axis=1)
    return np.column_stack((data_view_1, data_view_2))

def test_independence_inference_break():
    # Get lovecat to disassemble a view into two views.
    prng = get_prng(584)
    data = get_independence_inference_data(prng)
    # HACK: Use Cd to initialize CrossCat state to one view.
    Cd = ((0, 1, 2, 3, 4, 5, 6, 7),)
    ensemble = CrossCatEnsemble(outputs=range(8), inputs=[],
        distributions=[('normal', None)]*8, chains=1, Cd=Cd, rng=prng)
    ensemble.Cd = ()
    incorporate_data(ensemble, data)
    ensemble.transition(ensemble.make_default_inference_program(N=100))
    crosscat = ensemble.cgpms[0]
    Zv = {c: i for i, cgpm in enumerate(crosscat.cgpms) for c in cgpm.outputs}
    for output in [0, 1, 2, 3]:
        assert Zv[output] == Zv[0]
    for output in [4, 5, 6, 7]:
        assert Zv[output] == Zv[4]
    assert len(crosscat.cgpms) == 2

def test_independence_inference_merge():
    # Get lovecat to merge dependent columns into one view.
    prng = get_prng(582)
    data = get_independence_inference_data(prng)
    # Hack: Use Cd/Ci to initialize CrossCat as
    #   {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3}
    Cd = ((0,1), (2,3), (4,5), (6,7))
    Ci = ((0,2), (0,4), (0, 6), (2,4), (2,6), (4,6))
    ensemble = CrossCatEnsemble(outputs=range(8), inputs=[],
        distributions=[('normal', None)]*8, chains=1, Cd=Cd, Ci=Ci, rng=prng)
    ensemble.Ci = ()
    incorporate_data(ensemble, data)
    ensemble.transition(ensemble.make_default_inference_program(N=100))
    crosscat = ensemble.cgpms[0]
    Zv = {c: i for i, cgpm in enumerate(crosscat.cgpms) for c in cgpm.outputs}
    for output in [0, 1, 2, 3,]:
        assert Zv[output] == Zv[0]
    for output in [4, 5, 6, 7]:
        assert Zv[output] == Zv[4]
    assert len(crosscat.cgpms) == 2
