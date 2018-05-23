# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import pytest

from cgpm.utils.general import get_prng

from cgpm2.crosscat_ensemble import CrossCatEnsemble


def get_crosscat_ensemble(prng):
    ensemble = CrossCatEnsemble(outputs=[1,2], inputs=[],
        distributions=[('normal', None), ('normal', None)], chains=10, rng=prng)
    ensemble.observe(1, {1:-1, 2:-1})
    ensemble.observe_bulk([2,3], [{1:-2, 2:-2}, {1:-3, 2:-3}], multiprocess=0)
    ensemble.observe(4, {1:-4})
    return ensemble

def test_cc_ensemble_crash():
    prng = get_prng(2)
    ensemble = get_crosscat_ensemble(prng)
    program = ensemble.make_default_inference_program(N=10)
    with pytest.raises(AssertionError):
        # Misaligned rowids (for crosscat states with two views).
        ensemble.transition(program, multiprocess=0)

    ensemble.observe(4, {2:-4})
    ensemble.transition(program, multiprocess=1)

    # Test for simulate.

    samples = ensemble.simulate(None, [1,2])
    assert len(samples) == ensemble.chains
    assert all(len(sample)==2 for sample in samples)

    samples = ensemble.simulate(None, [1,2], N=15)
    assert len(samples) == ensemble.chains
    assert all(len(sample)==15 for sample in samples)
    assert all(len(s)==2 for sample in samples for s in sample)

    # Test for simulate_bulk.

    samples = ensemble.simulate_bulk([100, 102], [[],[]], Ns=None)
    assert len(samples) == ensemble.chains
    assert all(len(sample)==2 for sample in samples)
    assert all(len(s)==0 for sample in samples for s in sample)

    samples = ensemble.simulate_bulk(None, [[1,2], [1]], Ns=[10,3])
    assert len(samples) == ensemble.chains
    assert all(len(sample)==2 for sample in samples)
    assert all(len(sample[0])==10 for sample in samples)
    assert all(len(sample[1])==3 for sample in samples)
    assert all(len(s)==2 for sample in samples for s in sample[0])
    assert all(len(s)==1 for sample in samples for s in sample[1])

    # Test for logpdf.

    logps = ensemble.logpdf(None, samples[0][1][0])
    assert len(logps) == ensemble.chains
    assert all(isinstance(l, float) for l in logps)

    # Test for logpdf_bulk.

    logps = ensemble.logpdf_bulk(None, samples[0][0])
    assert len(logps) == ensemble.chains
    assert all(len(logp)==len(samples[0][0]) for logp in logps)
    assert all(isinstance(l, float) for logp in logps for l in logp)

    logps = ensemble.logpdf_bulk(None, [])
    assert len(logps) == ensemble.chains
    assert all(len(logp) == 0 for logp in logps)

    def custom_program(crosscat):
        from cgpm2.transition_crosscat import GibbsCrossCat
        synthesizer = GibbsCrossCat(crosscat)
        synthesizer.transition(N=1, progress=False)
        return synthesizer.crosscat
    ensemble.transition(custom_program, multiprocess=1)

    metadata = ensemble.to_metadata()
    crosscat = CrossCatEnsemble.from_metadata(metadata, prng)

    ensemble.get_same_assignment_column_pairwise()
    ensemble.get_same_assignment_row_pairwise(1)

def test_simulate_heterogeneous_dimensions():
    prng = get_prng(2)
    ensemble = get_crosscat_ensemble(prng)

    # Test for simulate_heterogeneous.
    samples = ensemble.simulate_heterogeneous(
        rowid=None,
        targets=[[1,2]]*5 + [[1]]*5,
        constraints=[None]*5 + [{2:0}]*5,
        N=None
    )
    assert len(samples) == ensemble.chains
    assert all(len(sample)==2 for sample in samples[:5])
    assert all(len(sample)==1 for sample in samples[5:])
    assert all(sample.keys()==[1,2] for sample in samples[:5])
    assert all(sample.keys()==[1] for sample in samples[5:])

    samples = ensemble.simulate_heterogeneous(
        rowid=None,
        targets=[[1,2]]*5 + [[1]]*5,
        constraints=[None]*5 + [{2:0}]*5,
        N=[15]*3 + [10]*7,
        multiprocess=0,
    )
    assert len(samples) == ensemble.chains
    assert all(len(sample)==15 for sample in samples[:3])
    assert all(len(sample)==10 for sample in samples[3:])
    assert all(len(s)==2 for sample in samples[:5] for s in sample)
    assert all(len(s)==1 for sample in samples[5:] for s in sample)

    # Test for simulate_heterogeneous_bulk.

    samples = ensemble.simulate_heterogeneous_bulk(
        rowid=None,
        targets=[
            [[1], ],
            [[2], ],
            [[1], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[1], [2], [1,2,3]],
            [[1], [2], [1,2,3]],
            [[1], [2], [1,2,3]],
        ],
    )
    assert len(samples) == ensemble.chains
    assert all(len(sample)==1 for sample in samples[:2])
    assert all(len(sample)==2 for sample in samples[2:7])
    assert all(len(sample)==3 for sample in samples[7:])

    samples = ensemble.simulate_heterogeneous_bulk(
        rowid=None,
        targets=[
            [[1], ],
            [[2], ],
            [[1], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[2], [1,2,3]],
            [[1], [2], [1,2,3]],
            [[1], [2], [1,2,3]],
            [[1], [2], [1,2,3]],
        ],
        Ns=[
            [1, ],
            [20, ],
            [10, 30],
            [10, 30],
            [10, 30],
            [10, 30],
            [10, 30],
            [40, 7, 8],
            [40, 7, 8],
            [40, 7, 8],
        ],
    )
    assert len(samples) == ensemble.chains
    # Query for chain 0.
    assert len(samples[0]) == 1
    assert len(samples[0][0]) == 1
    # Query for chain 1.
    assert len(samples[1]) == 1
    assert len(samples[1][0]) == 20
    # Query for chains 2-6.
    for i in range(2, 7):
        assert len(samples[i]) == 2
        assert len(samples[i][0]) == 10
        assert len(samples[i][1]) == 30
    # Query for chains 7-9.
    for i in range(7, 10):
        assert len(samples[i]) == 3
        assert len(samples[i][0]) == 40
        assert len(samples[i][1]) == 7
        assert len(samples[i][2]) == 8
