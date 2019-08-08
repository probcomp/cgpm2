# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np
import pytest

from cgpm2.crosscat_ensemble import CrossCatEnsemble
from cgpm2.utils import get_prng

def make_custom_program(N):
    def custom_program(crosscat):
        from cgpm2.transition_crosscat import GibbsCrossCat
        synthesizer = GibbsCrossCat(crosscat)
        synthesizer.transition(N=N,
            kernels=[
                'hypers_distributions',
                'hypers_row_divide',
                'hypers_column_divide',
                'row_assignments',
        ], progress=False)
        return synthesizer.crosscat
    return custom_program

def test_entropy_bernoulli_univariate__ci_():
    integration = pytest.config.getoption('--integration')
    n_data = 250 if integration else 10
    n_step = 10 if integration else 1
    n_samp = 1000 if integration else 10
    prng = get_prng(10)
    # Generate a biased Bernoulli dataset.
    T = prng.choice([0,1], p=[.3,.7], size=n_data)
    # Create ensemble.
    ensemble = CrossCatEnsemble(
        outputs=[0],
        inputs=[],
        distributions=[('bernoulli', None)],
        chains=16,
        rng=prng,
    )
    # Observe data.
    for rowid, x in enumerate(T):
        ensemble.observe(rowid, {0: x})
    # Run inference.
    program = make_custom_program(N=n_step)
    ensemble.transition(program, multiprocess=1)
    # Exact computation.
    entropy_exact = - (.3*np.log(.3) + .7*np.log(.7))
    # logpdf computation.
    logps = ensemble.logpdf_bulk(None, [{0:0}, {0:1}])
    entropy_logpdf = [-np.sum(np.exp(logp)*logp) for logp in logps]
    # Mutual_information computation.
    entropy_mi_estimates = ensemble.mutual_information([0], [0], N=n_samp)
    entropy_mi = [estimate.mean for estimate in entropy_mi_estimates]
    # Punt CLT analysis and go for 1 decimal place.
    assert not integration or np.allclose(entropy_exact, entropy_logpdf, atol=.1)
    assert not integration or np.allclose(entropy_exact, entropy_mi, atol=.1)
    assert not integration or np.allclose(entropy_logpdf, entropy_mi, atol=.05)

def test_entropy_bernoulli_bivariate__ci_():
    # Set the test parameters.
    integration = pytest.config.getoption('--integration')
    n_data = 250 if integration else 10
    n_step = 20 if integration else 1
    n_samp = 1000 if integration else 10
    prng = get_prng(10)
    # Generate a bivariate Bernoulli dataset.
    PX = [.3, .7]
    PY = [[.2, .8], [.6, .4]]
    TX = prng.choice([0,1], p=PX, size=n_data)
    TY = np.zeros(shape=len(TX))
    TY[TX==0] = prng.choice([0,1], p=PY[0], size=len(TX[TX==0]))
    TY[TX==1] = prng.choice([0,1], p=PY[1], size=len(TX[TX==1]))
    T = np.column_stack((TY,TX))
    # Create ensemble.
    ensemble = CrossCatEnsemble(
        outputs=[0,1],
        inputs=[],
        distributions=[('bernoulli', None)]*2,
        Cd=[[0,1]],
        chains=64,
        rng=prng,
    )
    # Observe data.
    for rowid, (x, y) in enumerate(T):
        ensemble.observe(rowid, {0: x, 1:y})
    # Run inference.
    program = make_custom_program(N=n_step)
    ensemble.transition(program, multiprocess=1)
    # Exact computation.
    entropy_exact = (
        - PX[0]*PY[0][0] * np.log(PX[0]*PY[0][0])
        - PX[0]*PY[0][1] * np.log(PX[0]*PY[0][1])
        - PX[1]*PY[1][0] * np.log(PX[1]*PY[1][0])
        - PX[1]*PY[1][1] * np.log(PX[1]*PY[1][1])
    )
    # logpdf computation.
    logps = ensemble.logpdf_bulk(None,
        [{0:0, 1:0}, {0:0, 1:1}, {0:1, 1:0}, {0:1, 1:1}]
    )
    entropy_logpdf = [-np.sum(np.exp(logp)*logp) for logp in logps]
    # Mutual_information computation.
    n_samp = 10 if not integration else 1000
    entropy_mi_estimates = ensemble.mutual_information([0,1], [0,1], N=n_samp)
    entropy_mi = [estimate.mean for estimate in entropy_mi_estimates]
    # Punt CLT analysis and go for a small tolerance.
    assert not integration or np.allclose(entropy_exact, entropy_logpdf, atol=.1)
    assert not integration or np.allclose(entropy_exact, entropy_mi, atol=.1)
    assert not integration or np.allclose(entropy_logpdf, entropy_mi, atol=.1)
