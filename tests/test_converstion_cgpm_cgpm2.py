# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np

from cgpm.crosscat.state import State
from cgpm.utils.general import get_prng

from cgpm2.conversion import convert_cgpm_state_to_cgpm2

def test_convert_cgpm_to_cgpm2():
    prng = get_prng(2)
    data = np.concatenate((
        prng.normal(loc=0, scale=2, size=20),
        prng.normal(loc=30, scale=1, size=20),
        prng.normal(loc=-30, scale=1, size=20),
    ))
    state = State(X=np.reshape(data, (len(data),1)), outputs=[0],
        cctypes=['normal'], rng=prng)
    view_cgpm1 = state.views[0]
    view_cgpm1.transition(N=5)
    # Convert
    product = convert_cgpm_state_to_cgpm2(state)
    view_cgpm2 = product.cgpms[0]
    # Verify row assignments.
    assignments0 = view_cgpm1.Zr()
    partition0 = [[r for r, z in assignments0.iteritems() if z==u]
        for u in set(assignments0.values())]
    assignments1 = view_cgpm2.cgpm_row_divide.data
    partition1 = [[r for r, z in assignments1.iteritems() if z==u]
        for u in set(assignments1.values())]
    partition0_sorted = sorted(partition0, key=min)
    partition1_sorted = sorted(partition1, key=min)
    assert partition0_sorted == partition1_sorted
    # Verify hyperparameters.
    hypers0 = view_cgpm1.dims[0].hypers
    hypers1 = view_cgpm2.cgpm_components_array.cgpm_base.cgpms[0].get_hypers()
    assert hypers0 == hypers1
    # Verify CRP alpha.
    alpha0 = view_cgpm1.crp.hypers
    alpha1 = view_cgpm2.cgpm_row_divide.get_hypers()
    assert alpha0 == alpha1
