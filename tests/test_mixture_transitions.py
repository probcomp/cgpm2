# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import Counter

import numpy as np

from cgpm.utils.general import get_prng

from cgpm2.crp import CRP
from cgpm2.normal import Normal

from cgpm2.flexible_rowmix import FlexibleRowMixture

from cgpm2.transition_hypers import transition_hyper_grids
from cgpm2.transition_hypers import transition_hypers
from cgpm2.transition_rows import transition_rows

from cgpm2.walks import get_cgpms_by_output_index

def test_transition_crp_mixture():
    prng = get_prng(2)
    data = np.concatenate((
        prng.normal(loc=0, scale=2, size=20),
        prng.normal(loc=30, scale=1, size=20),
        prng.normal(loc=-30, scale=1, size=20),
    ))
    infinite_mixture = FlexibleRowMixture(
        cgpm_row_divide=CRP([1], [], rng=prng),
        cgpm_components_base=Normal([0], [], rng=prng),
        rng=prng
    )
    for rowid, value in enumerate(data):
        infinite_mixture.observe(rowid, {0: value})
    cgpms = {
        0  : get_cgpms_by_output_index(infinite_mixture, 0),
        1  : get_cgpms_by_output_index(infinite_mixture, 1),
    }
    grids = {
        0 : transition_hyper_grids(cgpms[0], 30),
        1 : transition_hyper_grids(cgpms[1], 30),
    }
    for _step in xrange(50):
        rowids = prng.permutation(range(len(data)))
        for rowid in rowids:
            transition_rows(infinite_mixture, rowid, prng)
        for output in infinite_mixture.outputs:
            transition_hypers(cgpms[output], grids[output], prng)
    rowids = range(60)
    assignments0 = [infinite_mixture.simulate(r, [1])[1] for r in rowids[00:20]]
    assignments1 = [infinite_mixture.simulate(r, [1])[1] for r in rowids[20:40]]
    assignments2 = [infinite_mixture.simulate(r, [1])[1] for r in rowids[40:60]]
    mode0 = Counter(assignments0).most_common(1)[0][0]
    mode1 = Counter(assignments1).most_common(1)[0][0]
    mode2 = Counter(assignments2).most_common(1)[0][0]
    assert sum(a==mode0 for a in assignments0) > int(0.95*len(assignments0))
    assert sum(a==mode1 for a in assignments1) > int(0.95*len(assignments1))
    assert sum(a==mode2 for a in assignments2) > int(0.95*len(assignments2))
