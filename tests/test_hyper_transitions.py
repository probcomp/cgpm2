# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import get_prng

from cgpm2.crp import CRP
from cgpm2.normal import Normal
from cgpm2.poisson import Poisson

from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.transition_hypers import transition_hyper_grids
from cgpm2.transition_hypers import transition_hypers

from cgpm2.walks import get_cgpms_by_output_index


def test_transition_hypers_basic():
    prng = get_prng(2)
    component0 = Product([
        Poisson([0], [], hypers={'m':100}, rng=prng),
        Normal([1], [], hypers={'m':100}, rng=prng)
        ],
        rng=prng)
    cgpm_row_divide = CRP([2], [], rng=prng)
    infinite_mixture = FlexibleRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components_base=component0,
        rng=prng)
    # Make normal observations.
    infinite_mixture.observe(0, {1:100})
    infinite_mixture.observe(1, {1:300})
    infinite_mixture.observe(2, {1:-300})
    # Fetch log score.
    log_score0 = infinite_mixture.logpdf_score()
    # Run inference.
    normal_cgpms = get_cgpms_by_output_index(infinite_mixture, 1)
    grids_normal = transition_hyper_grids(normal_cgpms, 30)
    hypers_normal = [transition_hypers(normal_cgpms, grids_normal, prng)
        for _i in xrange(2)]
    assert not all(hypers == hypers_normal[0] for hypers in hypers_normal)
    log_score1 = infinite_mixture.logpdf_score()
    assert log_score0 < log_score1
