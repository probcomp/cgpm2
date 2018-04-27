# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import pytest

from cgpm.utils.general import get_prng

from cgpm2.crp import CRP
from cgpm2.normal import Normal
from cgpm2.poisson import Poisson

from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.walks import get_cgpms_by_output_index

def test_product_mixture_walk():
    prng = get_prng(2)
    component_base = Product([
        Poisson([0], [], hypers={'a': 10, 'b': 1}, rng=prng),
        Normal([1], [], hypers={'m':100}, rng=prng)
        ],
        rng=prng)
    cgpm_row_divide = CRP([2], [], rng=prng)
    infinite_mixture = FlexibleRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components_base=component_base,
        rng=prng)
    # Only the base CGPMs in the flexible mixture.
    cgpm_poisson = get_cgpms_by_output_index(infinite_mixture, 0)
    cgpm_normal = get_cgpms_by_output_index(infinite_mixture, 1)
    cgpm_crp = get_cgpms_by_output_index(infinite_mixture, 2)
    assert cgpm_poisson == [component_base.cgpms[0]]
    assert cgpm_normal == [component_base.cgpms[1]]
    assert cgpm_crp == [cgpm_row_divide]
    infinite_mixture.observe(0, {0:1})
    # New CGPMs in the flexible CGPM after observing.
    cgpm_poisson = get_cgpms_by_output_index(infinite_mixture, 0)
    cgpm_normal = get_cgpms_by_output_index(infinite_mixture, 1)
    assert len(cgpm_poisson) == len(cgpm_normal) == 2
    assert [cgpm_poisson[-1]] == [component_base.cgpms[0]]
    assert [cgpm_normal[-1]] == [component_base.cgpms[1]]
    assert cgpm_poisson[0].N == 1
    assert cgpm_normal[0].N == 0
    cgpm_crp = get_cgpms_by_output_index(infinite_mixture, 2)
    assert len(cgpm_crp) == 1
    assert cgpm_crp[0].N == 1
    assert cgpm_crp[0].data[0] == 0
    # Misc. errors, no such output.
    with pytest.raises(Exception):
        get_cgpms_by_output_index(infinite_mixture, -1)
