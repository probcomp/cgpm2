# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import pytest

from cgpm.utils.general import get_prng

from cgpm2.crp import CRP
from cgpm2.normal import Normal

from cgpm2.product import Product
from cgpm2.flexible_rowmix import FlexibleRowMixture

from cgpm2.walks import remove_cgpm
from cgpm2.walks import add_cgpm

def test_add_remove():
    prng = get_prng(2)
    mixture0 = FlexibleRowMixture(
        cgpm_row_divide=CRP([2], [], rng=prng),
        cgpm_components_base=Product([
            Normal([0], [], rng=prng),
            Normal([1], [], rng=prng),
        ], rng=prng),
        rng=prng)
    for rowid, row in enumerate([[0,.9] ,[.5, 1], [-.5, 1.2]]):
        mixture0.observe(rowid, {0:row[0], 1:row[1]})

    mixture1 = remove_cgpm(mixture0, 0)
    assert mixture0.outputs == [2, 0, 1]
    assert mixture1.outputs == [2, 1]

    mixture2 = add_cgpm(mixture1, Normal([0], [], rng=prng))
    assert mixture0.outputs == [2, 0, 1]
    assert mixture1.outputs == [2, 1]
    assert mixture2.outputs == [2, 1, 0]

    mixture3 = remove_cgpm(mixture2, 1)
    assert mixture0.outputs == [2, 0, 1]
    assert mixture1.outputs == [2, 1]
    assert mixture2.outputs == [2, 1, 0]
    assert mixture3.outputs == [2, 0]

    mixture4 = remove_cgpm(mixture3, 0)
    assert mixture0.outputs == [2, 0, 1]
    assert mixture1.outputs == [2, 1]
    assert mixture2.outputs == [2, 1, 0]
    assert mixture3.outputs == [2, 0]
    assert mixture4.outputs == [2]

    with pytest.raises(Exception):
        # Cannot remove the cgpm_row_divide for a mixture.
        mixture3 = remove_cgpm(mixture2, 2)
