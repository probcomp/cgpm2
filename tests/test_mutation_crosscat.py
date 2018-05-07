# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from cgpm.utils.general import get_prng

from cgpm2.crp import CRP
from cgpm2.normal import Normal

from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.transition_crosscat import GibbsCrossCat
from cgpm2.transition_crosscat import get_distribution_cgpms

from .test_inference_bivariate_mixture import make_bivariate_two_clusters

def get_crosscat_synthesizer(prng):
    view = FlexibleRowMixture(
        cgpm_row_divide=CRP([2], [], rng=prng),
        cgpm_components_base=Product([
            Normal([0], [], rng=prng),
            Normal([1], [], rng=prng),
        ], rng=prng),
        rng=prng)
    crosscat = Product(cgpms=[view], rng=prng)
    data = make_bivariate_two_clusters(prng)
    for rowid, row in enumerate(data):
        crosscat.observe(rowid, {0: row[0], 1: row[1]})
    return GibbsCrossCat(crosscat)


def test_mutation_hypers_component():
    prng = get_prng(2)
    synthesizer = get_crosscat_synthesizer(prng)
    normals = get_distribution_cgpms(synthesizer.crosscat, [0])[0]
    for v in np.linspace(0.01, 10, 1):
        synthesizer.set_hypers_distribution(0, {'m':v})
        for cgpm in normals:
            assert cgpm.get_hypers()['m'] == v

def test_mutation_hypers_crp():
    prng = get_prng(2)
    synthesizer = get_crosscat_synthesizer(prng)
    crp = synthesizer.crosscat.cgpms[0].cgpm_row_divide
    for v in np.linspace(0.01, 10, 1):
        synthesizer.set_hypers_row_divide(0, {'alpha':v})
        assert crp.get_hypers()['alpha'] == v

def test_mutation_set_view_assignment():
    prng = get_prng(2)
    synthesizer = get_crosscat_synthesizer(prng)
    # Move column 0 zero to singleton view.
    synthesizer.set_view_assignment(0, None)
    assert len(synthesizer.crosscat.cgpms) == 2
    # Move column 1 to view of column 0.
    synthesizer.set_view_assignment(1, 0)
    assert len(synthesizer.crosscat.cgpms) == 1

def test_mutation_set_rowid_component():
    prng = get_prng(2)
    synthesizer = get_crosscat_synthesizer(prng)
    crp = synthesizer.crosscat.cgpms[0].cgpm_row_divide
    # Move row 0 to singleton component.
    synthesizer.set_rowid_component(0, 0, None)
    new_cluster = crp.data[0]
    assert crp.counts[new_cluster] == 1
    # Move row 0 to singleton component again.
    synthesizer.set_rowid_component(0, 0, None)
    new_cluster_prime = crp.data[0]
    assert new_cluster_prime == new_cluster
    assert crp.counts[new_cluster_prime] == 1
    # Move row 10 to component of row 0.
    synthesizer.set_rowid_component(0, 10, 0)
    assert crp.data[10] == new_cluster_prime
    assert crp.counts[new_cluster_prime] == 2
