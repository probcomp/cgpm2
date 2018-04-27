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
import pandas as pd

from cgpm2.categorical import Categorical
from cgpm2.crp import CRP
from cgpm2.finite_rowmix import FiniteRowMixture
from cgpm2.normal import Normal
from cgpm2.product import Product
from cgpm2.transition_hypers import transition_hyper_grids
from cgpm2.transition_hypers import transition_hypers
from cgpm2.transition_rows import transition_rows
from cgpm2.transition_views import transition_cgpm_view_assigments
from cgpm2.walks import get_cgpms_by_output_index


def make_dataset(prng):
    data0 = prng.normal(loc=0, scale=2, size=20)
    data1 = prng.normal(loc=30, scale=1, size=20)
    data2 = prng.normal(loc=-30, scale=1, size=20)
    return np.concatenate((data0, data1, data2))


def test_basic_inference_quality_finite():
    
    finite_mixture = FiniteRowMixture(
        cgpm_row_divide=Categorical([-1], [], distargs={'k':3}, rng=prng),
        cgpm_components=[
            Normal([0], [], rng=prng),
            Normal([0], [], rng=prng),
            Normal([0], [], rng=prng)
        ],
        rng=prng,
    )

    data = make_dataset(2)

    for rowid, value in enumerate(data):
        finite_mixture.observe(rowid, {0: value})