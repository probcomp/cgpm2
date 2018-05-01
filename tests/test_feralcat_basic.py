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

from cgpm.utils.general import get_prng

from cgpm2.feralcat import CrossCat
from cgpm2.feralcat import make_default_inference_program


def test_feralcat_crash():
    prng = get_prng(2)
    crosscat = CrossCat(outputs=[1,2], inputs=[],
        distributions=[('normal', None), ('normal', None)], chains=10, rng=prng)
    observation = {1:1, 2:0}
    crosscat.observe(1, observation)
    crosscat.observe_bulk([2,3], [observation, observation], multiprocess=True)

    samples = crosscat.simulate(None, [1,2], N=15)
    assert len(samples) == crosscat.chains
    assert all(len(sample)==15 for sample in samples)
    assert all(len(s)==2 for sample in samples for s in sample)

    samples = crosscat.simulate_bulk(None, [[1,2], [1]], Ns=[10,3])
    assert len(samples) == crosscat.chains
    assert all(len(sample)==2 for sample in samples)
    assert all(len(sample[0])==10 for sample in samples)
    assert all(len(sample[1])==3 for sample in samples)
    assert all(len(s)==2 for sample in samples for s in sample[0])
    assert all(len(s)==1 for sample in samples for s in sample[1])

    logps = crosscat.logpdf(None, samples[0][1][0])
    assert len(logps) == crosscat.chains
    assert all(isinstance(l, float) for l in logps)

    logps = crosscat.logpdf_bulk(None, samples[0][0])
    assert len(logps) == crosscat.chains
    assert all(len(logp)==len(samples[0][0]) for logp in logps)
    assert all(isinstance(l, float) for logp in logps for l in logp)

    program = make_default_inference_program(N=10)
    crosscat.transition(program, multiprocess=1)

    def custom_program(crosscat):
        from cgpm2.transition_crosscat import GibbsCrossCat
        synthesizer = GibbsCrossCat(crosscat, crosscat.rng)
        synthesizer.transition(N=1)
        return synthesizer.crosscat
    crosscat.transition(custom_program, multiprocess=1)
