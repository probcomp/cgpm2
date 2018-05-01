# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

import numpy as np

from cgpm.utils.general import build_cgpm
from cgpm.utils.general import get_prng
from cgpm.utils.parallel_map import parallel_map

from cgpm2.sample_crosscat import generate_random_partition

from cgpm2.categorical import Categorical
from cgpm2.crp import CRP
from cgpm2.normal import Normal
from cgpm2.poisson import Poisson

from cgpm2.flexible_rowmix import FlexibleRowMixture
from cgpm2.product import Product

from cgpm2.transition_rows import get_rowids

from cgpm2.transition_crosscat import GibbsCrossCat
from cgpm2.transition_crosscat import get_distribution_outputs

# Initializer for CrossCat state.

distribution_to_cgpm = {
    'categorical' : Categorical,
    'crp'         : CRP,
    'normal'      : Normal,
    'poisson'     : Poisson,
}

def make_random_primitive(output, distribution, rng):
    initializer = distribution_to_cgpm[distribution[0]]
    return initializer([output], [], distargs=distribution[1], rng=rng)

def make_random_view(outputs, distributions, rng):
    crp_output = rng.randint(2**32-1)
    cgpm_row_divide = CRP([crp_output], [], rng=rng)
    cgpm_base_list = [make_random_primitive(output, distribution, rng)
        for output, distribution in zip(outputs, distributions)]
    view = FlexibleRowMixture(
        cgpm_row_divide=cgpm_row_divide,
        cgpm_components_base=Product(cgpm_base_list, rng=rng),
        rng=rng
    )
    return view

def make_random_crosscat((outputs, distributions, seed)):
    rng = get_prng(seed)
    alpha = rng.gamma(2,1)
    partition = generate_random_partition(alpha, len(outputs), rng)
    views = [
        make_random_view(
            outputs=[outputs[i] for i in block],
            distributions=[distributions[i] for i in block],
            rng=rng)
        for block in partition
    ]
    crosscat = Product(cgpms=views, rng=rng)
    return crosscat

# Multiprocessing functions.

def _modify((method, crosscat, args)):
    getattr(crosscat, method)(*args)
    return crosscat

def _modify_bulk((method, crosscat, args)):
    args_list = make_args_list(*args)
    for args in args_list:
        getattr(crosscat, method)(*args)
    return crosscat

def _evaluate((method, crosscat, args)):
    return getattr(crosscat, method)(*args)

def _evaluate_bulk((method, crosscat, args)):
    args_list = make_args_list(*args)
    return [getattr(crosscat, method)(*args) for args in args_list]

def _alter((funcs, crosscat)):
    for func in funcs:
        crosscat = func(crosscat)
    return crosscat

def make_args_list(*args):
    Ns = [len(a) for a in args if a is not None]
    assert all(n==Ns[0] for n in Ns)
    N = Ns[0]
    def listify(arg):
        if arg is None:
            return [None]*N
        else:
            assert len(arg) == N
            return arg
    args_list = map(listify, args)
    return zip(*args_list)

# CrossCat.

def make_default_inference_program(N=None, S=None, outputs=None):
    def func(crosscat):
        synthesizer = GibbsCrossCat(crosscat, crosscat.rng)
        synthesizer.transition_structure_cpp(N=N, S=S, outputs=outputs)
        return synthesizer.crosscat
    return func

def same_assignment_column((crosscat, output0, output1)):
    view_idx0 = crosscat.output_to_index[output0]
    view_idx1 = crosscat.output_to_index[output1]
    return view_idx0 == view_idx1

def same_assignment_row((crosscat, output, row0, row1)):
    view_idx = crosscat.output_to_index[output]
    view = crosscat.cgpms[view_idx]
    cluster_idx0 = view.cgpm_row_divide.data[row0]
    cluster_idx1 = view.cgpm_row_divide.data[row1]
    return cluster_idx0 == cluster_idx1

def same_assignment_column_pairwise((crosscat, outputs)):
    matrix = np.eye(len(outputs))
    reindex = {output: i for i, output in enumerate(outputs)}
    for i,j in itertools.combinations(outputs, 2):
        d = same_assignment_column((crosscat, i, j))
        matrix[reindex[i], reindex[j]] = matrix[reindex[j], reindex[i]] = d
    return matrix

def same_assignment_row_pairwise((crosscat, output)):
    view_idx = crosscat.output_to_index[output]
    rowids = get_rowids(crosscat.cgpms[view_idx])
    matrix = np.eye(len(rowids))
    reindex = {rowid: i for i, rowid in enumerate(rowids)}
    for i, j in itertools.combinations(rowids, 2):
        s = same_assignment_row((crosscat, output, i, j))
        matrix[reindex[i], reindex[j]] = matrix[reindex[j], reindex[i]] = s
    return matrix

class CrossCat(object):

    def __init__(self, outputs, inputs, distributions, chains=1, rng=None):
        # Assertion.
        assert len(outputs) == len(distributions)
        assert inputs == []
        # From constructor.
        self.outputs = outputs
        self.inputs = inputs
        self.chains = chains
        self.distributions = distributions
        self.rng = rng or get_prng(1)
        # Derived attributes.
        seeds = self.rng.randint(0, 2**32-1, size=chains)
        self.chains_list = range(chains)
        self.cgpms = map(make_random_crosscat,
            [(outputs, distributions, seed) for seed in seeds])

    # Observe.

    def _observe(self, func, rowid, observation, inputs, multiprocess):
        mapper = parallel_map if multiprocess else map
        args = [('observe', self.cgpms[chain],
                (rowid, observation, inputs))
                for chain in self.chains_list]
        self.cgpms = mapper(func, args)

    def observe(self, rowid, observation, inputs=None, multiprocess=0):
        self._observe(_modify, rowid, observation, inputs, multiprocess)

    def observe_bulk(self, rowids, observations, inputs=None, multiprocess=1):
        self._observe(_modify_bulk, rowids, observations, inputs, multiprocess)

    # Unobserve.

    def _unobserve(self, func, rowid, multiprocess):
        mapper = parallel_map if multiprocess else map
        args = [('unobserve', self.cgpms[chain],
                (rowid,))
                for chain in self.chains_list]
        self.cgpms = mapper(func, args)

    def unobserve(self, rowid, multiprocess=0):
        self._unobserve(_modify, rowid, multiprocess)

    def unobserve_bulk(self, rowids, multiprocess=0):
        self._unobserve(_modify_bulk, rowids, multiprocess)

    # logpdf

    def _logpdf(self, func, rowids, targets, constraints, inputs,
            multiprocess):
        mapper = parallel_map if multiprocess else map
        args = [('logpdf', self.cgpms[chain],
                (rowids, targets, constraints, inputs))
            for chain in self.chains_list]
        logpdfs = mapper(func, args)
        return logpdfs

    def logpdf(self, rowid, targets, constraints=None, inputs=None,
            multiprocess=0):
        return self._logpdf(_evaluate, rowid, targets, constraints,
            inputs, multiprocess)

    def logpdf_bulk(self, rowids, targets, constraints=None, inputs=None,
            multiprocess=1):
        return self._logpdf(_evaluate_bulk, rowids, targets, constraints,
            inputs, multiprocess)

    # simulate

    def _simulate(self, func, rowids, targets, constraints, inputs, N,
            multiprocess):
        mapper = parallel_map if multiprocess else map
        args = [('simulate', self.cgpms[chain],
                (rowids, targets, constraints, inputs, N))
            for chain in self.chains_list]
        samples = mapper(func, args)
        return samples

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None,
            multiprocess=0):
        return self._simulate(_evaluate, rowid, targets, constraints,
            inputs, N, multiprocess)

    def simulate_bulk(self, rowids, targets, constraints=None, inputs=None,
            Ns=None, multiprocess=1):
        return self._simulate(_evaluate_bulk, rowids, targets, constraints,
            inputs, Ns, multiprocess)

    # Arbitrary stochastic/deterministic mutation operators.

    def transition(self, program, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [([program], self.cgpms[chain],)
            for chain in self.chains_list]
        self.cgpms = mapper(_alter, args)

    # Structural properties.

    def get_same_assignment_column(self, output0, output1, multiprocess=0):
        mapper = parallel_map if multiprocess else map
        args = [(self.cgpms[chain], output0, output1)
            for chain in self.chains_list]
        return mapper(same_assignment_column, args)

    def get_same_assignment_column_pairwise(self, multiprocess=1):
        mapper = parallel_map if multiprocess else map
        args = [(self.cgpms[chain], self.outputs)
            for chain in self.chains_list]
        result = mapper(same_assignment_column_pairwise, args)
        return np.asarray(result)

    def get_same_assignment_row(self, output, rowid0, rowid1, multiprocess=0):
        mapper = parallel_map if multiprocess else map
        args = [(self.cgpms[chain], output, rowid0, rowid1)
            for chain in self.chains_list]
        return mapper(same_assignment_row, args)

    def get_same_assignment_row_pairwise(self, output, multiprocess=0):
        mapper = parallel_map if multiprocess else map
        args = [(self.cgpms[chain], output)
            for chain in self.chains_list]
        result = mapper(same_assignment_row_pairwise, args)
        return np.asarray(result)

    # Serialization.

    def to_metadata(self):
        metadata = dict()
        metadata['chains'] = self.chains
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['distributions'] = self.distributions
        metadata['cgpms'] = [cgpm.to_metadata() for cgpm in self.cgpms]
        metadata['factory'] = ('cgpm2.feralcat', 'CrossCat')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'],
            metadata['inputs'],
            metadata['distributions'],
            chains=metadata['chains'],
            rng=rng,
        )
        cgpms = [build_cgpm(blob, rng) for blob in metadata['cgpms']]
        model.cgpms = cgpms
        return model
