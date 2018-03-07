# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from itertools import chain

from math import isnan

from cgpm.utils.general import log_pflip

from .chain import Chain
from .distribution import DistributionCGPM
from .finite_array import FiniteArray
from .finite_rowmix import FiniteRowMixture
from .flexible_array import FlexibleArray
from .flexible_rowmix import FlexibleRowMixture
from .product import Product


def transition_hypers(cgpms, grids, rng):
    """Transitions the hyperparameters for list of cgpms jointly."""
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    assert all([type(cgpm) is type(cgpms[0]) for cgpm in cgpms])
    hyperparams = cgpms[0].get_hypers()
    shuffled_hypers = rng.permutation(hyperparams.keys())
    # For each hyper.
    for hyper in shuffled_hypers:
        logps = []
        # For each grid point.
        for grid_value in grids[hyper]:
            # Compute the probability of the grid point.
            hyperparams[hyper] = grid_value
            logp_k = 0
            for cgpm in cgpms:
                cgpm.set_hypers(hyperparams)
                logp_k += cgpm.logpdf_score()
            logps.append(logp_k)
        # Sample a new hyperparameter from the grid.
        index = log_pflip(logps, rng=rng)
        hyperparams[hyper] = grids[hyper][index]
    # Set the hyperparameters for each cgpm.
    for cgpm in cgpms:
        cgpm.set_hypers(hyperparams)
    return hyperparams


def transtion_hyper_grids(cgpms, n_grid=30):
    """Get hyperparameter grid using Empirical Bayes across all CGPMs."""
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    assert all([type(cgpm) is type(cgpms[0]) for cgpm in cgpms])
    X = [x for cgpm in cgpms for x in cgpm.data.itervalues() if not isnan(x)]
    return cgpms[0].construct_hyper_grids(X, n_grid)


def get_cgpms_by_output_index(cgpm, output):
    """Retrieve all CGPMs responsible for modeling given output."""
    if isinstance(cgpm, DistributionCGPM):
        return [cgpm] if cgpm.outputs == [output] else []
    elif isinstance(cgpm, (Chain, Product)):
        cgpm_list = [c for c in cgpm.cgpms if output in c.outputs]
        assert len(cgpm_list) == 1
        return get_cgpms_by_output_index(cgpm_list[0], output)
    elif isinstance(cgpm, FiniteArray):
        cgpm_list = [get_cgpms_by_output_index(c, output) for c in cgpm.cgpms]
        return list(chain.from_iterable(cgpm_list))
    elif isinstance(cgpm, FlexibleArray):
        cgpm_list = [get_cgpms_by_output_index(c, output) for c in
            cgpm.cgpms.values() + [cgpm.cgpm_base]]
        return list(chain.from_iterable(cgpm_list))
    elif isinstance(cgpm, (FiniteRowMixture, FlexibleRowMixture)):
        if output in cgpm.cgpm_row_divide.outputs:
            return get_cgpms_by_output_index(cgpm.cgpm_row_divide, output)
        else:
            return get_cgpms_by_output_index(cgpm.cgpm_components_array, output)
    else:
        assert False, 'Unknown CGPM'
