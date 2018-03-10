# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

from math import isnan

from cgpm.utils.general import log_pflip

from .distribution import DistributionCGPM


def transition_hypers(cgpms, grids, rng):
    """Transitions hyperparameters of cgpms greedily."""
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

def transition_hypers_full(cgpms, grids, rng):
    """Transitions hyperparameters of cgpms using full grid search."""
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    assert all([type(cgpm) is type(cgpms[0]) for cgpm in cgpms])
    hypers = grids.keys()
    cells = list(itertools.product(*(grids.itervalues())))
    logps = []
    for cell in cells:
        proposal = dict(zip(hypers, cell))
        logp_cell = 0
        for cgpm in cgpms:
            cgpm.set_hypers(proposal)
            logp_cell += cgpm.logpdf_score()
        logps.append(logp_cell)
    index = log_pflip(logps, rng=rng)
    selected = dict(zip(hypers, cells[index]))
    for cgpm in cgpms:
        cgpm.set_hypers(selected)
    return selected, cells, logps

def transtion_hyper_grids(cgpms, n_grid=30):
    """Get hyperparameter grid using Empirical Bayes across all CGPMs."""
    assert all([isinstance(cgpm, DistributionCGPM) for cgpm in cgpms])
    assert all([type(cgpm) is type(cgpms[0]) for cgpm in cgpms])
    X = [x for cgpm in cgpms for x in cgpm.data.itervalues() if not isnan(x)]
    return cgpms[0].construct_hyper_grids(X, n_grid)
