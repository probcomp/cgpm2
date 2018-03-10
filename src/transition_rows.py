# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import log_pflip

from .finite_rowmix import FiniteRowMixture
from .flexible_rowmix import FlexibleRowMixture


def transition_rows(cgpm_mixture, rowid, rng):
    """Performs a Gibbs step on the rowid in the given cgpm_mixture."""
    assert isinstance(cgpm_mixture, (FiniteRowMixture, FlexibleRowMixture))
    observation, inputs = cgpm_mixture.unincorporate(rowid)
    zs = cgpm_mixture.cgpm_row_divide.support()
    logps = []
    for z in zs:
        observation[cgpm_mixture.cgpm_row_divide.outputs[0]] = z
        logp_z = cgpm_mixture.logpdf(None, observation, None, inputs)
        logps.append(logp_z)
    assignment = log_pflip(logps, array=zs, rng=rng)
    observation[cgpm_mixture.cgpm_row_divide.outputs[0]] = assignment
    cgpm_mixture.incorporate(rowid, observation, inputs)

def get_rowids(cgpm_mixture):
    return cgpm_mixture.cgpm_row_divide.data.keys()
