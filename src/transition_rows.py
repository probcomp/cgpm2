# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from .finite_rowmix import FiniteRowMixture
from .flexible_rowmix import FlexibleRowMixture

from .utils import log_pflip

def transition_rows(cgpm_mixture, rowid, rng):
    """Performs a Gibbs step on the rowid in the given cgpm_mixture."""
    assert isinstance(cgpm_mixture, (FiniteRowMixture, FlexibleRowMixture))
    observation, inputs = cgpm_mixture.unobserve(rowid)
    zs = cgpm_mixture.cgpm_row_divide.support()
    logps = []
    for z in zs:
        observation[cgpm_mixture.cgpm_row_divide.outputs[0]] = z
        logp_z = cgpm_mixture.logpdf(None, observation, None, inputs)
        logps.append(logp_z)
    assignment = log_pflip(logps, array=zs, rng=rng)
    observation[cgpm_mixture.cgpm_row_divide.outputs[0]] = assignment
    cgpm_mixture.observe(rowid, observation, inputs)

def get_rowids(cgpm_mixture):
    """Return list of observed rowids in the cgpm_mixture."""
    # XXX TODO: Below is an optimization, implement a general recursive
    # function in walks.py that follows the trail of composition.
    return cgpm_mixture.cgpm_row_divide.data.keys()

def set_rowid_component(cgpm_mixture, rowid0, rowid1):
    """Move rowid0 to component of rowid1 (use None for a singleton)."""
    assert isinstance(cgpm_mixture, (FiniteRowMixture, FlexibleRowMixture))
    if rowid0 is rowid1:
        return
    if rowid1 is None:
        assert isinstance(cgpm_mixture, FlexibleRowMixture)
        observation, inputs = cgpm_mixture.unobserve(rowid0)
        assignment = cgpm_mixture.cgpm_row_divide.support()[-1]
    else:
        assignment = cgpm_mixture.cgpm_row_divide.data[rowid1]
        observation, inputs = cgpm_mixture.unobserve(rowid0)
    observation[cgpm_mixture.cgpm_row_divide.outputs[0]] = assignment
    cgpm_mixture.observe(rowid0, observation, inputs)
