# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from .icgpm import CGPM

class DistributionCGPM(CGPM):
    """Interface for generative population models representing univariate
    probability distribution.

    A typical DistributionGpm will have:
    - Sufficient statistics T, for the observed data X.
    - Parameters Q, for the likelihood p(X|Q).
    - Hyperparameters H, for the prior p(Q|H).

    Additionally, some DistributionGpms will require per query
    - Conditioning variables Y, for the distribution p(X|Q,H,Y=y).

    This interface is uniform for both collapsed and uncollapsed models.
    A collapsed model will typically have no parameters Q.
    An uncollapsed model will typically carry a single set of parameters Q,
    but may also carry an ensemble of parameters (Q1,...,Qn) for simple
    Monte Carlo averaging of queries. The collapsed case is "theoretically"
    recovered in the limit n \to \infty.
    """

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        raise NotImplementedError()

    def observe(self, rowid, observation, inputs=None):
        if rowid in self.data:
            raise ValueError('rowid already exists: %d' % (rowid,))
        assert not inputs
        assert list(observation) == self.outputs

    def unobserve(self, rowid):
        if rowid not in self.data:
            raise ValueError('no such rowid: %s' % (repr(rowid)),)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        assert rowid not in self.data
        assert not inputs
        assert not constraints
        assert list(targets) == self.outputs

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        assert not constraints
        assert not inputs
        assert targets == self.outputs

    def transition_hypers(self, N=None):
        """Resample the hyperparameters H conditioned on all observations X
        from an approximate posterior P(H|X,Q)."""
        raise NotImplementedError()

    def transition_params(self):
        """Resample the parameters Q conditioned on all observations X
        from an approximate posterior P(Q|X,H)."""
        raise NotImplementedError

    def set_hypers(self, hypers):
        """Force the hyperparameters H to new values."""
        raise NotImplementedError

    def get_hypers(self):
        """Return a dictionary of hyperparameters."""
        raise NotImplementedError

    def set_params(self, params):
        """Force the parameters Q to new values."""

    def get_params(self):
        """Return a dictionary of parameters."""
        raise NotImplementedError

    def get_suffstats(self):
        """Return a dictionary of sufficient statistics."""
        raise NotImplementedError

    def get_distargs(self):
        """Return a dictionary of distribution arguments."""
        raise NotImplementedError

    def render(self):
        """Return an AST-like representation of the CGPM."""
        return [
            '%s%s' % (self.name()[0].upper(), self.name()[1:],),
            ['outputs=', self.outputs],
            ['inputs=', self.inputs],
            ['distargs=', self.get_distargs()],
            ['params=', self.get_params()],
            ['hypers=', self.get_hypers()],
            ['suffstats=', self.get_suffstats()],
        ]

    @staticmethod
    def construct_hyper_grids(X, n_grid=20):
        """Return dictionary, where grids['hyper'] is a list of
        grid points for the binned hyperparameter distribution.

        This method is included in the interface since each GPM knows the
        valid values of its hypers, and may also use data-dependent
        heuristics from X to create better grids.
        """
        raise NotImplementedError

    @staticmethod
    def name():
        """Return the name of the distribution as a string."""
        raise NotImplementedError

    @staticmethod
    def is_collapsed():
        """Is the sampler collapsed?"""
        raise NotImplementedError

    @staticmethod
    def is_continuous():
        """Is the pdf defined on a continuous set?"""
        raise NotImplementedError

    @staticmethod
    def is_conditional():
        """Does the sampler require conditioning variables Y=y?"""
        raise NotImplementedError

    @staticmethod
    def is_numeric():
        """Is the support of the pdf a numeric or a symbolic set?"""
        raise NotImplementedError
