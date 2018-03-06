# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict
from math import log

from scipy.special import gammaln

from cgpm.utils.general import get_prng
from cgpm.utils.general import log_linspace
from cgpm.utils.general import log_pflip
from cgpm.utils.general import simulate_many

from .distribution import DistributionCGPM


class CRP(DistributionCGPM):

    def __init__(self, outputs, inputs, hypers=None, params=None,
            distargs=None, rng=None):
        # Populate default kwargs.
        hypers = hypers or dict()
        params = params or dict()
        distargs = distargs or dict()
        # From constructor.
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.params = params
        self.rng = rng or get_prng()
        # Internal attributes.
        self.data = OrderedDict()
        self.N = 0
        self.counts = OrderedDict()
        self.alpha = hypers.get('alpha', 1.)

    def incorporate(self, rowid, observation, inputs=None):
        DistributionCGPM.incorporate(self, rowid, observation, inputs)
        x = int(observation[self.outputs[0]])
        assert x in self.support()
        self.N += 1
        if x not in self.counts:
            self.counts[x] = 0
        self.counts[x] += 1
        self.data[rowid] = x

    def unincorporate(self, rowid):
        DistributionCGPM.unincorporate(self, rowid)
        x = self.data.pop(rowid)
        self.N -= 1
        self.counts[x] -= 1
        if self.counts[x] == 0:
            del self.counts[x]
        return {self.outputs[0]: x}, {}

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        # Do not call DistributionCGPM.logpdf since crp allows observed rowid.
        assert not inputs
        assert not constraints
        assert targets.keys() == self.outputs
        x = int(targets[self.outputs[0]])
        if x not in self.support():
            # TODO: Optimize this computation by caching valid tables.
            return float('-inf')
        if rowid not in self.data:
            return calc_predictive_logp(x, self.N, self.counts, self.alpha)
        elif self.data[rowid] == x:
            return 0
        elif self.data[rowid] != x:
            return -float('inf')
        else:
            assert False, 'Unknown failure'

    @simulate_many
    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        DistributionCGPM.simulate(self, rowid, targets, constraints, inputs, N)
        if rowid not in self.data:
            K = self.support()
            logps = [self.logpdf(rowid, {targets[0]: x}, None) for x in K]
            x = log_pflip(logps, array=K, rng=self.rng)
        else:
            x = self.data[rowid]
        return {self.outputs[0]: x}

    def logpdf_score(self):
        return calc_logpdf_marginal(self.N, self.counts.values(), self.alpha)

    def to_metadata(self):
        metadata = dict()
        metadata['outputs'] = self.outputs
        metadata['inputs'] = self.inputs
        metadata['N'] = self.N
        metadata['data'] = self.data.items()
        metadata['counts'] = self.counts.items()
        metadata['alpha'] = self.alpha
        metadata['factory'] = ('cgpm2.crp', 'CRP')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        model = cls(metadata['outputs'], metadata['inputs'], rng=rng)
        model.data = OrderedDict(metadata['data'])
        model.N = metadata['N']
        model.counts = OrderedDict(metadata['counts'])
        model.alpha = metadata['alpha']
        return model

    # DistributionCGPM methods.

    def transition_params(self):
        return

    def set_hypers(self, hypers):
        assert hypers['alpha'] > 0
        self.alpha = hypers['alpha']

    def get_hypers(self):
        return {'alpha': self.alpha}

    def get_params(self):
        return self.params

    def get_suffstats(self):
        return {'N': self.N, 'counts': list(self.counts)}

    def get_distargs(self):
        return {}

    def support(self):
        K = sorted(self.counts)
        return K + [max(K)+1] if K else [0]

    # Some Gibbs utilities.

    def gibbs_logps(self, rowid, m=1):
        """Compute the CRP probabilities for a Gibbs transition of rowid,
        with table counts Nk, table assignments Z, and m auxiliary tables."""
        assert rowid in self.data
        assert 0 < m
        singleton = self.is_singleton(rowid)
        p_aux = self.alpha / float(m)
        p_rowid = p_aux if singleton else self.counts[self.data[rowid]]-1
        tables = self.gibbs_tables(rowid, m=m)
        def p_table(t):
            if t == self.data[rowid]: return p_rowid    # rowid table.
            if t not in self.counts: return p_aux       # auxiliary table.
            return self.counts[t]                       # regular table.
        return [log(p_table(t)) for t in tables]

    def gibbs_tables(self, rowid, m=1):
        """Retrieve a list of possible tables for rowid.

        If rowid is an existing customer, then the standard Gibbs proposal
        tables  are returned (i.e. with the rowid unincorporated). If
        rowid was a singleton table, then the table is re-used as a proposal
        and m-1 additional auxiliary tables are proposed, else m auxiliary
        tables are returned.

        If rowid is a new customer, then the returned tables are from the
        predictive distribution, (using m auxiliary tables always).
        """
        assert 0 < m
        K = sorted(self.counts)
        singleton = self.is_singleton(rowid)
        m_aux = m - 1 if singleton else m
        t_aux = [max(self.counts) + 1 + m for m in range(m_aux)]
        return K + t_aux

    def is_singleton(self, rowid):
        return self.counts[self.data[rowid]] == 1 if rowid in self.data else 0

    @staticmethod
    def construct_hyper_grids(X, n_grid=30):
        grids = dict()
        grids['alpha'] = log_linspace(1./len(X), len(X), n_grid)
        return grids

    @staticmethod
    def name():
        return 'crp'

    @staticmethod
    def is_collapsed():
        return True

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_conditional():
        return False

    @staticmethod
    def is_numeric():
        return False


def calc_predictive_logp(x, N, counts, alpha):
    numerator = counts.get(x, alpha)
    denominator = N + alpha
    return log(numerator) - log(denominator)

def calc_logpdf_marginal(N, counts, alpha):
    # http://gershmanlab.webfactional.com/pubs/GershmanBlei12.pdf#page=4 (eq 8)
    return len(counts) * log(alpha) + sum(gammaln(counts)) \
        + gammaln(alpha) - gammaln(N + alpha)
