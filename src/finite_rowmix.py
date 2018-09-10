# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np

from cgpm.utils.general import build_cgpm
from cgpm.utils.general import get_intersection
from cgpm.utils.general import get_prng
from cgpm.utils.general import lchain
from cgpm.utils.general import log_normalize
from cgpm.utils.general import log_pflip
from cgpm.utils.general import logsumexp
from cgpm.utils.general import merged

from .finite_array import FiniteArray
from .icgpm import CGPM


class FiniteRowMixture(CGPM):

    def __init__(self, cgpm_row_divide, cgpm_components, rng=None):
        # Assertions.
        assert len(cgpm_row_divide.outputs) == 1
        assert isinstance(cgpm_components, (list, tuple))
        assert range(len(cgpm_components)) == cgpm_row_divide.support()
        # From constructor.
        self.cgpm_row_divide = cgpm_row_divide
        self.rng = rng or get_prng(1)
        # Derived attributes.
        self.outputs_z = cgpm_row_divide.outputs
        self.inputs_z = cgpm_row_divide.inputs
        self.outputs_x = cgpm_components[0].outputs
        self.inputs_x = cgpm_components[0].inputs
        self.outputs = lchain(self.outputs_z, self.outputs_x)
        self.inputs = lchain(self.inputs_z, self.inputs_x)
        self.indexer = self.outputs[0]
        # Internal attributes.
        self.rowid_to_component = {}
        self.cgpm_components_array = FiniteArray(
            cgpm_components, self.indexer, self.rng)

    def simulate(self, rowid, targets, constraints=None, inputs=None, N=None):
        if rowid in self.rowid_to_component:
            assert not constraints or self.indexer not in constraints
            z = self.rowid_to_component[rowid]
            return self._simulate_one(rowid, targets, constraints, inputs, N, z)
        elif constraints and self.indexer in constraints:
            z = constraints[self.indexer]
            if z not in self.cgpm_row_divide.support():
                raise ValueError('Constrained cluster has 0 density: %s' % (z,))
            return self._simulate_one(rowid, targets, constraints, inputs, N, z)
        z_support = self.cgpm_row_divide.support()
        z_weights = [self.logpdf(rowid, {self.indexer: z}, constraints, inputs)
            for z in z_support]
        zs = log_pflip(z_weights, array=z_support, size=(N or 1), rng=self.rng)
        counts = {z:n for z,n in enumerate(np.bincount(zs)) if n}
        samples = [self._simulate_one(rowid, targets, constraints, inputs, n, z)
            for z, n in counts.iteritems()]
        return samples[0][0] if N is None else lchain(*samples)

    def logpdf(self, rowid, targets, constraints=None, inputs=None):
        if rowid in self.rowid_to_component:
            # Condition on the cluster assignment directly.
            # p(xT|xC,z=k)
            assert not constraints or self.indexer not in constraints
            z = self.rowid_to_component[rowid]
            return self._logpdf_one(rowid, targets, constraints, inputs, z)
        elif self.indexer in targets:
            # Query the cluster assignment.
            # p(z=k,xT|xC)
            # = p(z=k,xT,xC) / p(xC)            Bayes rule
            # = p(z=k)p(xT,xC|z=k) / p(xC)      chain rule on numerator
            # The terms are then:
            # p(z=k)                            lp_z
            # p(xT,xC|z=k)                      lp_x_joint
            # p(xC) = \sum_z P(xC,z)            lp_x_constraints (recursively)
            z = targets[self.indexer]
            inputs_z = get_intersection(self.inputs_z, inputs)
            lp_z = self.cgpm_row_divide.logpdf(
                rowid=rowid,
                targets={self.indexer: z},
                constraints=None,
                inputs=inputs_z)
            targets_joint = merged(targets, constraints or {})
            lp_x_joint = self._logpdf_one(
                rowid=rowid,
                targets=targets_joint,
                constraints=None,
                inputs=inputs,
                component=z
            )
            lp_x_constraints = self.logpdf(
                rowid=rowid,
                targets=constraints,
                constraints=None,
                inputs= inputs
            ) if constraints else 0
            return (lp_z + lp_x_joint) - lp_x_constraints
        elif constraints and self.indexer in constraints:
            # Condition on the cluster assignment
            # P(xT|xC,z=k)
            # = P(xT,xC,z=k) / P(xC,z=k)
            # = P(xT,xC|z=k)P(z=k) / P(xC|z=k)
            # = P(xT,xC|z=k) / P(xC|z=k)
            # The terms are then:
            # P(xT,xC|z=k)                  lp_x_joint
            # P(xC|z=k)                     lp_x_constraints
            z = constraints[self.indexer]
            if z not in self.cgpm_row_divide.support():
                raise ValueError('Constrained cluster has 0 density: %s' % (z,))
            targets_joint = merged(targets, constraints)
            lp_x_joint = self._logpdf_one(
                rowid=rowid,
                targets=targets_joint,
                constraints=None,
                inputs=inputs,
                component=z
            )
            lp_x_constraints = self._logpdf_one(
                rowid=rowid,
                targets=constraints,
                constraints=None,
                inputs=inputs,
                component=z
            )
            return lp_x_joint - lp_x_constraints
        else:
            # Marginalize over cluster assignment by enumeration.
            # Let K be a list of values for the support of z:
            # P(xT|xC)
            # = \sum_i P(xT,z=K[i]|xC)
            # = \sum_i P(xT|xC,z=K[i])P(z=K[i]|xC)  chain rule
            #
            # The posterior is given by:
            # P(z=K[i]|xC) = P(xC|z=K[i])P(z=K[i]) / \sum_i P(xC,z=K[i])
            #
            # The terms are therefore
            # P(z=K[i])                            lp_z_prior[i]
            # P(xC|z=K[i])                         lp_constraints_likelihood[i]
            # P(xC,z=K[i])                         lp_z_constraints[i]
            # P(z=K[i]|xC)                         lp_z_posterior[i]
            # P(xT|xC,z=K[i])                      lp_targets_likelihood[i]
            # P(xT|xC,z=K[i])P(z=K[i]|xC)          lp_joint[i]
            inputs_z = get_intersection(self.inputs_z, inputs)
            z_support = self.cgpm_row_divide.support()
            lp_z_prior = [
                self.cgpm_row_divide.logpdf(
                    rowid, {self.indexer: z}, None, inputs_z)
                for z in z_support
            ]
            lp_constraints_likelihood = [
                self._logpdf_one(rowid, constraints, None, inputs, z)
                for z in z_support
            ]
            lp_z_constraints = np.add(lp_z_prior, lp_constraints_likelihood)
            lp_z_posterior = log_normalize(lp_z_constraints)
            lp_targets_likelihood = [
                self._logpdf_one(rowid, targets, constraints, inputs, z)
                for z in z_support
            ]
            lp_joint = np.add(lp_targets_likelihood, lp_z_posterior)
            return logsumexp(lp_joint)

    def logpdf_score(self):
        score_z = self.cgpm_row_divide.logpdf_score()
        score_x = self.cgpm_components_array.logpdf_score()
        return score_z + score_x

    def observe(self, rowid, observation, inputs=None):
        if rowid in self.rowid_to_component:
            component = {self.indexer: self.rowid_to_component[rowid]}
        else:
            inputs_z = get_intersection(self.inputs_z, inputs)
            if self.indexer in observation:
                component = {self.indexer: observation[self.indexer]}
            else:
                component = self.cgpm_row_divide.simulate(
                    rowid, [self.indexer], inputs_z)
            inputs_z = get_intersection(self.inputs_z, inputs)
            self.cgpm_row_divide.observe(rowid, component, inputs_z)
            self.rowid_to_component[rowid] = component[self.indexer]
        inputs_x = get_intersection(self.inputs_x, inputs)
        observation_x = get_intersection(self.outputs_x, observation)
        inputs_arr = merged(inputs_x, component)
        self.cgpm_components_array.observe(rowid, observation_x, inputs_arr)

    def unobserve(self, rowid):
        obs_z, inputs_z = self.cgpm_row_divide.unobserve(rowid)
        obs_x, inputs_x = self.cgpm_components_array.unobserve(rowid)
        del self.rowid_to_component[rowid]
        observation = merged(obs_z, obs_x)
        inputs = merged(inputs_z, inputs_x)
        return observation, inputs

    def transition(self, **kwargs):
        return

    def to_metadata(self):
        metadata = dict()
        metadata['cgpm_row_divide'] = self.cgpm_row_divide.to_metadata()
        metadata['cgpm_components_array'] = \
            self.cgpm_components_array.to_metadata()
        metadata['rowid_to_component'] = self.rowid_to_component.items()
        metadata['factory'] = ('cgpm2.finite_rowmix', 'FiniteRowMixture')
        return metadata

    @classmethod
    def from_metadata(cls, metadata, rng):
        cgpm_row_divide = build_cgpm(metadata['cgpm_row_divide'], rng)
        cgpm_components_array = \
            build_cgpm(metadata['cgpm_components_array'], rng)
        model = cls(cgpm_row_divide, cgpm_components_array.cgpms, rng)
        model.rowid_to_component = dict(metadata['rowid_to_component'])
        model.cgpm_components_array = cgpm_components_array
        return model

    def render(self):
        return [
            'FiniteRowMixture',
            ['cgpm_row_divide=', self.cgpm_row_divide.render()],
            ['cgpm_components=', self.cgpm_components_array.render()],
        ]

    # Helpers

    def _simulate_one(self, rowid, targets, constraints, inputs, N, component):
        """Simulate from a fixed mixture component."""
        targets_x = get_intersection(self.outputs_x, targets)
        if targets_x:
            constraints_x = get_intersection(self.outputs_x, constraints)
            inputs_x = get_intersection(self.outputs_x, inputs)
            inputs_arr = merged(inputs_x, {self.indexer: component})
            samples = self.cgpm_components_array.simulate(
                rowid=rowid,
                targets=targets_x,
                constraints=constraints_x,
                inputs=inputs_arr,
                N=N,
            )
        else:
            samples = {} if N is None else [{}]*N
        if N is None and self.indexer in targets:
            samples[self.indexer] = component
        elif N is not None and self.indexer in targets:
            for sample in samples:
                sample[self.indexer] = component
        return samples

    def _logpdf_one(self, rowid, targets, constraints, inputs, component):
        """Assess logpdf in fixed mixture component."""
        targets_x = get_intersection(self.outputs_x, targets)
        if not targets_x:
            return 0
        constraints_x = get_intersection(self.outputs_x, constraints)
        inputs_x = get_intersection(self.outputs_x, inputs)
        inputs_arr = merged(inputs_x, {self.indexer: component})
        return self.cgpm_components_array.logpdf(
            rowid=rowid,
            targets=targets_x,
            constraints=constraints_x,
            inputs=inputs_arr,
        )
