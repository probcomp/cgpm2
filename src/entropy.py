# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.


import itertools

from collections import namedtuple

import numpy

from cgpm.utils.general import merged

DEFAULT_SAMPLES_MONTE_CARLO = 100
DEFAULT_SAMPLES_MARGINALIZE = 100

Estimate = namedtuple('Estimate', ['mean', 'variance'])

def get_estimate(samples):
    mean = numpy.mean(samples)
    variance = numpy.var(samples) / len(samples)
    return Estimate(mean, variance)

def _validate_query(outputs, targets0, targets1, constraints, marginalize):
    set_t0 = set(targets0)
    set_t1 = set(targets1)
    set_t = set_t0 | set_t1
    set_c = set(constraints or [])
    set_m = set(marginalize or [])
    set_o = set(outputs)
    # Assert all variables exist in the cgpm's outputs.
    set_uk = ((set_t | set_c | set_m) - set_o)
    if set_uk:
        raise ValueError('Unknown output variables: %s' % (set_uk,))
    # Disallow duplicated variables in targets and constraints.
    set_tc = set_t & set_c
    if set_tc:
        raise ValueError('Overlap, targets and constraints: %s' % (set_tc,))
    # Disallow duplicated variables in targets and marginalize.
    set_tm = set_t & set_m
    if set_tm:
        raise ValueError('Overlap, targets and marginalize: %s' % (set_tm,))
    # Disallow duplicated variables in constraints and marginalize.
    set_cm = set_c & set_m
    if set_cm:
        raise ValueError('Overlap, constraints and marginalize: %s' % (set_cm,))
    # Disallow duplicated variables in targets (except exact match for entropy).
    set_t01 = (set_t0 & set_t1)
    if (set_t0 != set_t1) and set_t01:
        raise ValueError('Overlap, targets0 and targets1: %s' % (set_t01,))

def _get_estimator(targets0, targets1):
    return _sample_entropy if set(targets0) == set(targets1) else _sample_mi

def _sample_entropy(cgpm, targets0, targets1, constraints, N):
    assert set(targets0) == set(targets1)
    samples = cgpm.simulate(None, targets0, constraints, None, N)
    PX = [cgpm.logpdf(None, sample, constraints) for sample in samples]
    return [-p for p in PX]

def _sample_mi(cgpm, targets0, targets1, constraints, N):
    samples = cgpm.simulate(None, targets0 + targets1, constraints, None, N)
    samples_t0 = [{t: sample[t] for t in targets0} for sample in samples]
    samples_t1 = [{t: sample[t] for t in targets1} for sample in samples]
    PXY = [cgpm.logpdf(None, sample, constraints) for sample in samples]
    PX = [cgpm.logpdf(None, sample, constraints) for sample in samples_t0]
    PY = [cgpm.logpdf(None, sample, constraints) for sample in samples_t1]
    return [pxy - px - py for (pxy, px, py) in zip(PXY, PX, PY)]

def mutual_information(cgpm, targets0, targets1, constraints=None,
        marginalize=None, T=None, N=None):
    _validate_query(cgpm.outputs,targets0, targets1, constraints, marginalize)
    N = N or DEFAULT_SAMPLES_MONTE_CARLO
    T = T or DEFAULT_SAMPLES_MARGINALIZE
    estimator = _get_estimator(targets0, targets1)
    if not marginalize:
        samples_mi = estimator(cgpm, targets0, targets1, constraints, N)
    else:
        samples_marginalize = cgpm.simulate(None, marginalize, N=T)
        constraints_cm = [merged(constraints, m) for m in samples_marginalize]
        estimates = [estimator(cgpm, targets0, targets1, constraint_cm, N)
            for constraint_cm in constraints_cm]
        samples_mi = itertools.chain.from_iterable(estimates)
    return get_estimate(samples_mi)
