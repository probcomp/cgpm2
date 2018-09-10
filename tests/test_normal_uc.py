# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np

from scipy.stats import chi2
from scipy.stats import norm

from cgpm2.normal_uc import NormalUC

def test_basic_simulate_logpdf_score():
    mu = 12.71
    var = 1.312
    n_samples = 1000
    model = NormalUC((0,), (), params={'mu': mu, 'var': var})
    samples = model.simulate(None, [0], N=n_samples)
    mean_samples = np.mean([s[0] for s in samples])
    var_samples = np.var([s[0] for s in samples], ddof=1)
    # The sampling distribution of the sample mean is N(10, 0.3/N).
    # Check it falls within its 99% confidence interval.
    sm_loc = mu
    sm_scale = np.sqrt(var / n_samples)
    ci_mean_lower = norm.ppf(0.005, loc=sm_loc, scale=sm_scale)
    ci_mean_upper = norm.ppf(0.995, loc=sm_loc, scale=sm_scale)
    assert ci_mean_lower < mean_samples < ci_mean_upper
    # The sampling distribution of sample variance is chi2(n-1).
    # Check it falls within its 99% confidence interval.
    # https://onlinecourses.science.psu.edu/stat414/node/174/
    var_samples_transformed = (n_samples-1) * var_samples / var
    ci_var_lower = chi2.ppf(0.005, df=n_samples-1)
    ci_var_upper = chi2.ppf(0.995, df=n_samples-1)
    assert ci_var_lower < var_samples_transformed < ci_var_upper

    logp_likelihood_manual = 0
    assert np.allclose(model.logpdf_score(), logp_likelihood_manual)

    for rowid, sample in enumerate(samples):
        logp_model = model.logpdf(None, sample)
        logp_manual = norm.logpdf(sample[0], loc=mu, scale=np.sqrt(var))
        assert np.allclose(logp_model, logp_manual)

        model.observe(rowid, sample)
        logp_likelihood_manual += logp_manual

    logp_likelihood_model = model.logpdf_score()
    assert np.allclose(logp_likelihood_manual, logp_likelihood_model)

    metadata = model.to_metadata()
    model2 = NormalUC.from_metadata(metadata, model.rng)
    logp_likelihood_model2 = model2.logpdf_score()
    assert np.allclose(logp_likelihood_manual, logp_likelihood_model2)
