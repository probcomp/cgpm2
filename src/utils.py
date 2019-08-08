# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import importlib
import itertools

from math import exp
from math import fabs
from math import isinf
from math import isnan
from math import log

import numpy as np

def build_cgpm(metadata, rng):
    modname, attrname = metadata['factory']
    module = importlib.import_module(modname)
    builder = getattr(module, attrname)
    return builder.from_metadata(metadata, rng)

def dummy_code(x, discretes):
    """Dummy code a vector of covariates x.

    Parameters
    ----------
    x : list
        List of data. Categorical values must be integer starting at 0.
    discretes : dict{int:int}
        discretes[i] is the number of discrete categories in x[i].

    Returns
    -------
    xd : list
        Dummy coded version of x as list.

    Example
    -------
    >>> dummy_code([12.1, 3], {1:5})
    [12.1, 0, 0, 0, 1]
    # Note only 4 dummy codes since all 0000 indicates last category.
    """
    if len(discretes) == 0:
        return list(x)
    def as_code(i, val):
        if i not in discretes:
            return [val]
        if float(val) != int(val):
            raise TypeError('Discrete value must be integer: {},{}'.format(x,i))
        k = discretes[i]
        if not 0 <= val < k:
            raise ValueError('Discrete value not in {0..%s}: %d.' % (k-1, val))
        r = [0]*(k-1)
        if val < k-1:
            r[int(val)] = 1
        return r
    xp = [as_code(i, val) for i, val in enumerate(x)]
    return list(itertools.chain.from_iterable(xp))

def flatten_cgpms(cgpms, tpe):
    return list(itertools.chain.from_iterable(
        cgpm.cgpms if isinstance(cgpm, tpe) else [cgpm] for cgpm in cgpms
    ))

def get_intersection(left, right):
    if right is None:
        return {} if isinstance(left, dict) else []
    if isinstance(right, dict):
        return {i: right[i] for i in right if i in left}
    elif isinstance(right, list):
        return [i for i in right if i in left]
    else:
        assert False, 'Unknown args type.'

def get_prng(seed=None):
    if seed is None:
        seed = np.random.randint(low=1, high=2**31)
    return np.random.RandomState(seed)

def lchain(*args):
    return list(itertools.chain(*args))

def logsumexp(array):
    # https://github.com/probcomp/bayeslite/blob/master/src/math_util.py
    if len(array) == 0:
        return float('-inf')
    m = max(array)

    # m = +inf means addends are all +inf, hence so are sum and log.
    # m = -inf means addends are all zero, hence so is sum, and log is
    # -inf.  But if +inf and -inf are among the inputs, or if input is
    # NaN, let the usual computation yield a NaN.
    if isinf(m) and min(array) != -m and \
       all(not isnan(a) for a in array):
        return m

    # Since m = max{a_0, a_1, ...}, it follows that a <= m for all a,
    # so a - m <= 0; hence exp(a - m) is guaranteed not to overflow.
    return m + log(sum(exp(a - m) for a in array))

def logmeanexp(array):
    # https://github.com/probcomp/bayeslite/blob/master/src/math_util.py
    inf = float('inf')
    if len(array) == 0:
        # logsumexp will DTRT, but math.log(len(array)) will fail.
        return -inf

    # Treat -inf values as log 0 -- they contribute zero to the sum in
    # logsumexp, but one to the count.
    #
    # If we pass -inf values through to logsumexp, and there are also
    # +inf values, then we get NaN -- but if we had averaged exp(-inf)
    # = 0 and exp(+inf) = +inf, we would sensibly get +inf, whose log
    # is still +inf, not NaN.  So strip -inf values first.
    #
    # Can't say `a > -inf' because that excludes NaNs, but we want to
    # include them so they propagate.
    noninfs = [a for a in array if not a == -inf]

    # probs = map(exp, logprobs)
    # log(mean(probs)) = log(sum(probs) / len(probs))
    #   = log(sum(probs)) - log(len(probs))
    #   = log(sum(map(exp, logprobs))) - log(len(logprobs))
    #   = logsumexp(logprobs) - log(len(logprobs))
    return logsumexp(noninfs) - log(len(array))

def log_linspace(a, b, n):
    """linspace from a to b with n entries over log scale."""
    return np.exp(np.linspace(log(a), log(b), n))

def log_normalize(logp):
    """Normalizes a np array of log probabilites."""
    return np.subtract(logp, logsumexp(logp))

def log_pflip(logp, array=None, size=None, rng=None):
    """Categorical draw from a vector logp of log probabilities."""
    p = np.exp(log_normalize(logp))
    return pflip(p, array=array, size=size, rng=rng)

def merged(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def mergedl(dicts):
    return merged(*dicts)

def normalize(p):
    """Normalizes a np array of probabilites."""
    return np.asarray(p, dtype=float) / sum(p)

def pflip(p, array=None, size=None, rng=None):
    """Categorical draw from a vector p of probabilities."""
    if array is None:
        array = range(len(p))
    if len(p) == 1:
        return array[0] if size is None else [array[0]] * size
    if rng is None:
        rng = get_prng()
    p = normalize(p)
    try:
        10**(-8) < fabs(1. - sum(p))
    except:
        import ipdb; ipdb.set_trace()

    return rng.choice(array, size=size, p=p)

def simulate_many(simulate):
    """Simple wrapper for a cgpm `simulate` method to call itself N times."""
    def simulate_wrapper(*args, **kwargs):
        if len(args) == 6:
            N = args[-1]
        else:
            N = kwargs.get('N', None)
        if N is None:
            return simulate(*args, **kwargs)
        return [simulate(*args, **kwargs) for _i in range(N)]
    return simulate_wrapper

def simulate_crp_constrained(N, alpha, Cd, Ci, Rd, Ri, rng=None):
    """Simulates a CRP with N customers and concentration alpha. Cd is a list,
    where each entry is a list of friends. Ci is a list of tuples, where each
    tuple is a pair of enemies."""
    if rng is None:
        rng = get_prng()

    validate_crp_constrained_input(N, Cd, Ci, Rd, Ri)
    assert N > 0 and alpha > 0.

    # Initial partition.
    Z = [-1]*N

    # Friends dictionary from Cd.
    friends = {col: block for block in Cd for col in block}

    # Assign customers.
    for cust in range(N):
        # If the customer has been assigned, skip.
        if Z[cust] > -1:
            continue
        # Find valid tables for cust and friends.
        assert all(Z[f] == -1 for f in friends.get(cust, [cust]))
        prob_table = [0] * (max(Z)+1)
        for t in range(max(Z)+1):
            # Current customers at table t.
            t_custs = [i for i,z in enumerate(Z) if z==t]
            prob_table[t] = len(t_custs)
            # Does f \in {cust \union cust_friends} have an enemy in table t?
            for tc in t_custs:
                for f in friends.get(cust, [cust]):
                    if not check_compatible_customers(Cd,Ci,Ri,Rd,f,tc):
                        prob_table[t] = 0
                        break
        # Choose from valid tables using CRP.
        prob_table.append(alpha)
        assignment = pflip(prob_table, rng=rng)
        for f in friends.get(cust, [cust]):
            Z[f] = assignment

    # At most N tables.
    assert all(0 <= t < N for t in Z)
    assert validate_crp_constrained_partition(Z, Cd, Ci, Rd, Ri)
    return Z

def check_compatible_constraints(Cd1, Ci1, Cd2, Ci2):
    """Returns True if (Cd1, Ci1) is compatible with (Cd2, Ci2)."""
    try:
        validate_dependency_constraints(None, Cd1, Ci1)
        validate_dependency_constraints(None, Cd1, Ci2)
        validate_dependency_constraints(None, Cd2, Ci1)
        validate_dependency_constraints(None, Cd2, Ci2)
        return True
    except ValueError:
        return False

def check_compatible_customers(Cd, Ci, Ri, Rd, a, b):
    """Checks if customers a,b are compatible."""
    # Explicitly independent.
    if (a,b) in Ci or (b,a) in Ci:
        return False
    # Incompatible Rd/Ri constraints.
    if (a in Rd or a in Ri) and (b in Rd or b in Ri):
        return check_compatible_constraints(
            Rd.get(a,[]), Ri.get(a,[]), Rd.get(b,[]), Ri.get(b,[]))
    return True

def validate_dependency_constraints(N, Cd, Ci):
    """Validates Cd and Ci constraints on N columns."""
    # Allow unknown number of customers.
    if N is None:
        N = 1e10
    counts = {}
    for block in Cd:
        # Every constraint must be more than one customer.
        if len(block) == 1:
            raise ValueError('Single customer in dependency constraint.')
        for col in block:
            # Every column must have correct index.
            if N <= col:
                raise ValueError('Dependence customer out of range.')
            # Every column must appear once only.
            if col not in counts:
                counts[col] = 0
            counts[col] += 1
            if counts[col] > 1:
                raise ValueError('Multiple customer dependencies.')
        for pair in Ci:
            # Ci cannot include columns in same Cd block.
            if pair[0] in block and pair[1] in block:
                raise ValueError('Contradictory customer independence.')
    for pair in Ci:
        # Ci entries are tuples only.
        if len(pair) != 2:
            raise ValueError('Independencies require two customers.')
        if N <= pair[0] or N <= pair[1]:
            raise ValueError('Independence customer of out range.')
        # Dummy case.
        if pair[0] == pair[1]:
            raise ValueError('Independency specified for same customer.')
    return True

def validate_crp_constrained_input(N, Cd, Ci, Rd, Ri):
    # First validate outer Cd, Ci.
    validate_dependency_constraints(N, Cd, Ci)
    # Validate all inner Rd, Ri.
    for c in Rd:
        col_dep = Rd[c]
        row_dep = Ri.get(c,{})
        validate_dependency_constraints(None, col_dep, row_dep)
    # For each block in Cd, validate their Rd, Ri are compatible.
    for block in Cd:
        for a,b in itertools.combinations(block, 2):
            if not check_compatible_customers(Cd, Ci, Ri, Rd, a, b):
                raise ValueError('Incompatible row constraints for dep cols.')
    return True

def validate_crp_constrained_partition(Zv, Cd, Ci, Rd, Ri):
    """Only tests the outer CRP partition Zv."""
    valid = True
    for block in Cd:
        valid = valid and all(Zv[block[0]] == Zv[b] for b in block)
        for a, b in itertools.combinations(block, 2):
            valid = valid and check_compatible_customers(Cd, Ci, Ri, Rd, a, b)
    for a, b in Ci:
        valid = valid and not Zv[a] == Zv[b]
    return valid
