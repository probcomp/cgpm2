# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.


import numpy as np
import yaml

from cStringIO import StringIO
from cgpm.utils.general import get_prng


def sample_hyperparameter(_distribution, _hyper, rng):
    """Dummy function to fetch hyperparameters for univariate distribution."""
    return round(rng.gamma(1,1), 6)


def generate_random_hyperparameters(distribution, rng):
    """Dispatcher for sampling hyperparameters, based on the distribution."""
    distname, _distargs = distribution
    if distname == 'normal':
        return {
          'm'       : sample_hyperparameter('normal', 'm', rng),
          'r'       : sample_hyperparameter('normal', 'r', rng),
          's'       : sample_hyperparameter('normal', 's', rng),
          'nu'      : sample_hyperparameter('normal', 'nu', rng),
        }
    elif distname == 'categorical':
        return  {
          'alpha'   : sample_hyperparameter('categorical', 'alpha', rng),
        }
    elif distname == 'poisson':
        return {
          'a'       : sample_hyperparameter('poisson', 'a', rng),
          'b'       : sample_hyperparameter('poisson', 'b', rng),
        }
    elif distname == 'crp':
        return {
            'alpha' : sample_hyperparameter('crp', 'alpha', rng),
        }
    else:
        assert False, 'Unknown distribution'


def generate_random_partition(alpha, N, rng):
    """Randomly samples a partition of [N], distributed as CRP(alpha)."""
    partition = []
    for customer in xrange(N):
        weights = np.array([len(block) for block in partition] + [alpha])
        probs = weights/float(sum(weights))
        assignment = rng.choice(range(len(weights)), p=probs)
        if len(partition) <= assignment:
            partition.append([])
        partition[assignment].append(customer)
    return partition


def generate_random_row_divider(rng):
    """Randomly sample hyperparameter for CRP in each view of the partition."""
    distribution = ('crp', None)
    return (distribution, generate_random_hyperparameters(distribution, rng))


def generate_random_ast(distributions, rng):
    """End-to-end simulator for AST of Core DSL."""
    partition_alpha = rng.gamma(1,1)
    partition = generate_random_partition(partition_alpha, len(distributions), rng)
    row_dividers = [generate_random_row_divider(rng) for _i in partition]
    primitives = [
        (d, generate_random_hyperparameters(d, rng))
        for d in distributions
    ]
    return [
        (row_divider, [primitives[b] for b in block])
        for row_divider, block in zip(row_dividers, partition)
    ]

# AST -> Core DSL compiler

def core_compile_indent(stream, i):
    indentation = ' ' * i
    stream.write(indentation)

def core_compile_key_val(stream, i, k, v):
    core_compile_indent(stream, i)
    stream.write('%s: ' % (k,))
    if isinstance(v, str):
        stream.write('%s\n' % (v,))
    elif float(v) == int(v):
        stream.write('%d\n' % (int(v),))
    else:
        stream.write('%1.4f\n' % (v,))

def core_compile_key(stream, i, k):
    core_compile_indent(stream, i)
    stream.write('%s:\n' % (k,))

def core_compile_hypers(stream, i, hypers):
    core_compile_key(stream, i, 'hypers')
    for k,v in hypers.iteritems():
        core_compile_key_val(stream, i+2, k, v)

def core_compile_distargs(stream, i, distargs):
    core_compile_key(stream, i, 'distargs')
    if distargs:
        for k, v in distargs.iteritems():
            core_compile_key_val(stream, i+2, k, v)

def core_compile_distribution(stream, i, distribution):
    ((distname, distargs), hypers) = distribution
    core_compile_key(stream, i, distname)
    core_compile_distargs(stream, i+2, distargs)
    core_compile_hypers(stream, i+2, hypers)

def core_compile_distributions(stream, i, distributions):
    core_compile_key(stream, i, 'distribution models')
    for distribution in distributions:
        core_compile_distribution(stream, i+2, distribution)

def core_compile_row_clustering(stream, i, distribution):
    core_compile_key(stream, i, 'row clustering model')
    core_compile_distribution(stream, i+2, distribution)

def core_compile_view(stream, i, c, ast_view):
    row_clustering, distributions = ast_view
    stream.write('%s %d:\n' % ('view', c,))
    core_compile_row_clustering(stream, i+2, row_clustering)
    core_compile_distributions(stream, i+2, distributions)

def compile_ast_to_core_dsl(ast, stream=None):
    stream = stream or StringIO()
    for c, ast_view in enumerate(ast):
        core_compile_view(stream, 0, c, ast_view)
    return stream


# Parser: Core DSL -> AST.

def core_parse_distribution(distname, distargs, hypers):
    return ((distname, distargs), hypers)

def core_parse_distributions(distributions):
    return [core_parse_distribution(k, v['distargs'], v['hypers'])
        for k,v in distributions.iteritems()]

def core_parse_view(yaml_view):
    row_clustering = yaml_view['row clustering model']
    distributions = yaml_view['distribution models']
    ast_crp = core_parse_distributions(row_clustering)[0]
    ast_distributions = core_parse_distributions(distributions)
    return [ast_crp, ast_distributions]

def parse_core_dsl_to_ast(core_dsl):
    core_dsl_yaml = yaml.load(core_dsl)
    return [core_parse_view(yaml_view) for yaml_view in core_dsl_yaml.values()]

# Core DSL -> Embedded DSL compiler.

primitive_constructors = {
    'categorical'   : 'Categorical',
    'crp'           : 'CRP',
    'normal'        : 'Normal',
    'poisson'       : 'Poisson',
}

imports = [
    'from cgpm2.categorical import Categorical',
    'from cgpm2.crp import CRP',
    'from cgpm2.flexible_rowmix import FlexibleRowMixture',
    'from cgpm2.normal import Normal',
    'from cgpm2.poisson import Poisson',
    'from cgpm2.product import Product',
]

generate_output = iter(xrange(10**4))

def embedded_compile_kwarg(stream, i, k, v):
    core_compile_indent(stream, i)
    stream.write('%s=%s,' % (k,v))

def embedded_compile_primitive(stream, i, distname, kwargs):
    core_compile_indent(stream, i)
    constructor = primitive_constructors[distname]
    stream.write('%s(' % (constructor))
    embedded_compile_kwarg(stream, 0, 'outputs', [next(generate_output)])
    embedded_compile_kwarg(stream, 1, 'inputs', [])
    for k, v in kwargs.iteritems():
        if v is not None:
            embedded_compile_kwarg(stream, 1, k, v)
    stream.write(')')

def embedded_compile_product(stream, i, distributions):
    core_compile_indent(stream, i)
    stream.write('Product(cgpms=[\n')
    for k, (distname, kwargs) in enumerate(distributions.iteritems()):
        embedded_compile_primitive(stream, i+4, distname, kwargs)
        stream.write(',')
        if k < len(distributions) - 1:
            stream.write('\n')
    stream.write('])')

def embedded_compile_row_mixture(stream, i, v, ast_view):
    row_clustering = ast_view['row clustering model'].items()[0]
    distributions = ast_view['distribution models']
    core_compile_indent(stream, i)
    stream.write('view%d = FlexibleRowMixture(\n' % (v,))
    core_compile_indent(stream, i+2)
    stream.write('cgpm_row_divide=')
    embedded_compile_primitive(stream, 0, row_clustering[0], row_clustering[1])
    stream.write(',\n')
    core_compile_indent(stream, i+2)
    stream.write('cgpm_components_base=')
    embedded_compile_product(stream, 0, distributions)
    stream.write('\n')
    stream.write(')')

def compile_core_dsl_to_embedded_dsl(core_dsl, stream=None):
    stream = stream or StringIO()
    core_dsl_yaml = yaml.load(core_dsl)
    for import_stmt in imports:
        stream.write('%s\n' % (import_stmt,))
    stream.write('\n')
    for v, ast_view in enumerate(core_dsl_yaml.itervalues()):
        embedded_compile_row_mixture(stream, 0, v, ast_view)
        stream.write('\n')
    views = ', '.join('view%d' % (v,) for v in xrange(len(core_dsl_yaml)))
    stream.write('crosscat = Product(cgpms=[%s])' % (views,))
    return stream

distributions = [
    ('normal', None),
    ('normal', None),
    ('poisson', None),
    ('categorical', {'k':3}),
    ('categorical', {'k':10}),
]

rng = get_prng(1)
