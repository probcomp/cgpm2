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
        (i, d, generate_random_hyperparameters(d, rng))
        for i, d in enumerate(distributions)
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

def core_compile_key_list(stream, i, k):
    core_compile_indent(stream, i)
    stream.write('- %s:\n' % (k,))

def core_compile_hypers(stream, i, hypers):
    core_compile_key(stream, i, 'hypers')
    for k,v in hypers.iteritems():
        core_compile_key_val(stream, i+2, k, v)

def core_compile_distargs(stream, i, distargs):
    core_compile_key(stream, i, 'distargs')
    if distargs:
        for k, v in distargs.iteritems():
            core_compile_key_val(stream, i+2, k, v)

def core_compile_distribution_indexed(stream, i, distribution):
    (column, (distname, distargs), hypers) = distribution
    core_compile_key_list(stream, i, '%s{%d}' % (distname, column))
    core_compile_distargs(stream, i+4, distargs)
    core_compile_hypers(stream, i+4, hypers)

def core_compile_distribution_unindexed(stream, i, distribution):
    ((distname, distargs), hypers) = distribution
    core_compile_key_list(stream, i, distname)
    core_compile_distargs(stream, i+4, distargs)
    core_compile_hypers(stream, i+4, hypers)

def core_compile_distributions(stream, i, distributions):
    core_compile_key(stream, i, 'distribution models')
    for distribution in distributions:
        core_compile_distribution_indexed(stream, i+2, distribution)

def core_compile_row_clustering(stream, i, distribution):
    core_compile_key(stream, i, 'row clustering model')
    core_compile_distribution_unindexed(stream, i+2, distribution)

def core_compile_view(stream, i, ast_view):
    row_clustering, distributions = ast_view
    core_compile_key_list(stream, i, 'view')
    core_compile_row_clustering(stream, i+4, row_clustering)
    core_compile_distributions(stream, i+4, distributions)

def compile_ast_to_core_dsl(ast, stream=None):
    stream = stream or StringIO()
    for ast_view in ast:
        core_compile_view(stream, 0, ast_view)
    return stream


# Parser: Core DSL -> AST.

def core_parse_distribution_indexed(distname, distargs, hypers):
    return ((distname, distargs), hypers)

def core_parse_distribution(ast_distribution):
    assert len(ast_distribution) == 1
    distname = ast_distribution.keys()[0]
    distargs = ast_distribution.values()[0]['distargs']
    hypers = ast_distribution.values()[0]['hypers']
    if '{' in distname:
        distname, index = distname.replace('}','').split('{')
        return (int(index), (distname, distargs), hypers)
    else:
        return ((distname, distargs), hypers)

def core_parse_view(yaml_view):
    row_clustering = yaml_view['row clustering model']
    distributions = yaml_view['distribution models']
    ast_crp = core_parse_distribution(row_clustering[0])
    ast_distributions = [core_parse_distribution(d) for d in distributions]
    return [ast_crp, ast_distributions]

def parse_core_dsl_to_ast(core_dsl):
    core_dsl_yaml = yaml.load(core_dsl)
    return [core_parse_view(yaml_view['view']) for yaml_view in core_dsl_yaml]

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

generate_output = iter(xrange(10**5, 10**6))

def embedded_get_distname_output(ast_primitive_key):
    if '{' in ast_primitive_key:
        distname, index = ast_primitive_key.replace('}','').split('{')
        return (distname, int(index))
    return (ast_primitive_key, next(generate_output))

def embedded_compile_kwarg(stream, i, k, v):
    core_compile_indent(stream, i)
    stream.write('%s=%s,' % (k,v))

def embedded_compile_primitive(stream, i, ast_primitive):
    assert len(ast_primitive) == 1
    distname, output = embedded_get_distname_output(ast_primitive.keys()[0])
    kwargs = ast_primitive.values()[0]
    core_compile_indent(stream, i)
    constructor = primitive_constructors[distname]
    stream.write('%s(' % (constructor))
    embedded_compile_kwarg(stream, 0, 'outputs', [output])
    embedded_compile_kwarg(stream, 1, 'inputs', [])
    for k, v in kwargs.iteritems():
        if v is not None:
            embedded_compile_kwarg(stream, 1, k, v)
    stream.write(')')

def embedded_compile_product(stream, i, distributions):
    core_compile_indent(stream, i)
    stream.write('Product(cgpms=[\n')
    for k, ast_distribution in enumerate(distributions):
        embedded_compile_primitive(stream, i+4, ast_distribution)
        stream.write(',')
        if k < len(distributions) - 1:
            stream.write('\n')
    stream.write('])')

def embedded_compile_row_mixture(stream, i, v, ast_view):
    row_clustering = ast_view['row clustering model'][0]
    distributions = ast_view['distribution models']
    core_compile_indent(stream, i)
    stream.write('view%d = FlexibleRowMixture(\n' % (v,))
    core_compile_indent(stream, i+2)
    stream.write('cgpm_row_divide=')
    embedded_compile_primitive(stream, 0, row_clustering)
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
    for v, ast_view_full in enumerate(core_dsl_yaml):
        ast_view = ast_view_full['view']
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
