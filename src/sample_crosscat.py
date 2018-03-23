# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from math import isnan

import itertools
import yaml

from cStringIO import StringIO
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import tempfile

from cgpm.utils.general import get_prng
from cgpm.utils.general import merged
from cgpm.utils.general import mergedl

from cgpm2.transition_crosscat import get_distribution_outputs
from cgpm2.transition_rows import get_rowids
from cgpm2.transition_views import get_cgpm_current_view_index


generate_output_ast = iter(xrange(10**5, 10**6))

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
    output = next(generate_output_ast)
    distribution = ('crp', None)
    hypers = generate_random_hyperparameters(distribution, rng)
    return [output, distribution, hypers]

def generate_random_ast(schema, rng):
    """End-to-end simulator for AST of Core DSL."""
    distributions = [schemum[1] for schemum in schema]
    partition_alpha = rng.gamma(1,1)
    partition = generate_random_partition(partition_alpha, len(distributions), rng)
    row_dividers = [generate_random_row_divider(rng) for _i in partition]
    primitives = [
        [output, dist, generate_random_hyperparameters(dist, rng)]
        for output, dist in enumerate(distributions)
    ]
    return [
        [row_divider, [primitives[b] for b in block]]
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

def core_compile_distribution(stream, i, distribution):
    (output, (distname, distargs), hypers) = distribution
    core_compile_key_list(stream, i, '%s{id:%d}' % (distname, output))
    core_compile_distargs(stream, i+4, distargs)
    core_compile_hypers(stream, i+4, hypers)

def core_compile_distributions(stream, i, distributions):
    core_compile_key(stream, i, 'distribution models')
    for distribution in distributions:
        core_compile_distribution(stream, i+2, distribution)

def core_compile_row_clustering(stream, i, distribution):
    core_compile_key(stream, i, 'row clustering model')
    core_compile_distribution(stream, i+2, distribution)

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

def core_parse_distribution(ast_distribution):
    assert len(ast_distribution) == 1
    distname = ast_distribution.keys()[0]
    distargs = ast_distribution.values()[0]['distargs']
    hypers = ast_distribution.values()[0]['hypers']
    assert '{' in distname
    distname, output = distname.replace('}','').replace('id:','').split('{')
    return (int(output), (distname, distargs), hypers)

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

preamble = [
    'nan = float(\'nan\')',
]

def embedded_get_distname_output(ast_primitive_key):
    assert '{' in ast_primitive_key
    distname, output = \
        ast_primitive_key.replace('}','').replace('id:','').split('{')
    return (distname, int(output))

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
    for preamble_stmt in preamble:
        stream.write('%s\n' % (preamble_stmt,))
    stream.write('\n')
    for v, ast_view_full in enumerate(core_dsl_yaml):
        ast_view = ast_view_full['view']
        embedded_compile_row_mixture(stream, 0, v, ast_view)
        stream.write('\n')
    views = ', '.join('view%d' % (v,) for v in xrange(len(core_dsl_yaml)))
    stream.write('crosscat = Product(cgpms=[%s])' % (views,))
    return stream

# Embedded DSL -> CrossCat Binary.

def interpret_embedded_dsl(embedded_dsl):
    exec(embedded_dsl)
    return crosscat

def interpret_ast(ast):
    core_dsl = compile_ast_to_core_dsl(ast)
    embedded_dsl = compile_core_dsl_to_embedded_dsl(core_dsl.getvalue())
    return interpret_embedded_dsl(embedded_dsl.getvalue())

# CrossCat Binary -> Core DSL.

def convert_primitive_to_ast(primitive):
    ast_ot = primitive.outputs[0]
    ast_nm = (primitive.name(), primitive.get_distargs() or None)
    ast_hy = primitive.get_hypers()
    return (ast_ot, ast_nm, ast_hy)

def convert_product_to_ast(product):
    return [convert_primitive_to_ast(cgpm) for cgpm in product.cgpms]

def convert_view_to_ast(view):
    cgpm_crp = view.cgpm_row_divide
    cgpm_components_base = view.cgpm_components_array.cgpm_base
    ast_crp = convert_primitive_to_ast(cgpm_crp)
    ast_components_base = convert_product_to_ast(cgpm_components_base)
    return (ast_crp, ast_components_base)

def convert_crosscat_to_ast(crosscat):
    return [convert_view_to_ast(view) for view in crosscat.cgpms]

# CrossCat Binary -> Embedded DSL modeling.

def convert_crosscat_to_embedded_dsl_model(crosscat, stream=None):
    stream = stream or StringIO()
    crosscat_ast = convert_crosscat_to_ast(crosscat)
    crosscat_core_dsl = compile_ast_to_core_dsl(crosscat_ast)
    compile_core_dsl_to_embedded_dsl(crosscat_core_dsl.getvalue(), stream)
    return stream

# CrossCat Binary -> Embedded DSL observes.

def reindex_crp_observes(observes):
    output = next(observes[0].iterkeys())
    assert all(observe.keys() == [output] for observe in observes)
    assignments = [observe.items()[0] for observe in observes]
    tables = sorted(set([table for _output, table in assignments]))
    mapping = {t:i for i,t in enumerate(tables)}
    return [{output: mapping[table]} for output, table in assignments]

def get_sorted_rowids(rowids, observes):
    output = next(observes[0].iterkeys())
    assert all(observe.keys() == [output] for observe in observes)
    assert len(observes) == len(rowids)
    table_to_rowids = {}
    for rowid, observe in zip(rowids, observes):
        table = observe[output]
        if table not in table_to_rowids:
            table_to_rowids[table] = []
        table_to_rowids[table].append(rowid)
    sorted_tables = sorted(table_to_rowids)
    return list(itertools.chain.from_iterable([
        table_to_rowids[t] for t in sorted_tables
    ]))

def get_primitive_observes(primitive, rowid):
    output = primitive.outputs[0]
    observation = primitive.data.get(rowid, None)
    return {output: observation} if observation is not None else {}

def get_product_observes(product, rowid):
    observes = [get_primitive_observes(c, rowid) for c in product.cgpms]
    return mergedl(observes)

def get_components_observes(components_array, rowid):
    product_observes_all = [get_product_observes(product, rowid)
        for product in components_array.cgpms.itervalues()
    ]
    product_observes = filter(lambda x: x, product_observes_all)
    assert len(product_observes) == 1
    return product_observes[0]

def get_view_observes(view):
    rowids = get_rowids(view)
    # Handle observe for component assignment cgpm.
    cgpm_crp = view.cgpm_row_divide
    observe_crp = OrderedDict([
        (rowid, get_primitive_observes(cgpm_crp, rowid))
        for rowid in rowids
    ])
    observe_crp_reindex = reindex_crp_observes(observe_crp.values())
    sorted_rowids = get_sorted_rowids(rowids, observe_crp_reindex)
    rowid_to_index = {rowid:i for i, rowid in enumerate(sorted_rowids)}
    observe_crp_sorted = [
        observe_crp_reindex[rowid_to_index[rowid]]
        for rowid in sorted_rowids
    ]
    # Handle observe for component data cgpm.
    cgpm_components = view.cgpm_components_array
    observe_components_sorted = [
        get_components_observes(cgpm_components, rowid)
        for rowid in sorted_rowids
    ]
    # Return overall row-wise observation.
    return OrderedDict([
        (rowid, merged(i0, i1)) for rowid, i0, i1 in
        zip(sorted_rowids, observe_crp_sorted, observe_components_sorted)
    ])

def get_crosscat_observes(crosscat):
    return [get_view_observes(view) for view in crosscat.cgpms]

def convert_observes_to_embedded_dsl(observes, stream):
    for rowid, observation in observes.iteritems():
        stream.write('crosscat.observe(%d, %s)' % (rowid, observation))
        stream.write('\n')

def convert_crosscat_to_embedded_dsl_observe(crosscat, stream=None):
    stream = stream or StringIO()
    observes_views = get_crosscat_observes(crosscat)
    for v, observes in enumerate(observes_views):
        stream.write('# Incorporates for view %d.\n' % (v,))
        convert_observes_to_embedded_dsl(observes, stream)
        stream.write('\n')
    return stream

# CrossCat Binary -> Embedded DSL.

def render_trace_in_embedded_dsl(crosscat, stream=None):
    stream = stream or StringIO()
    convert_crosscat_to_embedded_dsl_model(crosscat, stream)
    stream.write('\n')
    stream.write('\n')
    convert_crosscat_to_embedded_dsl_observe(crosscat, stream)
    stream.write('\n')
    return stream

# CrossCat Binary -> VentureScript probabilistic program.

VSView = namedtuple('VSView', [
    'weights',
    'distributions',
    'observes_crp',
    'observes_data'
])

def get_variable_name(output):
    return 'var-%d' % (output,)

def convert_variable_name(varname, schema):
    output = int(varname.split('-')[1])
    return schema[output][0]

def get_primitive_distribution(cgpm):
    varname = get_variable_name(cgpm.outputs[0])
    if cgpm.name() == 'normal':
        # XXX Convert the hypers to (m, V, a, b) format.
        hypers = cgpm.get_hypers()
        maker = 'make_nig_normal(%1.4f, %1.4f, %1.4f, %1.4f)' % \
            (hypers['m'], hypers['r'], hypers['s'], hypers['nu'])
    elif cgpm.name() == 'poisson':
        hypers = cgpm.get_hypers()
        maker = 'make_gamma_poisson(%1.4f, %1.4f)' \
            % (hypers['a'], hypers['b'])
    elif cgpm.name() == 'categorical':
        hypers = cgpm.get_hypers()
        maker = 'make_symm_dirichlet_categorical(%1.4f)' % (hypers['alpha'])
    else:
        assert False, 'Unknown distribution: %s' % (cgpm.name(),)
    return (varname, maker)

def get_product_distributions(product):
    return [get_primitive_distribution(cgpm) for cgpm in product.cgpms]

def get_array_distributions(cgpm_components_array, tables):
    if len(cgpm_components_array.cgpms) == 0:
        return [get_product_distributions(cgpm_components_array.cgpm_base)]
    else:
        return [
            get_product_distributions(cgpm_components_array.cgpms[table])
            for table in tables
        ]

def tranpose_product_list(products):
    num_distributions = len(products[0])
    return [[prod[i] for prod in products] for i in xrange(num_distributions)]

def get_crp_tables_weights(crp_cgpm):
    tables = sorted(crp_cgpm.counts.keys())
    if not tables:
        return [0, [1.]]
    else:
        counts = [crp_cgpm.counts[table] for table in tables]
        sum_counts = float(sum(counts))
        return tables, [c/sum_counts for c in counts]

def get_view_observes_crp_vs(view):
    rowids = get_rowids(view)
    if not rowids:
        return []
    cgpm_crp = view.cgpm_row_divide
    observe_crp = OrderedDict([
        (rowid, get_primitive_observes(cgpm_crp, rowid))
        for rowid in rowids
    ])
    observe_crp_reindex = reindex_crp_observes(observe_crp.values())
    return [(rowid, obs.values()[0]) for
        rowid, obs in zip(rowids, observe_crp_reindex)]

def get_view_observes_data_vs(view):
    rowids = get_rowids(view)
    if not rowids:
        return []
    cgpm_components = view.cgpm_components_array
    observe_components = [
        get_components_observes(cgpm_components, rowid)
        for rowid in rowids
    ]
    def convert_observation(rowid, obs):
        return ((rowid, get_variable_name(k), v) for k,v in obs.iteritems())
    return list(itertools.chain.from_iterable(
        convert_observation(rowid, obs) for obs in observe_components
    ))

def get_view_representation(view):
    # Obtain the mixture weights from crp.
    crp_cgpm = view.cgpm_row_divide
    tables, weights = get_crp_tables_weights(crp_cgpm)
    # Obtain primitive distributions from components array.
    cgpm_components = view.cgpm_components_array
    product_distributions = get_array_distributions(cgpm_components, tables)
    # Transpose list of products into product of lists.
    primitive_distributions = tranpose_product_list(product_distributions)
    assert all(len(p) == len(weights) for p in primitive_distributions)
    # Obtain data incorporated in this view.
    observes_crp = get_view_observes_crp_vs(view)
    observes_data = get_view_observes_data_vs(view)
    # Build the shebang.
    return VSView(
        weights=weights,
        distributions=primitive_distributions,
        observes_crp=observes_crp,
        observes_data=observes_data
    )

def get_variables_to_view_assignment(crosscat):
    distribution_outputs = sorted(get_distribution_outputs(crosscat))
    view_idx = [
        get_cgpm_current_view_index(crosscat, [output])
        for output in distribution_outputs
    ]
    return [(get_variable_name(output), v_idx)
        for (output, v_idx) in zip(distribution_outputs, view_idx)]

def convert_trace_to_venturescript_ast(crosscat):
    vs_views = [get_view_representation(view) for view in crosscat.cgpms]
    variable_to_view = get_variables_to_view_assignment(crosscat)
    return vs_views, variable_to_view

def compile_distributions_view(stream, distributions_list, view_idx, terminal,
        schema):
    for i, distributions in enumerate(distributions_list):
        varnames, cgpms = zip(*distributions)
        varname = varnames[0]
        assert all(v==varname for v in varnames)
        varname_proper = convert_variable_name(varname, schema)
        core_compile_indent(stream, 4)
        stream.write('["%s",\t[%d, [\n' % (varname_proper, view_idx))
        for j, cgpm in enumerate(cgpms):
            core_compile_indent(stream, 8)
            stream.write('%s' % (cgpm,))
            if j < len(cgpms) - 1:
                stream.write(',')
            stream.write('\n')
        core_compile_indent(stream, 8)
        stream.write(']]]')
        if not terminal:
            stream.write(',')
        elif i < len(distributions_list) - 1:
            stream.write(',')
        stream.write('\n')

def compile_distributions_list(stream, distributions_view, schema):
    stream.write('assume variable_to_distributions = dict(\n')
    for view_idx, distributions_list in enumerate(distributions_view):
        core_compile_indent(stream, 4)
        stream.write('// Mixture models in view %d.\n' % (view_idx,))
        terminal = (view_idx == len(distributions_view) - 1)
        compile_distributions_view(stream, distributions_list,
            view_idx, terminal, schema)
    stream.write(');')

def compile_weights_list(stream, weights_list):
    stream.write('assume view_to_mixture_weights = dict(\n')
    for i, weights in enumerate(weights_list):
        core_compile_indent(stream, 4)
        weights_str = ','.join(['%1.4f' % (w,) for w in weights])
        stream.write('[%d,\t simplex(%s)]' % (i, weights_str))
        if i < len(weights_list) - 1:
            stream.write(',')
        stream.write('\n')
    stream.write(');')

def compile_variable_to_view_assignment(stream, variable_to_view, schema):
    stream.write('assume variable_to_view_assignment = dict(\n')
    # Sort by view_idx.
    variable_to_view_sorted = sorted(variable_to_view, key=lambda t: t[1])
    for i, (varname, view_idx) in enumerate(variable_to_view_sorted):
        core_compile_indent(stream, 4)
        varname_proper = convert_variable_name(varname, schema)
        stream.write('["%s", %d]' % (varname_proper, view_idx))
        if i < len(variable_to_view) - 1:
            stream.write(',')
        stream.write('\n')
    stream.write(');')

def compile_get_row_mixture_assignment(stream):
    stream.write(
'''assume get_row_mixture_assignment = mem((rowid, view) ~> {
    weights = view_to_mixture_weights[view];
    categorical(weights) #mixture_assignment:pair(rowid, view)
});''')

def compile_get_cell_value(stream):
    stream.write(
'''assume get_cell_value = (rowid, variable) ~> {
    variable_distributions = variable_to_distributions[variable];
    view = variable_distributions[0];
    distributions = variable_distributions[1];
    row_mixture_assignment = get_row_mixture_assignment(rowid, view);
    sampler = distributions[row_mixture_assignment];
    sampler() #cell_value:pair(rowid, variable)
};''')

def compile_observes_crp_list(stream, observes_crp_list, explicit):
    stream.write('// Observe mixture assignment of each rowid in each view.\n')
    with tempfile.NamedTemporaryFile(prefix='cgpm2vs-crp', delete=False) as f:
        for view_idx, observes_crp in enumerate(observes_crp_list):
            for rowid, assignment in observes_crp:
                if explicit:
                    stream.write(
                        'observe get_row_mixture_assignment(rowid: %d, '
                            'view:%d) = %d;'
                        % (rowid, view_idx, assignment))
                    stream.write('\n')
                f.write('%d, %d, %d\n' % (rowid, view_idx, assignment))

def compile_observes_data_list(stream, observes_data_list, schema, explicit):
    stream.write('// Observe each datum from the training data table.\n')
    with tempfile.NamedTemporaryFile(prefix='cgpm2vs-datum', delete=False) as f:
        for observes_data in observes_data_list:
            for rowid, varname, datum in observes_data:
                varname_proper = convert_variable_name(varname, schema)
                if explicit and not isnan(datum):
                    stream.write(
                        'observe get_cell_value(rowid:%d, "%s") = %1.4f;'
                        % (rowid, varname_proper, datum))
                    stream.write('\n')
                f.write('%d,"%s",%1.4f\n' % (rowid, varname_proper, datum))

def render_trace_in_venturescript(crosscat, schema, observes=None, stream=None):
    stream = stream or StringIO()
    vs_views, variable_to_view = convert_trace_to_venturescript_ast(crosscat)
    # Compile the distributions.
    distributions_list = [v.distributions for v in vs_views]
    compile_distributions_list(stream, distributions_list, schema)
    stream.write('\n\n')
    # Compile the variable to view assignments.
    #
    # XXX These identities have been moved into the distributions section.
    compile_variable_to_view_assignment(stream, variable_to_view, schema)
    stream.write('\n\n')
    #
    # Compile the mixture weights.
    weights_list = [v.weights for v in vs_views]
    compile_weights_list(stream, weights_list)
    stream.write('\n\n')
    # Compile sampler for (rowid, view) to mixture assignments.
    compile_get_row_mixture_assignment(stream)
    stream.write('\n\n')
    # Compile sampler for (rowid, variable) to datum.
    compile_get_cell_value(stream)
    stream.write('\n\n')
    if observes:
        # Compile observes for CRP.
        observes_crp_list = [v.observes_crp for v in vs_views]
        compile_observes_crp_list(stream, observes_crp_list, observes)
        stream.write('\n')
        # Compile observes for data.
        observes_data_list = [v.observes_data for v in vs_views]
        compile_observes_data_list(stream, observes_data_list, schema, observes)
    return stream

# Testing.


if __name__ == '__main__':
    prng = get_prng(10)

    schema = [
        ['perigee_km',      ('normal', None)],
        ['apogee_km',       ('normal', None)],
        ['failures',        ('poisson', None)],
        ['class_of_orbit',  ('categorical', {'k':3})],
        ['type_of_orbit',   ('categorical', {'k':10})],
    ]

    ast = generate_random_ast(schema, prng)
    print ast

    core_dsl = compile_ast_to_core_dsl(ast)
    print core_dsl.getvalue()

    print parse_core_dsl_to_ast(core_dsl.getvalue())

    embedded_dsl = compile_core_dsl_to_embedded_dsl(core_dsl.getvalue())
    print embedded_dsl.getvalue()

    exec(embedded_dsl.getvalue())

    crosscat.observe(1, {0:1, 1:-1, 2: 3})
    crosscat.observe(2, {0:2, 1:3, 4:1, 5:8})

    # Go from the model -> ast -> code (will be done post inference).
    # print convert_crosscat_to_embedded_dsl_model(crosscat).getvalue()
    # observes = convert_crosscat_to_embedded_dsl_observe(crosscat)
    # print observes.getvalue()

    print render_trace_in_embedded_dsl(crosscat).getvalue()

    print render_trace_in_venturescript(crosscat, schema).getvalue()
