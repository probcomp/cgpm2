# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools
import sys

from collections import OrderedDict

import numpy as np

from crosscat.LocalEngine import LocalEngine

from .progress import report_progress

from .transition_rows import set_rowid_component

from .transition_views import get_cgpm_base
from .transition_views import get_cgpm_current_view_index
from .transition_views import get_dataset
from .transition_views import set_cgpm_view_assignment


def partition_assignments_to_blocks(assignments):
    unique = set(assignments)
    return [[i for i, a in enumerate(assignments) if a == u] for u in unique]

def get_distribution_outputs(crosscat):
    return list(itertools.chain.from_iterable(
        [view.outputs[1:] for view in crosscat.cgpms]
    ))

def get_crosscat_dataset_one(crosscat, output):
    dataset_raw = get_dataset(crosscat, output)
    dataset_sorted = sorted(dataset_raw, key=lambda v:v[0])
    values = [value[output] for _rowid, value in dataset_sorted]
    return values

def get_crosscat_dataset(crosscat):
    outputs = get_distribution_outputs(crosscat)
    datasets = [
        {rowid: obs[output] for rowid, obs in get_dataset(crosscat, output)}
        for output in outputs
    ]
    return OrderedDict(zip(outputs, datasets))

def get_crosscat_cgpm_name_distargs_hypers_one(crosscat, output):
    cgpm_base = get_cgpm_base(crosscat, output)
    return cgpm_base.name(), cgpm_base.get_distargs(), cgpm_base.get_hypers()

def get_crosscat_cgpm_name_distargs_hypers(crosscat, outputs):
    return zip(*[
        get_crosscat_cgpm_name_distargs_hypers_one(crosscat, output)
        for output in outputs
    ])

# CrossCat M_c.

def create_metadata(data, cctype, distargs):
    if cctype == 'normal':
        return {
            unicode('modeltype') : unicode('normal_inverse_gamma'),
            unicode('value_to_code') : {},
            unicode('code_to_value') : {},
        }
    elif cctype == 'categorical':
        categories = [v for v in sorted(set(data)) if not np.isnan(v)]
        assert all(0 <= c < distargs['k'] for c in categories)
        codes = [unicode('%d') % (c,) for c in categories]
        ncodes = range(len(codes))
        return {
            unicode('modeltype') : unicode('symmetric_dirichlet_discrete'),
            unicode('value_to_code') : dict(zip(map(unicode, ncodes), codes)),
            unicode('code_to_value') : dict(zip(codes, ncodes)),
        }
    else:
        assert False

def _get_crosscat_M_c(crosscat, observations):
    outputs = get_distribution_outputs(crosscat)
    cctypes, distargs, _hyperparams = \
        get_crosscat_cgpm_name_distargs_hypers(crosscat, outputs)

    assert len(observations) == len(outputs) == len(cctypes) == len(distargs)
    assert all(c in ['normal', 'categorical'] for c in cctypes)
    ncols = len(outputs)

    column_names = [unicode('c%d') % (i,) for i in outputs]
    # Convert all numerical datatypes to normal for lovecat.
    column_metadata = [
        create_metadata(observations[output].values(), cctype, distarg)
        for output, cctype, distarg in zip(outputs, cctypes, distargs)
    ]

    return {
        unicode('name_to_idx'):
            dict(zip(column_names, range(ncols))),
        unicode('idx_to_name'):
            dict(zip(map(unicode, range(ncols)), column_names)),
        unicode('column_metadata'):
            column_metadata,
    }

# CrossCat T.

def observations_to_rowids_mapping(observations):
    rowids_list = [set(d.iterkeys()) for d in observations.itervalues()]
    assert all(rowids_list[0]==r for r in rowids_list[1:]), 'Misaligned rowids'
    rowids_sorted = sorted(rowids_list[0])
    return OrderedDict(enumerate(rowids_sorted))

def crosscat_value_to_code(M_c, val, col):
    if np.isnan(val):
        return val
    lookup = M_c['column_metadata'][col]['code_to_value']
    if lookup:
        # For hysterical raisins, code_to_value and value_to_code are
        # backwards, so to convert from a raw value to a crosscat value we
        # need to do code->value.
        assert unicode(int(val)) in lookup
        return float(lookup[unicode(int(val))])
    else:
        return float(val)

def _get_crosscat_T(crosscat, M_c, observations):
    """Create dataset T from crosscat."""
    outputs = get_distribution_outputs(crosscat)
    rowids_mapping = observations_to_rowids_mapping(observations)
    rowids = rowids_mapping.itervalues()
    return [
        [crosscat_value_to_code(M_c, observations[col][row], i)
            for (i, col) in enumerate(outputs)]
        for row in rowids
    ]

# CrossCat X_D.

def _get_crosscat_X_D(crosscat):
    """Create X_D from crosscat."""
    views = crosscat.cgpms
    row_partition_assignments = [
        get_crosscat_dataset_one(crosscat, view.outputs[0])
        for view in views
    ]
    row_partition_blocks = [
        sorted(set(row_partition_assignment))
        for row_partition_assignment in row_partition_assignments
    ]
    row_partition_block_to_code = [
        {b: i for (i, b) in enumerate(blocks)}
        for blocks in row_partition_blocks
    ]
    row_partition_assignments_remapped = [
        [block_to_code[b] for b in row_partition_assignment]
        for (block_to_code, row_partition_assignment)
            in zip(row_partition_block_to_code, row_partition_assignments)
    ]
    # row_partition_assignments_remapped[i] contains row partition assignments
    # within view i, indexed so that clusters are numbered starting at zero.
    return row_partition_assignments_remapped

# CrossCat X_L.

def create_hypers(M_c, index, cctype, hypers):
    if cctype == 'normal':
        return {
            unicode('fixed'): 0.0,
            unicode('mu'): hypers['m'],
            unicode('nu'): hypers['nu'],
            unicode('r'): hypers['r'],
            unicode('s'): hypers['s'],
        }
    elif cctype == 'categorical':
        K = len(M_c['column_metadata'][index]['code_to_value'])
        assert K > 0
        return {
            unicode('fixed'): 0.0,
            unicode('dirichlet_alpha'): hypers['alpha'],
            unicode('K'): K
        }
    else:
        assert False

def create_view_state(view, row_partition):
    # Generate X_L['view_state'][v]['column_component_suffstats']
    num_blocks = len(set(row_partition))
    column_component_suffstats = [
        [{} for _b in xrange(num_blocks)]
        for _d in view.outputs[1:]
    ]

    # Generate X_L['view_state'][v]['column_names']
    column_names = \
        [unicode('c%d' % (o,)) for o in view.outputs[1:]]

    # Generate X_L['view_state'][v]['row_partition_model']
    view_alpha = view.cgpm_row_divide.get_hypers()['alpha']
    counts = list(np.bincount(row_partition))
    assert 0 not in counts
    return {
        unicode('column_component_suffstats'):
            column_component_suffstats,
        unicode('column_names'):
            column_names,
        unicode('row_partition_model'): {
            unicode('counts'): counts,
            unicode('hypers'): {unicode('alpha'): view_alpha}
        }
    }

def _get_crosscat_X_L(crosscat, M_c, X_D):
    """Create X_L from crosscat."""
    outputs = get_distribution_outputs(crosscat)
    cctypes, _distargs, hyperparams = \
        get_crosscat_cgpm_name_distargs_hypers(crosscat, outputs)

    # -- Generates X_L['column_hypers'] --
    column_hypers = [
        create_hypers(M_c, i, cctype, hypers)
        for i, (cctype, hypers) in enumerate(zip(cctypes, hyperparams))
    ]

    # -- Generates X_L['column_partition'] --
    # views_remapped[i] contains the zero-based view index for outputs[i].
    view_assignments = [
        get_cgpm_current_view_index(crosscat, [output])
        for output in outputs]
    counts = list(np.bincount(view_assignments))
    assert 0 not in counts
    column_partition = {
        unicode('assignments'): view_assignments,
        unicode('counts'): counts,
        unicode('hypers'): {unicode('alpha'): 1.}
    }

    # -- Generates X_L['view_state'] --
    # view_states[i] is the view for code views_to_code[i], so we need to
    # iterate in the same order of views_unique to agree with both X_D (the row
    # partition in each view), as well as X_L['column_partition']['assignments']
    view_states = [
        create_view_state(view, row_partition)
        for view, row_partition in zip(crosscat.cgpms, X_D)
    ]

    # Generates X_L['col_ensure'].
    col_ensure = dict()

    return {
        unicode('column_hypers'): column_hypers,
        unicode('column_partition'): column_partition,
        unicode('view_state'): view_states,
        unicode('col_ensure'): col_ensure
    }

# Convert (X_L, X_D) -> CGPM.

def _get_crosscat_updated(crosscat, observations, M_c, X_L, X_D):
    # Fetch crosscat data structures.
    outputs = get_distribution_outputs(crosscat)
    cctypes, _distargs, _hyperparams = \
        get_crosscat_cgpm_name_distargs_hypers(crosscat, outputs)

    # Checking on M_c.
    assert len(M_c['name_to_idx']) == len(outputs)
    for i, cctype in enumerate(cctypes):
        modeltype = M_c['column_metadata'][i]['modeltype']
        if cctype == 'normal':
            assert modeltype == 'normal_inverse_gamma'
        elif cctype == 'categorical':
            assert modeltype == 'symmetric_dirichlet_discrete'
        else:
            assert False, 'Unknown component distribution'

    # Check X_D.
    assert all(len(X_D[0]) == len(partition) for partition in X_D)
    assert len(X_D) == len(X_L['view_state'])

    # Checking X_L.
    assert len(X_L['column_partition']['assignments']) == len(outputs)

    # Get mapping from zero-based indexes.
    outputs_mapping = {i: output for i, output in enumerate(outputs)}
    rowids_mapping = observations_to_rowids_mapping(observations)

    # Set the view partition
    view_assignments = X_L['column_partition']['assignments']
    view_partition_blocks = partition_assignments_to_blocks(view_assignments)
    assert len(view_partition_blocks) == len(X_D)
    for block, row_assignments in zip(view_partition_blocks, X_D):
        # Move all the columns to the same view.
        members = [outputs_mapping[a] for a in block]
        crosscat = set_cgpm_view_assignment(crosscat, members[0], None)
        for m in members[1:]:
            crosscat = set_cgpm_view_assignment(crosscat, m, members[0])
        view_index = get_cgpm_current_view_index(crosscat, members)
        view = crosscat.cgpms[view_index]
        # Set the row partition.
        row_blocks = partition_assignments_to_blocks(row_assignments)
        for row_block in row_blocks:
            row_members = [rowids_mapping[r] for r in row_block]
            set_rowid_component(view, row_members[0], None)
            for rm in row_members[1:]:
                set_rowid_component(view, rm, row_members[0])
    return crosscat

def _progress(n_steps, max_time, step_idx, elapsed_secs, end=None):
    if end:
        print '\rCompleted: %d iterations in %f seconds.' %\
            (step_idx, elapsed_secs)
    else:
        p_seconds = elapsed_secs / max_time if max_time != -1 else 0
        p_iters = float(step_idx) / n_steps
        percentage = max(p_iters, p_seconds)
        report_progress(percentage, sys.stdout)

def transition_cpp(crosscat, N=None, S=None, kernels=None, rowids=None,
        cols=None, seed=None, progress=None):
    """Runs full Gibbs sweeps of all kernels on the cgpm.state.State object.

    Permissible kernels:
       'column_partition_hyperparameter'
       'column_partition_assignments'
       'column_hyperparameters'
       'row_partition_hyperparameters'
       'row_partition_assignments'
    """
    if seed is None:
        seed = 1
    if kernels is None:
        kernels = ()
    if (progress is None) or progress:
        progress = _progress

    if N is None and S is None:
        n_steps = 1
        max_time = -1
    if N is not None and S is None:
        n_steps = N
        max_time = -1
    elif S is not None and N is None:
        # This is a hack, lovecat has no way to specify just max_seconds.
        n_steps = 150000
        max_time = S
    elif S is not None and N is not None:
        n_steps = N
        max_time = S
    else:
        assert False

    if cols is None:
        cols = ()
    else:
        outputs = get_distribution_outputs(crosscat)
        outputs_mapping_inverse = {c:i for i,c in enumerate(outputs)}
        cols = [outputs_mapping_inverse[c] for c in cols]

    observations = get_crosscat_dataset(crosscat)
    if not observations:
        return crosscat

    if rowids is None:
        rowids = ()
    else:
        rowids_mapping = observations_to_rowids_mapping(observations)
        rowids_mapping_inverse = {r:i for i, r in rowids_mapping.iteritems()}
        rowids = [rowids_mapping_inverse[r] for r in rowids]

    M_c = _get_crosscat_M_c(crosscat, observations)
    T = _get_crosscat_T(crosscat, M_c, observations)
    X_D = _get_crosscat_X_D(crosscat)
    X_L = _get_crosscat_X_L(crosscat, M_c, X_D)
    LE = LocalEngine(seed=seed)
    X_L_new, X_D_new = LE.analyze(
        M_c=M_c,
        T=T,
        X_L=X_L,
        X_D=X_D,
        seed=seed,
        kernel_list=kernels,
        n_steps=n_steps,
        max_time=max_time,
        c=cols,
        r=rowids,
        progress=progress,
    )
    # XXX This reconstruction is wasteful: can find the diff in the trace
    # and apply those, but it is some work to get that right.
    return _get_crosscat_updated(crosscat, observations, M_c, X_L_new, X_D_new)
