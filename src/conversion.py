# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cgpm.utils.general import merged

from .categorical import Categorical
from .chain import Chain
from .crp import CRP
from .distribution import DistributionCGPM
from .finite_array import FiniteArray
from .finite_rowmix import FiniteRowMixture
from .flexible_array import FlexibleArray
from .flexible_rowmix import FlexibleRowMixture
from .normal import Normal
from .poisson import Poisson
from .product import Product

# CGPM to CGPM2 conversion.

cctype_to_primitive = {
    'categorical' : Categorical,
    'crp'         : CRP,
    'normal'      : Normal,
    'poisson'     : Poisson,
}

def convert_dim_to_base_cgpm(dim):
    return cctype_to_primitive[dim.cctype](dim.outputs, [], hypers=dim.hypers)

def rebase_cgpm_row_assignments(Zr):
    unique_tables = set(Zr.itervalues())
    tables_map = {t:i for i, t in enumerate(unique_tables)}
    tables_new = [(rowid, tables_map[t]) for rowid, t in Zr.iteritems()]
    return sorted(tables_new, key=lambda a: a[1])

def convert_view_to_rowmixture(view):
    component_base_cgpms = Product([
        convert_dim_to_base_cgpm(d) for d in view.dims.itervalues()
    ])
    cgpm_row_divide = convert_dim_to_base_cgpm(view.crp)
    cgpm_row_mixture = FlexibleRowMixture(cgpm_row_divide, component_base_cgpms)
    for rowid, assignment in rebase_cgpm_row_assignments(view.Zr()):
        obs_z = {cgpm_row_divide.outputs[0]: assignment}
        obs_x = {c: view.X[c][rowid] for c in component_base_cgpms.outputs}
        observation = merged(obs_z, obs_x)
        cgpm_row_mixture.observe(rowid, observation)
    return cgpm_row_mixture

def convert_cgpm_state_to_cgpm2(state):
    views = [convert_view_to_rowmixture(view) for view in state.views.values()]
    return Product(views)

# AST Conversion

def convert_cgpm_to_ast(cgpm):
    # XXX I am a hack for a particular DSL.
    if isinstance(cgpm, DistributionCGPM):
        return (
        # tuple(cgpm.outputs),
        (cgpm.name(), cgpm.get_distargs()),
        cgpm.get_hypers(),
    )
    elif isinstance(cgpm, (Chain, Product)):
        return [convert_cgpm_to_ast(cgpm) for cgpm in cgpm.cgpms]
    elif isinstance(cgpm, FlexibleRowMixture):
        ast_row_divide = convert_cgpm_to_ast(cgpm.cgpm_row_divide)
        ast_compments_base = \
            convert_cgpm_to_ast(cgpm.cgpm_components_array.cgpm_base)
        return (ast_row_divide, ast_compments_base)
    elif isinstance(cgpm, FiniteRowMixture):
        raise NotImplementedError('No relevant AST')
    elif isinstance(cgpm, FlexibleArray):
        raise NotImplementedError('No relevant AST')
    elif isinstance(cgpm, FiniteArray):
        raise NotImplementedError('No relevant AST')
    else:
        assert False, 'Unknown CGPM'

def convert_cgpm_state_to_ast(state):
    cgpm2 = convert_cgpm_state_to_cgpm2(state)
    return convert_cgpm_to_ast(cgpm2)
