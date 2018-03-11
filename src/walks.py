# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import itertools

from cgpm.utils.general import build_cgpm

from .chain import Chain
from .distribution import DistributionCGPM
from .finite_array import FiniteArray
from .finite_rowmix import FiniteRowMixture
from .flexible_array import FlexibleArray
from .flexible_rowmix import FlexibleRowMixture
from .product import Product


def clone_cgpm(cgpm_base, rng):
    """Create clone of given CGPM."""
    metadata = cgpm_base.to_metadata()
    return build_cgpm(metadata, rng)

def get_cgpms_by_output_index(cgpm, output):
    """Retrieve all cgpms internal to cgpm that generate the given output."""
    if isinstance(cgpm, DistributionCGPM):
        return [cgpm] if cgpm.outputs == [output] else []
    elif isinstance(cgpm, (Chain, Product)):
        cgpm_list = [c for c in cgpm.cgpms if output in c.outputs]
        assert len(cgpm_list) == 1
        return get_cgpms_by_output_index(cgpm_list[0], output)
    elif isinstance(cgpm, FiniteArray):
        cgpm_list = [get_cgpms_by_output_index(c, output) for c in cgpm.cgpms]
        return list(itertools.chain.from_iterable(cgpm_list))
    elif isinstance(cgpm, FlexibleArray):
        cgpm_list = [get_cgpms_by_output_index(c, output) for c in
            cgpm.cgpms.values() + [cgpm.cgpm_base]]
        return list(itertools.chain.from_iterable(cgpm_list))
    elif isinstance(cgpm, (FiniteRowMixture, FlexibleRowMixture)):
        if output in cgpm.cgpm_row_divide.outputs:
            return get_cgpms_by_output_index(cgpm.cgpm_row_divide, output)
        else:
            return get_cgpms_by_output_index(cgpm.cgpm_components_array, output)
    else:
        assert False, 'Unknown CGPM'

def remove_cgpm(cgpm, output):
    """Remove CGPM responsible for output from the given composite cgpm."""
    if isinstance(cgpm, DistributionCGPM):
        assert False, 'DistributionCGPM is not composite'
    elif isinstance(cgpm, Product):
        cgpms_new = [c for c in cgpm.cgpms if output not in c.outputs]
        assert len(cgpms_new) in [len(cgpm.cgpms)-1, len(cgpm.cgpms)]
        result = Product(cgpms_new, rng=cgpm.rng)
        return result
    elif isinstance(cgpm, FlexibleArray):
        cgpm_base_new = remove_cgpm(cgpm.cgpm_base, output)
        cgpms_new = {i: remove_cgpm(c, output) for i, c in cgpm.cgpms.items()}
        result = FlexibleArray(cgpm_base_new, cgpm.indexer, rng=cgpm.rng)
        result.cgpms = cgpms_new
        result.rowid_to_index = cgpm.rowid_to_index
        return result
    elif isinstance(cgpm, FlexibleRowMixture):
        assert output != cgpm.indexer
        cgpm_row_divide_new = cgpm.cgpm_row_divide
        cgpm_components_array_new = \
            remove_cgpm(cgpm.cgpm_components_array, output)
        cgpm_base_new = cgpm_components_array_new.cgpm_base
        result = FlexibleRowMixture(
            cgpm_row_divide_new, cgpm_base_new, rng=cgpm.rng)
        result.rowid_to_component = cgpm.rowid_to_component
        result.cgpm_components_array = cgpm_components_array_new
        return result
    elif isinstance(cgpm, FiniteArray):
        cgpms_new = [remove_cgpm(c, output) for c in cgpm.cgpms]
        result = FiniteArray(cgpms_new, cgpm.indexer, rng=cgpm.rng)
        result.rowid_to_index = cgpm.rowid_to_index
        return result
    elif isinstance(cgpm, FiniteRowMixture):
        assert output != cgpm.indexer
        cgpm_row_divide_new = cgpm.cgpm_row_divide
        cgpm_components_array_new = \
            remove_cgpm(cgpm.cgpm_components_array, output)
        cgpm_components_new = cgpm_components_array_new.cgpms
        result = FiniteRowMixture(
            cgpm_row_divide_new, cgpm_components_new, rng=cgpm.rng)
        result.cgpm_components_array = cgpm_components_array_new
        return result
    else:
        assert False, 'Not implemented'

def add_cgpm(cgpm, cgpm_new):
    """Add cgpm_new to the given cgpm."""
    if isinstance(cgpm, Product):
        cgpms_new = cgpm.cgpms + [cgpm_new]
        result = Product(cgpms_new, rng=cgpm.rng)
        return result
    elif isinstance(cgpm, FlexibleArray):
        cgpm_base_new = add_cgpm(cgpm.cgpm_base, cgpm_new)
        cgpms_new = {i: add_cgpm(c, clone_cgpm(cgpm_new, cgpm.rng))
            for i, c in cgpm.cgpms.iteritems()}
        result = FlexibleArray(cgpm_base_new, cgpm.indexer, rng=cgpm.rng)
        result.cgpms = cgpms_new
        result.rowid_to_index = cgpm.rowid_to_index
        return result
    elif isinstance(cgpm, FlexibleRowMixture):
        cgpm_row_divide_new = cgpm.cgpm_row_divide
        cgpm_components_array_new = \
            add_cgpm(cgpm.cgpm_components_array, cgpm_new)
        cgpm_base_new = cgpm_components_array_new.cgpm_base
        result = FlexibleRowMixture(
            cgpm_row_divide_new, cgpm_base_new, rng=cgpm.rng)
        result.rowid_to_component = cgpm.rowid_to_component
        result.cgpm_components_array = cgpm_components_array_new
        return result
    elif isinstance(cgpm, FiniteArray):
        cgpms_new = [add_cgpm(c, clone_cgpm(cgpm_new, cgpm.rng))
            for c in cgpm.cgpms]
        result = FiniteArray(cgpms_new, cgpm.indexer, rng=cgpm.rng)
        result.rowid_to_index = cgpm.rowid_to_index
        return result
    elif isinstance(cgpm, FiniteRowMixture):
        cgpm_row_divide_new = cgpm.cgpm_row_divide
        cgpm_components_array_new = \
            add_cgpm(cgpm.cgpm_components_array, cgpm_new)
        cgpm_components_new = cgpm_components_array_new.cgpms
        result = FiniteRowMixture(
            cgpm_row_divide_new, cgpm_components_new, rng=cgpm.rng)
        result.cgpm_components_array = cgpm_components_array_new
        return result
    else:
        assert False, 'Not implemented'
