import torch

from itertools import zip_longest
from cormorant.cg_lib import SO3Tau, SO3TensorBase

from cormorant.cg_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar

def mul(val1, val2, same_maxl=True):
    # Both va1 and val2 are instances of SO3Tensor
    if isinstance(val1, SO3TensorBase) and isinstance(val2, SO3TensorBase):
        val1._mul_type_check(type(val1), type(val2))
        val2._mul_type_check(type(val1), type(val2))
        if same_maxl and val1.maxl != val2.maxl:
            raise ValueError('Two SO3Tensors have different maxl values ({} {}), '
                             'but same_maxl=True! Try directly calling '
                             'SO3Tensor.mul() with same_maxl=False'.format(val1.maxl, val2.maxl))
        return val2.__class__([part1 * part2 for part1, part2 in zip_longest([val1, val2], fillvalue=1)])
    # Multiply val1 with a list/tuple
    elif isinstance(val1, SO3TensorBase) and type(val2) in [list, tuple]:
        return val1.__class__([part1 * part2 for part1, part2 in zip(val1, val2)])
    # Multiply val1 with something else
    elif isinstance(val1, SO3TensorBase) and not isinstance(val2, SO3TensorBase):
        return val1.__class__([part1 * val2 for part1 in val1])
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, SO3TensorBase) and type(val1) in [list, tuple]:
        return val2.__class__([part1 * part2 for part1, part2 in zip(val1, val2)])
    # Multiply val2 with something else
    elif not isinstance(val1, SO3TensorBase) and isinstance(val2, SO3TensorBase):
        return val2.__class__([val1 * part2 for part2 in val2])
    else:
        raise ValueError('Neither class inherits from SO3Tensor!')


def add(val1, val2, same_maxl=True):
    # Both va1 and val2 are instances of SO3Tensor, and same_maxl=True
    if isinstance(val1, SO3TensorBase) and isinstance(val2, SO3TensorBase):
        if val1.maxl != val2.maxl and same_maxl:
            raise ValueError('Two SO3Tensors have different maxl values ({} {}), '
                             'but same_maxl=True! Try directly calling '
                             'SO3Tensor.mul() with same_maxl=False'.format(val1.maxl, val2.maxl))
        return val2.__class__([part1 + part2 for part1, part2 in zip_longest([val1, val2], fillvalue=0)])
    # Add val1 with a list/tuple
    elif isinstance(val1, SO3TensorBase) and type(val2) in [list, tuple]:
        return val1.__class__([part1 + part2 for part1, part2 in zip(val1, val2)])
    # Add val1 with something else
    elif isinstance(val1, SO3TensorBase) and not isinstance(val2, SO3TensorBase):
        return val1.__class__([part1 + val2 for part1 in val1])
    # Add val2 with a list/tuple
    elif not isinstance(val1, SO3TensorBase) and type(val1) in [list, tuple]:
        return val2.__class__([part1 + part2 for part1, part2 in zip(val1, val2)])
    # Add val2 with something else
    elif not isinstance(val1, SO3TensorBase) and isinstance(val2, SO3TensorBase):
        return val2.__class__([val1 + part2 for part2 in val2])
    else:
        raise ValueError('Neither class inherits from SO3Tensor!')


def cat(reps_list):
    """
    Concatenate (direct sum) a :obj:`list` of :obj:`SO3Tensor` representations.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`SO3Tensor` or compatible

    Return
    ------
    rep_cat : :obj:`SO3Tensor`
        Direct sum of all :obj:`SO3Tensor` in `reps_list`
    """
    reps_cat = [list(filter(lambda x: x is not None, reps)) for reps in zip_longest(*reps_list, fillvalue=None)]
    reps_cat = [torch.cat(reps, dim=reps_list[0].cdim) for reps in reps_cat]

    return reps_list[0].__class__(reps_cat)

def mix(rep, weights):
    """
    Linearly mix representation.

    Parameters
    ----------
    rep : :obj:`SO3Vec` or compatible
    weights : :obj:`SO3Weights` or compatible

    Return
    ------
    :obj:`SO3Vec`
        Mixed direct sum of all :obj:`SO3Vec` in `reps_list`
    """
    if len(rep) != len(weights):
        raise ValueError('Must have one mixing weight for each part of SO3Vec!')

    rep_mix = SO3Vec([mul_zscalar_zirrep(weight, part) for weight, part in zip(weights, rep)])

    return rep_mix


def cat_mix(reps_list, weights):
    """
    First concatenate (direct sum) and then linearly mix a :obj:`list` of
    :obj:`SO3Vec` objects with :obj:`SO3Weights` weights.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`SO3Vec` or compatible
    weights : :obj:`SO3Weights` or compatible

    Return
    ------
    :obj:`SO3Vec`
        Mixed direct sum of all :obj:`SO3Vec` in `reps_list`
    """

    return weights @ cat(reps_list)
