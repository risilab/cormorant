import torch

from itertools import zip_longest

from cormorant.so3_lib import so3_tau, so3_tensor, so3_vec, so3_scalar

SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor
SO3Vec = so3_vec.SO3Vec

from cormorant.so3_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar

def _check_maxl(val1, val2):
    if len(val1) != len(val2):
        raise ValueError('Two SO3Tensor subclasses have different maxl values '
                         '({} {})!'.format(len(val1), len(val2)))

def _dispatch_op(op, val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    SO3Tensor.
    """

    # Both va1 and val2 are instances of SO3Tensor
    if isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        _check_maxl(val1, val2)
        val1._mul_type_check(type(val1), type(val2))
        val2._mul_type_check(type(val1), type(val2))
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val2)
    # Multiply val1 with a list/tuple
    elif isinstance(val1, SO3Tensor) and type(val2) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val1 with something else
    elif isinstance(val1, SO3Tensor) and not isinstance(val2, SO3Tensor):
        applied_op = [op(val2, part1) for part1 in val1]
        output_class =  type(val1)
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, SO3Tensor) and type(val1) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val2 with something else
    elif not isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        applied_op = [op(val1, part2) for part2 in val2]
        output_class = type(val2)
    else:
        raise ValueError('Neither class inherits from SO3Tensor!')

    return output_class(applied_op)


def mul(val1, val2):
    return _dispatch_op(torch.mul, val1, val2)

def add(val1, val2):
    return _dispatch_op(torch.add, val1, val2)

def sub(val1, val2):
    return _dispatch_op(torch.sub, val1, val2)

def div(val1, val2):
    return _dispatch_op(torch.div, val1, val2)


def cat(reps_list):
    """
    Concatenate (direct sum) a :obj:`list` of :obj:`SO3Tensor` representations.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`SO3Tensor`

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
