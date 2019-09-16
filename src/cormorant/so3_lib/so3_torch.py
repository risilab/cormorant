import torch

from itertools import zip_longest

from cormorant.so3_lib import so3_tau, so3_tensor
from cormorant.so3_lib import so3_vec, so3_scalar, so3_weight, so3_wigner_d

SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor
SO3Vec = so3_vec.SO3Vec
SO3Scalar = so3_scalar.SO3Scalar
SO3Weight = so3_weight.SO3Weight
SO3WignerD = so3_wigner_d.SO3WignerD

from cormorant.so3_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar
from cormorant.so3_lib.cplx_lib import mix_zweight_zvec, mix_zweight_zscalar

import cormorant.so3_lib.rotations as rot

def _check_maxl(val1, val2):
    if len(val1) != len(val2):
        raise ValueError('Two SO3Tensor subclasses have different maxl values '
                         '({} {})!'.format(len(val1)-1, len(val2)-1))

def _dispatch_op(op, val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    SO3Tensor.
    """

    # Hack to make SO3Vec/SO3Scalar multiplication work
    # TODO: Figure out better way of doing this?
    if isinstance(val1, SO3Scalar) and isinstance(val2, SO3Vec):
        _check_maxl(val1, val2)
        applied_op = [op(part1.unsqueeze(val2.rdim), part2)
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    elif isinstance(val1, SO3Vec) and isinstance(val2, SO3Scalar):
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2.unsqueeze(val1.rdim))
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    # Both va1 and val2 are other instances of SO3Tensor
    elif isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        _check_maxl(val1, val2)
        val1._bin_op_type_check(type(val1), type(val2))
        val2._bin_op_type_check(type(val1), type(val2))
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

def mix(weights, rep):
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

    if isinstance(rep, SO3Vec):
        rep_mix = SO3Vec([mix_zweight_zvec(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Scalar):
        rep_mix = SO3Scalar([mix_zweight_zscalar(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Weight):
        rep_mix = SO3Weight([mix_zweight_zvec(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Tensor):
        raise NotImplementedError('Mixing for object {} not yet implemented!'.format(type(rep)))
    else:
        raise ValueError('Mixing only implemented for SO3Tensor subclasses!')

    return rep_mix


def cat_mix(weights, reps_list):
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

    return mix(weights, cat(reps_list))


def apply_wigner(rep, wigner_d):
    """
    Apply a Wigner-D rotation to a :obj:`SO3Vec` representation
    """
    return SO3Vec(rot.rotate_rep(wigner_d, rep))
