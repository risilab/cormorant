import torch

from .group_reps import GroupRep, GroupIrrep

class SO3Irrep(GroupIrrep):
    """
    Class to define an irreducible representation of SO(3).
    """
    def __init__(self, data, weight, *args, **kwargs):
        super().__init__(data, weight, *args, **kwargs)

    _integer_weights = True
    _multiplicity_dim = -3

    @classmethod
    def expected_weight(cls, data):
        """
        The expected weight for a complex SO(3) rep.
        :input: Torch tensor or SO(3) irrep.
        :output: Expected weight of irrep.
        """
        shape = data.shape
        assert(len(shape) >= 3), 'Must have at least three dimensions! (multiplicity, m=(2*l+1), cplx). {}'.format(shape)
        assert(shape[-1] == 2), 'Final axis must have length 2!'

        # Number of elements is set by 2*l+1, where w is the weight.
        return (shape[-2] - 1) // 2

    @property
    def weight(self):
        """
        These weights are not a real representation, so there is no concept
        of the expected weight, therefore return None.
        """
        shape = data.shape
        assert(len(shape) >= 3), 'Must have at least three dimensions! (multiplicity, m=(2*l+1), cplx). {}'.format(shape)
        assert(shape[-1] == 2), 'Final axis must have length 2!'

        # Number of elements is set by 2*l+1, where w is the weight.
        return (shape[-2] - 1) // 2


class SO3Rep(GroupRep):
    """
    Class to define a vector of SO(3) weights, which can be used to mix a SO3Tensor
    from tau_{in} to tau_{out}
    """
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    _Irrep = SO3Irrep
