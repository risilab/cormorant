import torch

from .group_reps import GroupRep, GroupIrrep, GroupTau


class MixIrrep(GroupIrrep):
    """
    Class to define a single SO(3) weight, which is a matrix that will mix
    a specific SO(3) irrep.
    """
    def __init__(self, data, real=False, device=None, dtype=None):

        # Initialize CGModule
        super().__init__()

    _integer_weights = True
    _multiplicity_dim = -1

    @classmethod
    def rand(cls, mult_in, mult_out, real=False, device=None, dtype=None):
        shape = (mult_out, mult_in) if real else (mult_out, mult_in, 2)

        data = torch.rand(shape, device=device, dtype=dtype)
        if requires_grad:
            data.requires_grad_()

        return cls(data)

    @classmethod
    def expected_weight(cls, data):
        """
        These weights are not a real representation, so there is no concept
        of the expected weight, therefore return None.
        """
        return None

    @property
    def weight(self):
        """
        These weights are not a real representation, so there is no concept
        of the expected weight, therefore return None.
        """
        return None

class MixRep(GroupRep):
    """
    Class to define a vector of SO(3) weights, which can be used to mix a SO3Tensor
    from tau_{in} to tau_{out}
    """
    def __init__(self, data, real=False, device=None, dtype=None):

        # Initialize CGModule
        super().__init__()

    _Irrep = MixIrrep

    @classmethod
    def rand(self, tau_in, tau_out, **kwargs):
        assert(len(tau_in) == len(tau_out)), 'Must have same input/output length!'
        return GroupMix([IrrepMix.rand(mult_in, mult_out, **kwargs) for mult_in, mult_out in zip(tau_in, tau_out)])
