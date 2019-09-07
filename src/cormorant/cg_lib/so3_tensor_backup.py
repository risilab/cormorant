import torch

from abc import ABC, abstractmethod
from itertools import zip_longest

from cormorant.cg_lib import SO3Tau

from cormorant.cg_lib import so3_torch, so3_tensor, so3_tensor_base

class SO3Tensor(so3_tensor_base.SO3TensorBase):

    def add(self, other):
        return so3_torch.add(self, other)

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.add(self, other)

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        return so3_torch.add(other, self)

    @staticmethod
    def _mul_type_check(type1, type2):
        pass

    def mul(self, other):
        return so3_torch.mul(self, other)

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.mul(self, other)

    def __rmul__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        return so3_torch.mul(self, other)
