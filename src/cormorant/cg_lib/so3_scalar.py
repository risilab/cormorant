import torch

from cormorant.cg_lib import SO3Tensor, SO3Tau, SO3Vec

class SO3Scalar(SO3Tensor):
    """
    Core class for creating and tracking SO(3) Scalars that
    are used to part-wise multiply :obj:`SO3Vec`.

    At the core of each :obj:`SO3Scalar` is a list of :obj:`torch.Tensors` with
    shape `(B, C, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a SO(3) Scalar.
    """

    def bdim(self):
        return slice(0, -2)

    def cdim(self):
        return -2

    def rdim(self):
        return None

    def zdim(self):
        return -1

    def check_data(self, data):
        shapes = set(part.shape for part in data)
        if len(shapes) > 1:
            raise ValueError('All parts (torch.Tensors) must have same number of'
                             'batch dimensions! {}'.format(part.shape for part in data))

        shapes = shapes.pop()

        if not shapes[self.zdim] == 2
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))

    def __mul__(self, other):
        """
        Multiply operation of :obj:`SO3Scalar`
        """
        if type(other) in [float, int]:
            mul = [other*part for part in self]
        elif type(other) is torch.Tensor and (other.numel() == 1):
            mul = [other*part for part in self]
        elif type(other) is torch.Tensor:
            raise NotImplementedError('Multiplication by a non-scalar tensor'
                                      'not yet implemented!')
        elif issubclass(SO3Scalar, other):
            if self.maxl != other.maxl:
                raise ValueError('SO3Vec and SO3Scalar do not have the same maxl! {} {}'.format(self.maxl, other.maxl))
            mul = [mul_zscalar_zirrep(scalar1, scalar2, zdim=self.zdim) for scalar1, scalar2 in zip(self, other)]
        elif issubclass(SO3Vec, other):
            return other.__mul__(self)
        else:
            raise NotImplementedError()

        return self.__class__(mul)
