import torch

from cormorant.cg_lib import SO3Tensor, SO3Tau, SO3Scalar

class SO3Vec(SO3Tensor):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).

    At the core of each :obj:`SO3Vec` is a list of :obj:`torch.Tensors` with
    shape `(B, C, 2*l+1, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    def bdim(self):
        return slice(0, -3)

    def cdim(self):
        return -3

    def rdim(self):
        return -2

    def zdim(self):
        return -1

    def check_data(self, data):
        bdims = set(part.shape[self.bdim] for part in data)
        if len(bdims) > 1:
            raise ValueError('All parts (torch.Tensors) must have same number of'
                             'batch  dimensions! {}'.format(part.shape[self.bdim] for part in data))

        shapes = [part.shape for part in data]

        cdims = [shape[self.cdim] for shape in shapes]
        rdims = [shape[self.rdim] for shape in shapes]
        zdims = [shape[self.zdim] for shape in shapes]

        if not all([rdim == 2*l+1 for l, rdim in enumerate(rdims)]):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, list(eumerate(rdims))))

        if not all([zdim == 2 for zdim in zdims]):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    def __mul__(self, other):
        """
        Define multiplication by a scalar (:obj:`float`, or :obj:`torch.Tensor`),
        or a :obj:`SO3Scalar`.
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
            mul = [mul_zscalar_zirrep(scalar, part, cdim=self.cdim, zdim=self.zdim) for scalar, part in zip(self, other)]
        else:
            raise NotImplementedError()

        return self.__class__(mul)
