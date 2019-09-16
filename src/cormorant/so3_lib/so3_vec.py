import torch

# Hack to avoid circular imports
from cormorant.so3_lib import so3_tau, so3_tensor, so3_torch
from cormorant.so3_lib import so3_scalar, so3_wigner_d

SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor
SO3Scalar = so3_scalar.SO3Scalar
SO3WignerD = so3_wigner_d.SO3WignerD

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

    @property
    def bdim(self):
        return slice(0, -3)

    @property
    def cdim(self):
        return -3

    @property
    def rdim(self):
        return -2

    @property
    def zdim(self):
        return -1

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    @staticmethod
    def _get_shape(batch, l, channels):
        return tuple(batch) + (channels, 2*l+1, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Vec not currrently enabled!')

        bdims = set(part.shape[self.bdim] for part in data)
        if len(bdims) > 1:
            raise ValueError('All parts (torch.Tensors) must have same number of'
                             'batch  dimensions! {}'.format(part.shape[self.bdim] for part in data))

        shapes = [part.shape for part in data]

        cdims = [shape[self.cdim] for shape in shapes]
        rdims = [shape[self.rdim] for shape in shapes]
        zdims = [shape[self.zdim] for shape in shapes]

        if not all([rdim == 2*l+1 for l, rdim in enumerate(rdims)]):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, list(enumerate(rdims))))

        if not all([zdim == 2 for zdim in zdims]):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    @staticmethod
    def _bin_op_type_check(type1, type2):
        if type1 == SO3Vec and type2 == SO3Vec:
            raise ValueError('Cannot multiply two SO3Vecs!')


    def apply_wigner(self, wigner_d):
        """
        Apply a WignerD matrix to `self`

        Parameters
        ----------
        wigner_d : :class:`SO3WignerD`
            The Wigner D matrix rotation to apply to `self`

        Returns
        -------
        :class:`SO3Vec`
            The current :class:`SO3Vec` rotated by :class:`SO3Vec`
        """

        return so3_torch.apply_wigner(self, wigner_d)
