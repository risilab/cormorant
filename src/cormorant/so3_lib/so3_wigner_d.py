import torch
from numpy import pi

# Hack to avoid circular imports
from cormorant.so3_lib import so3_tau, so3_tensor
from cormorant.so3_lib import rotations as rot

SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor


class SO3WignerD(SO3Tensor):
    """
    Core class for creating and tracking WignerD matrices.

    At the core of each :obj:`SO3WignerD` is a list of :obj:`torch.Tensors` with
    shape `(2*l+1, 2*l+1, 2)`, where:

    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Note
    ----

    For now, there is no batch or channel dimensions included. Although a
    SO3 covariant network architecture with Wigner-D matrices is possible,
    the current scheme using PyTorch built-ins would be too slow to implement.
    A custom CUDA kernel would likely be necessary, and is a work in progress.

    Warning
    -------
    The constructor __init__() does not check that the tensor is actually
    a Wigner-D matrix, (that is an irreducible representation of the group SO3)
    so it is important to ensure that the input tensor is generated appropraitely.

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    @property
    def bdim(self):
        return None

    @property
    def cdim(self):
        return None

    @property
    def rdim1(self):
        return 0

    @property
    def rdim2(self):
        return 1

    rdim = rdim2

    @property
    def zdim(self):
        return 2

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    @staticmethod
    def _get_shape(batch, l, channels):
        return (2*l+1, 2*l+1, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3WignerD not currrently enabled!')

        shapes = [part.shape for part in data]

        rdims1 = [shape[self.rdim1] for shape in shapes]
        rdims2 = [shape[self.rdim2] for shape in shapes]
        zdims = [shape[self.zdim] for shape in shapes]

        if not all(rdim1 == 2*l+1 and rdim2 == 2*l+1 for l, (rdim1, rdim2) in enumerate(zip(rdims1, rdims2))):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, list(enumerate(zip(rdims1, rdims2)))))

        if not all(zdim == 2 for zdim in zdims):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    @staticmethod
    def _bin_op_type_check(type1, type2):
        if type1 == SO3WignerD and type2 == SO3WignerD:
            raise ValueError('Cannot multiply two SO3WignerD!')

    @staticmethod
    def euler(maxl, angles=None, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new :obj:`SO3Weight`.

        If `angles=None`, will generate a uniformly distributed random Euler
        angle and then instantiate a SO3WignerD accordingly.
        """

        if angles is None:
            alpha, beta, gamma = torch.rand(3) * 2 * pi
            beta = beta / 2

        wigner_d = rot.WignerD_list(maxl, alpha, beta, gamma, device=device, dtype=dtype)

        return SO3WignerD(wigner_d)

    @staticmethod
    def rand(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def randn(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def zeros(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def ones(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')
