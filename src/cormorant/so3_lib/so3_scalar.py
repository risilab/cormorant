import torch

from cormorant.so3_lib import so3_tau, so3_tensor

SO3Tensor = so3_tensor.SO3Tensor
SO3Tau = so3_tau.SO3Tau


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

    @property
    def bdim(self):
        return slice(0, -2)

    @property
    def cdim(self):
        return -2

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return -1

    @staticmethod
    def _get_shape(batch, weight, channels):
        return tuple(batch) + (channels, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Scalars not currrently enabled!')

        shapes = [part.shape[self.bdim] for part in data]
        if len(set(shapes)) > 1:
            raise ValueError('Batch dimensions are not identical!')

        if any(part.shape[self.zdim] != 2 for part in data):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))
