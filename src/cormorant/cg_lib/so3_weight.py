import torch

from cormorant.cg_lib import SO3Tensor, SO3Tau

class SO3Weight(SO3Tensor):
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
        return None

    def cdim(self):
        return -2

    def rdim(self):
        return None

    def zdim(self):
        return -1

    def check_data(self, data):
        shapes = set(part.shape for part in data)
        shapes = shapes.pop()

        if not shapes[self.zdim] == 2:
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))
