import torch

from cormorant.cg_lib import SO3Tensor, SO3Tau

class SO3Weight(SO3Tensor):
    """
    Core class for creating and tracking SO(3) Weights that
    are used to part-wise mix a :obj:`SO3Vec`.

    At the core of each :obj:`SO3Weight` is a list of :obj:`torch.Tensors` with
    shape `(C_{out}, C_{in}, 2)`, where:

    * `C_{in}` is the channels/multiplicity (tau) of the input :obj:`SO3Vec`.
    * `C_{out}` is the channels/multiplicity (tau) of the output :obj:`SO3Vec`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a SO(3) Weight object.
    """

    @property
    def bdim(self):
        return None

    @property
    def cdim(self):
        return -2

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return -1

    def check_data(self, data):
        shapes = set(part.shape for part in data)
        shapes = shapes.pop()

        if not shapes[self.zdim] == 2:
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))
