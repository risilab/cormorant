import torch

from cormorant.cg_lib import so3_tensor, SO3Tau
SO3Tensor = so3_tensor.SO3Tensor

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
        return None

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return 2

    @property
    def tau_in(self):
        return SO3Tau([part.shape[1] for part in self])

    @property
    def tau_out(self):
        return SO3Tau([part.shape[0] for part in self])

    tau = tau_out

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Weights not currrently enabled!')

        shapes = set(part.shape for part in data)
        shapes = shapes.pop()

        if not shapes[self.zdim] == 2:
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))
