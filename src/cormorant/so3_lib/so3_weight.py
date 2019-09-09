import torch

from cormorant.so3_lib import so3_tensor, so3_tau
SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor

from torch.nn import Parameter, ParameterList

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

    @staticmethod
    def _get_shape(batch, t_out, t_in):
        return (t_out, t_in, 2)

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

    def as_parameter(self):
        """
        Return the weight as a :obj:`ParameterList` of :obj:`Parameter` so
        the weights can be added as parameters to a :obj:`torch.Module`.
        """
        return ParameterList([Parameter(weight) for weight in self._data])

    @staticmethod
    def rand(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Weight`.
        """

        shapes = [(t1, t2, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.rand(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def randn(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random-normal :obj:`SO3Weight`.
        """

        shapes = [(t1, t2, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.randn(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def zeros(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Weight`.
        """

        shapes = [(t1, t2, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.randn(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def zeros(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new all-zeros :obj:`SO3Weight`.
        """

        shapes = [(t1, t2, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.zeros(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def ones(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new all-ones :obj:`SO3Weight`.
        """

        shapes = [(t1, t2, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.ones(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])
