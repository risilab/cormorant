import torch

from sortedcontainers import SortedDict

class SO3Tau():
    """
    Multiplicity (tau) of a SO3Vector.
    """

class SO3Irrep():
    """
    SO3 Irrep
    """
    def __init__(self, data, zdim=-1, cdim=-2, rdim=-3, bdim=None):
        self._zdim = -1
        self._cdim = -2
        self._rdim = -3
        self._bdim = data.ndim()

    @property
    def zdim(self):
        """
        Complex dimension. This should not be manually changed.
        """
        return self._zdim

    @property
    def cdim(self):
        """
        Channel dimension. This should not be manually changed.
        """
        return self._cdim

    @property
    def rdim(self):
        """
        SO(3) representation dimension. This should not be manually changed.
        """
        return self._rdim

    @property
    def bdim(self):
        """
        Batch dimension(s). This should not be manually changed.
        """
        return self._bdim


class SO3Vector():
    """
    Basic SO3Vector class.

    CG Tensor module. This is to be viewed as a generalization of the torch.Tensor()
    class, with most of the functionality of a PyTorch tensor included.

    Conceputally, this is just a list of torch.Tensor() objects. However, it will
    include sanity checks, a built-in type (tau) that labels the number of
    irreducible fragments of each type.

    The basic interface is designed to be nearly identical with that of the
    torch.Tensor class. Unfortunately, the limitations of PyTorch and Python
    make it impossible to be identical. Warnings or notes will be listed
    at points where things could possible be an issue.
    """
    def __init__(self, data, copy=False, *args, **kwargs):
        # self.data setter will initialize self.device, self.dtype
        self.data = data

        self.device = kwargs['device']
        self.dtype = kwargs['dtype']
        self.requires_grad = kwargs['requires_grad']

        # Initialize CGModule
        # super().__init__(device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)

    _Irrep = None
    device = None
    dtype = None
    requires_grad = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._data = map(lambda x: x.to(*args, **kwargs), self._data)

        return self

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        """
        Set the data vector. Stored as a SortedDict, which operates essentially
        as a list but with the possibility of missing elements.

        Also includes tensor setting/checking properties at end
        """
        if (type(data) in [list, tuple]) and self._integer_weights:
            # If input is list, and group has integer weights, try to set data by enumerating list.
            # Note that GroupIrrep() is called with the weight, so it will check that the weight and
            # shape of irrep are consistent.
            self._data = SortedDict({ell: self._Irrep(part, ell) for ell, part in enumerate(data)})
        elif (type(data) in [list, tuple]) and not self._Irrep._integer_weights:
            # If input list with a list of tensors, but with an irrep without integer weights, throw an error.
            raise ValueError('Cannot initialize from list unless Representation weights are integer!')
        elif ifinstance(data, dict):
            self._data = SortedDict({weight: self._Irrep(part, weight) for weight, part in data.items()})
        else:
            raise ValueError('Input data must be list, tuple, dict, or OrderedDict!')

        self._init_group_rep()

    def _init_group_rep(self):
        """
        Initialize a group representation. Consists of two parts:
        (1) Apply consistency check to tensor.
        (2) Initialize tau and batch dimension of tensor.

        Set representation type (tau) and shape (of batch dimension) based upon input data.

        Also includes significant error checking:

        Error checking on tensor shapes. The input is a SortedDict of torch.tensors
        of the shape {weight: batch + tau_weight + irrep_shape(weight)}

        We want to check the following things:
        (1) All irreps have same dtype.
        (2) All irreps have same device.
        (3) All irreps have same batch shape.
        """
        # (1) All irreps have same dtype.
        dtypes = set(irrep.dtype for irrep in self.values())
        assert(len(dtypes) == 1), 'Can only have one input tensor data type! {}'.format(dtypes)

        # (2) All irreps have same device.

        # WARNING: there is a situation where one CUDA tensor is, e.g., `cuda:0`
        # and another is `cuda` with `torch.cuda.current_device() == 0`.
        # This will cause a crash, and for now we will ignore this.
        # To get around this, can apply `.to()`.
        devices = set(irrep.device for irrep in self.values())
        assert(len(devices) == 1), 'Can only have one input tensor devce! {}'.format(devices)

        # (3) Set batch shape, which includes check all irreps have same batch shape.
        self._set_batch()

        # Finally, set tau of Representation
        self._set_tau()

    #### tau (multiplicity of each irrep) of representation ###

    # Integer weights are used for SO(3), but groups such as SU(2)
    # will need half-integer weights and SO(4) will need 2-tuples of weights.
    def _set_tau(self):
        """
        Set representation tau (multiplicity of each irrep).
        Must be defined for each implementation!
        """
        self._tau = list(self.keys())
        # self._tau = GroupTau(self.keys(), self._Irrep._integer_weights)

    @property
    def tau(self):
        return self._tau

    #### (batch dimension) .bshape/.bsize() consistent with PyTorch notation ###

    @property
    def bshape(self):
        return self._bshape

    def bsize(self):
        return self._bshape

    def _set_batch(self):
        """
        Set batch dimension, including check all irreps have same batch shape.
        """
        bshape = [irrep.bshape for irrep in self._data.values()]
        assert(len(set(bshape)) == 1), 'Batch dimensions are not identical! {}'.format(bshape)

        self._bshape = bshape[0]

    #### Total tensor .shape/.size() consistent with PyTorch notation ###

    @property
    def shape(self):
        return {weight: irrep.shape for weight, irrep in self.items()}

    def size(self):
        return {weight: irrep.shape for weight, irrep in self.items()}

    #### Gradients -- TBD ###

    @property
    def grad(self):
        raise NotImplementedError

    #### Key/value pairs based upon SortedDict ###

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    #### Basic arithmatic operations ###
    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __div__(self, other):
        pass


    #### Iteration based upon SortedDict ###

    def __iter__(self):
        """
        Iterate over the parts of the SO3vector. It is not clear how to deal with
        missing parts. For now, don't return anything. May change in the future.
        """
        return iter(self._data)

    def __getitem__(self, idx):
        """
        Getitem/setitem are defined such that slices are over weights.

        Specifically,
        """
        return self._data[idx]

    def __setitem__(self, idx, irreps):
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            self._data[idx] = self._Irrep(irreps, idx)
        else:
            raise ValueError('Index must be integer or slice! {}'.format(idx))
