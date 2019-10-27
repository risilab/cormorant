import torch

from abc import ABC, abstractmethod


from cormorant.so3_lib import so3_torch, so3_tau

SO3Tau = so3_tau.SO3Tau


class SO3Tensor(ABC):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).

    Parameters
    ----------
    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """
    def __init__(self, data, ignore_check=False):
        if isinstance(data, type(self)):
            data = data.data

        if not ignore_check:
            self.check_data(data)

        self._data = data

    @abstractmethod
    def check_data(self, data):
        """
        Implement a data checking method.
        """
        pass

    @property
    @abstractmethod
    def bdim(self):
        """
        Define the batch dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def cdim(self):
        """
        Define the tau (channels) dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def rdim(self):
        """
        Define the representation (2*l+1) dimension for each part. Should be None
        if it is not applicable for this type of SO3Tensor.
        """
        pass

    @property
    @abstractmethod
    def zdim(self):
        """
        Define the complex dimension for each part.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_shape(batch, weight, channels):
        """
        Generate the shape of part based upon a batch size, multiplicity/number
        of channels, and weight.
        """
        pass

    def __len__(self):
        """
        Length of SO3Vec.
        """
        return len(self._data)

    @property
    def maxl(self):
        """
        Maximum weight (maxl) of SO3 object.

        Returns
        -------
        int
        """
        return len(self._data) - 1

    def truncate(self, maxl):
        """
        Update the maximum weight (`maxl`) by truncating parts of the
        :class:`SO3Tensor` if they correspond to weights greater than `maxl`.

        Parameters
        ----------
        maxl : :obj:`int`
            Maximum weight to truncate the representation to.

        Returns
        -------
        :class:`SO3Tensor` subclass
            Truncated :class:`SO3Tensor`
        """
        return self[:maxl+1]

    @property
    def tau(self):
        """
        Multiplicity of each weight if SO3 object.

        Returns
        -------
        :obj:`SO3Tau`
        """
        return SO3Tau([part.shape[self.cdim] for part in self])

    @property
    def bshape(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        bshapes = [p.shape[self.bdim] for p in self]

        if len(set(bshapes)) != 1:
            raise ValueError('Every part must have the same shape! {}'.format(bshapes))

        return bshapes

    @property
    def shapes(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        return [p.shape for p in self]

    @property
    def channels(self):
        """
        Constructs :obj:`SO3Tau`, and then gets the corresponding `SO3Tau.channels`
        method.
        """
        return self.tau.channels

    @property
    def device(self):
        if any(self._data[0].device != part.device for part in self._data):
            raise ValueError('Not all parts on same device!')

        return self._data[0].device

    @property
    def dtype(self):
        if any(self._data[0].dtype != part.dtype for part in self._data):
            raise ValueError('Not all parts using same data type!')

        return self._data[0].dtype

    def keys(self):
        return range(len(self))

    def values(self):
        return iter(self._data)

    def items(self):
        return zip(range(len(self)), self._data)

    def __iter__(self):
        """
        Loop over SO3Vec
        """
        for t in self._data:
            yield t

    def __getitem__(self, idx):
        """
        Get item of SO3Vec.
        """
        if type(idx) is slice:
            return self.__class__(self._data[idx])
        else:
            return self._data[idx]

    def __setitem__(self, idx, val):
        """
        Set index of SO3Vec.
        """
        self._data[idx] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`SO3Vec` compatible objects.
        """
        if len(self) != len(other):
            return False
        return all((part1 == part2).all() for part1, part2 in zip(self, other))

    @staticmethod
    def allclose(rep1, rep2, **kwargs):
        """
        Check equality of two :obj:`SO3Tensor` compatible objects.
        """
        if len(rep1) != len(rep2):
            raise ValueError('')
        return all(torch.allclose(part1, part2, **kwargs) for part1, part2 in zip(rep1, rep2))

    def __and__(self, other):
        return self.cat([self, other])

    def __rand__(self, other):
        return self.cat([other, self])

    def __str__(self):
        return str(list(self._data))

    __datar__ = __str__

    @classmethod
    def requires_grad(cls):
        return cls([t.requires_grad() for t in self._data])

    def requires_grad_(self, requires_grad=True):
        self._data = [t.requires_grad_(requires_grad) for t in self._data]
        return self

    def to(self, *args, **kwargs):
        self._data = [t.to(*args, **kwargs) for t in self._data]
        return self

    def cpu(self):
        self._data = [t.cpu() for t in self._data]
        return self

    def cuda(self, **kwargs):
        self._data = [t.cuda(**kwargs) for t in self._data]
        return self

    def long(self):
        self._data = [t.long() for t in self._data]
        return self

    def byte(self):
        self._data = [t.byte() for t in self._data]
        return self

    def bool(self):
        self._data = [t.bool() for t in self._data]
        return self

    def half(self):
        self._data = [t.half() for t in self._data]
        return self

    def float(self):
        self._data = [t.float() for t in self._data]
        return self

    def double(self):
        self._data = [t.double() for t in self._data]
        return self

    def clone(self):
        return type(self)([t.clone() for t in self])

    def detach(self):
        return type(self)([t.detach() for t in self])

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return type(self)([t.grad for t in self])

    def add(self, other):
        return so3_torch.add(self, other)

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.add(self, other)

    __radd__ = __add__

    def sub(self, other):
        return so3_torch.sub(self, other)

    def __sub__(self, other):
        """
        Subtract element wise `torch.Tensors`
        """
        return so3_torch.sub(self, other)

    __rsub__ = __sub__

    def mul(self, other):
        return so3_torch.mul(self, other)

    def complex_mul(self, other):
        return so3_torch.mul(self, other)

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.mul(self, other)

    __rmul__ = __mul__

    def div(self, other):
        return so3_torch.div(self, other)

    def __truediv__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.div(self, other)

    __rtruediv__ = __truediv__

    def abs(self):
        """
        Calculate the element-wise absolute value of the :class:`torch.SO3Tensor`.

        Warning
        -------
        Will break covariance!
        """

        return type(self)([part.abs() for part in self])

    __abs__ = abs

    def max(self):
        """
        Returns a list of maximum values of each part in the
        :class:`torch.SO3Tensor`.
        """

        return [part.max() for part in self]

    def min(self):
        """
        Returns a list of minimum values of each part in the
        :class:`torch.SO3Tensor`.
        """

        return [part.min() for part in self]

    @classmethod
    def rand(cls, batch, tau, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.rand(shape, device=device, dtype=dtype,
                               requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def randn(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.randn(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def zeros(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.zeros(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def ones(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.ones(shape, device=device, dtype=dtype,
                               requires_grad=requires_grad) for shape in shapes])
