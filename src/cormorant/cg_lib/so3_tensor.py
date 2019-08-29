import torch

from abs import ABC
from itertools import zip_longest

from cormorant.cg_lib import SO3Tau

class SO3Tensor(ABC):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """
    def __init__(self, data):
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
        Define the representation (2*l+1) dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def zdim(self):
        """
        Define the complex dimension for each part.
        """
        pass

    def __len__(self):
        """
        Length of SO3Vec.
        """
        return len(self._data)

    @property
    def maxl(self):
        return len(self._data) - 1

    @property
    def tau(self):
        return SO3Tau([part.shape[self.cdim] for part in self])

    @property
    def channels(self):
        return self.tau.channels

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
    def allclose(rep1, rep2):
        """
        Check equality of two :obj:`SO3Tensor` compatible objects.


        """
        if len(self) != len(other):
            raise ValueError('')
        return all(torch.allclose(part1, part2) for part1, part2 in zip(rep1, rep2))

    @classmethod
    def cat(cls, reps):
        """
        Concatenate (direct sum) a :obj:`list` of :obj:`SO3Tensor` representations.

        Parameters
        ----------
        reps : :obj:`list` of :obj:`SO3Tensor` or compatible

        Return
        ------
        rep_cat : :obj:`SO3Tensor`
            Direct sum of all :obj:`SO3Tensor` in `reps`

        """
        reps_cat = [filter(lambda x: x is not None, rep) for rep in zip_longest(*reps, fillvalue=None)]
        reps_cat = [torch.cat(reps, dim=cls.cdim) for reps in reps_cat if len(reps) > 0 else torch.tensor([])]

        return cls(reps_cat)

    def __and__(self, other):
        return self.cat([self, other])

    def __rand__(self, other):
        return self.cat([other, self])

    def __str__(self):
        return str(list(self._data))

    __datar__ = __str__

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return self.__class__([part1 + part2 for part1, part2 in zip_longest([self, other], fillvalue=0)])

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        if type(other) is int:
            return self
        return self.__class__([part1 + part2 for part1, part2 in zip_longest([self, other], fillvalue=0)])
