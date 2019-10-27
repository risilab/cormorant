import torch

from math import inf

from itertools import zip_longest

class SO3Tau():
    """
    Class for keeping track of multiplicity (number of channels) of a SO(3)
    vector.

    Parameters
    ----------
    tau : :class:`list` of :class:`int`, :class:`SO3Tau`, or class with `.tau` property.
        Multiplicity of an SO(3) vector.
    """
    def __init__(self, tau):
        if type(tau) in [list, tuple]:
            if not all(type(t) == int for t in tau):
                raise ValueError('Input must be list or tuple of ints! {} {}'.format(type(tau), [type(t) for t in tau]))
        else:
            try:
                tau = tau.tau
            except AttributeError:
                raise AttributeError('Input is of type %s does not have a defined .tau property!' % type(tau))

        self._tau = tuple(tau)

    @property
    def maxl(self):
        return len(self._tau) - 1

    def keys(self):
        return range(len(self))

    def values(self):
        return self._tau

    def items(self):
        return zip(self._tau, range(len(self)))

    def __iter__(self):
        """
        Loop over SO3Tau
        """
        for t in self._tau:
            yield t

    def __getitem__(self, idx):
        """
        Get item of SO3Tau.
        """
        if type(idx) is slice:
            return SO3Tau(self._tau[idx])
        else:
            return self._tau[idx]

    def __len__(self):
        """
        Length of SO3Tau
        """
        return len(self._tau)

    def __setitem__(self, idx, val):
        """
        Set index of SO3Tau
        """
        self._tau[idx] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`SO3Tau` objects or :math:`SO3Tau` and
        a list.
        """
        self_tau = tuple(self._tau)
        other_tau = tuple(other)

        return self_tau == other_tau

    @staticmethod
    def cat(tau_list):
        """
        Return the multiplicity :class:`SO3Tau` corresponding to the concatenation
        (direct sum) of a list of objects of type :class:`SO3Tensor`.
        
        Parameters
        ----------
        tau_list : :class:`list` of :class:`SO3Tau` or :class:`list` of :class:`int`s
            List of multiplicites of input :class:`SO3Tensor`

        Return
        ------

        tau : :class:`SO3Tau`
            Output tau of direct sum of input :class:`SO3Tensor`.
        """
        return SO3Tau([sum(taus) for taus in zip_longest(*tau_list, fillvalue=0)])


    def __and__(self, other):
        return SO3Tau.cat([self, other])

    def __rand__(self, other):
        return SO3Tau.cat([self, other])

    def __str__(self):
        return str(list(self._tau))

    __repr__ = __str__

    def __add__(self, other):
        return SO3Tau(list(self) + list(other))

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        if type(other) is int:
            return self
        return SO3Tau(list(other) + list(self))

    @staticmethod
    def from_rep(rep):
        """
        Construct SO3Tau object from an SO3Vector representation.

        Parameters
        ----------
        rep : :obj:`SO3Tensor` :obj:`list` of :obj:`torch.Tensors`
            Input representation.

        """
        from cormorant.so3_lib.so3_tensor import SO3Tensor

        if rep is None:
            return SO3Tau([])

        if isinstance(rep, SO3Tensor):
            return rep.tau

        if torch.is_tensor(rep):
            raise ValueError('Input not compatible with SO3Tensor')
        elif type(rep) in [list, tuple] and any(type(irrep) != torch.Tensor for irrep in rep):
            raise ValueError('Input not compatible with SO3Tensor')

        ells = [(irrep[0].shape[-2] - 1) // 2 for irrep in rep]

        minl, maxl = ells[0], ells[-1]

        assert ells == list(range(minl, maxl+1)), 'Rep must be continuous from minl to maxl'

        tau = [irrep.shape[-3] for irrep in rep]

        return SO3Tau(tau)

    @property
    def tau(self):
        return self._tau

    @property
    def channels(self):
        channels = set(self._tau)
        if len(channels) == 1:
            return channels.pop()
        else:
            return None
