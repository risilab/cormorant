import torch

from math import inf

from itertools import zip_longest

def cg_product_tau(tau1, tau2, maxl=inf):
    """
    Calulate output multiplicty of the CG Product of two SO3 Vectors
    given the multiplicty of two input SO3 Vectors.

    Parameters
    ==========
    tau1 : :obj:`list` of ``int`` or :obj:`SO3Tau`
        Multiplicity of first representation.

    tau2 : :obj:`list` of ``int`` or :obj:`SO3Tau`
        Multiplicity of second representation.

    maxl : int
        Largest weight to include in CG Product.

    Return
    ======

    tau : :obj:`list` of ``int`` or :obj:`SO3Tau`
        Multiplicity of output representation.

    """
    tau1 = list(tau1)
    tau2 = list(tau2)

    L1, L2 = len(tau1) - 1, len(tau2) - 1
    L = min(L1 + L2, maxl)

    tau = [0]*(L+1)

    for l1 in range(L1+1):
        for l2 in range(L2+1):
            lmin, lmax = abs(l1-l2), min(l1+l2, maxl)
            for l in range(lmin, lmax+1):
                tau[l] += tau1[l1]

    return SO3Tau(tau)


class SO3Tau():
    """
    Class for keeping track of multiplicity (number of channels) of a SO(3)
    vector.

    Parameters
    ----------

    tau : :obj:`list` of ``int`` or :obj:`SO3Tau`
        Multiplicity of a SO(3) vector.
    """
    def __init__(self, tau):
        if type(tau) is SO3Tau:
            tau = list(tau)

        assert type(tau) in [list, tuple] and all(type(t) == int for t in tau), 'Input must be list or tuple of ints! {} {}'.format(type(tau), [type(t) for t in tau])

        self._tau = tau

    def __iter__(self):
        """
        Loop over SO3Tau
        """
        for t in self._tau:
            yield t

    def __getitem__(self, idx):
        """
        Get index of SO3Tau
        """
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
        self_tau = list(self._tau)
        other_tau = list(other)

        return self_tau == other_tau

    @staticmethod
    def cat(tau1, tau2):
        """
        Return the :obj:`SO3Tau` corresponding to the concatenation of two tensors.

        Parameters
        ----------
        tau1 : :obj:`SO3Tau` or list of ints
            Multiplicity of :rep1:
        tau2 : :obj:`SO3Tau` or list of ints
            Multiplicity of :rep2:

        Return
        ------

        tau : :obj:`SO3Tau`
            Output type of direct sum of ``rep1`` and ``rep2``

        Example
        -------
        >>> tau1 = SO3Tau([1, 2, 3])
        >>> tau2 = SO3Tau([1, 1])
        >>> tau = tau1 & tau2
        >>> print(tau)

        """
        return SO3Tau([t1 + t2 for t1, t2 in zip_longest(tau1, tau2, fillvalue=0)])

    def __and__(self, other):
        return SO3Tau.cat(self, other)

    def __rand__(self, other):
        return SO3Tau.cat(self, other)

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
        rep : :obj:`list` of :obj:`torch.Tensors`
            Input representation.

        """
        assert type(rep) is list and all(type(irrep) == torch.Tensor for irrep in rep), 'Input must be list of torch.Tensors! {} {}'.format(type(rep), [type(irrep) for irrep in rep])

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
