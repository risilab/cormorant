import torch
from torch.nn import Module

from math import sqrt

from itertools import zip_longest
from functools import reduce

from cormorant.cg_lib import CGModule
from cormorant.so3_lib import so3_torch, SO3Weight, SO3Tau

class MixReps(CGModule):
    """
    Module to linearly mix a representation from an input type `tau_in` to an
    output type `tau_out`.

    Input must have pre-defined types `tau_in` and `tau_out`.

    Parameters
    ----------
    tau_in : :obj:`SO3Tau` (or compatible object).
        Input tau of representation.
    tau_out : :obj:`SO3Tau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.

    """
    def __init__(self, tau_in, tau_out, real=False, weight_init='randn', gain=1,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        tau_in = SO3Tau(tau_in)
        tau_out = SO3Tau(tau_out) if type(tau_out) is not int else tau_out

        # Allow one to set the output tau to a pre-specified number of output channels.
        if type(tau_out) is int:
            tau_out = [tau_out] * len(tau_in)

        self.tau_in = SO3Tau(tau_in)
        self.tau_out = SO3Tau(tau_out)
        self.real = real

        weights = SO3Weight.rand(self.tau_in, self.tau_out, device=device, dtype=dtype)
        weights = (2*weights - 1) / sqrt(gain)
        self.weights = weights.as_parameter()

    def forward(self, rep):
        """
        Linearly mix a represention.

        Parameters
        ----------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Representation to mix.

        Returns
        -------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Mixed representation.
        """
        if SO3Tau.from_rep(rep) != self.tau_in:
            raise ValueError('Tau of input rep does not match initialized tau!'
                            ' rep: {} tau: {}'.format(SO3Tau.from_rep(rep), self.tau_in))

        return so3_torch.mix_rep(self.weights, rep, real=self.real)

    @property
    def tau(self):
        return self.tau_out


class CatReps(Module):
    """
    Module to concanteate a list of reps. Specify input type for error checking
    and to allow network to fit into main architecture.

    Parameters
    ----------
    taus_in : :obj:`list` of :obj:`SO3Tau` or compatible.
        List of taus of input reps.
    maxl : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    """
    def __init__(self, taus_in, maxl=None):
        super().__init__()

        self.taus_in = taus_in = [SO3Tau(tau) for tau in taus_in]

        if maxl is None:
            maxl = max([tau.maxl for tau in taus_in])
        self.maxl = maxl

        self.tau_out = reduce(lambda x,y: x & y, taus_in)[:self.maxl+1]

    def forward(self, reps):
        """
        Concatenate a list of reps

        Parameters
        ----------
        reps : :obj:`list` of :obj:`list` of :obj:`torch.Tensor`
            List of representations to concatenate.

        Returns
        -------
        reps_cat : :obj:`list` of :obj:`torch.Tensor`

        """
        # Error checking
        reps_taus_in = [SO3Tau.from_rep(rep) for rep in reps]
        if reps_taus_in != self.taus_in:
            raise ValueError('Tau of input reps does not match predefined version!'
                                'got: {} expected: {}'.format(reps_taus_in, self.taus_in))

        return so3_torch.cat(reps)

    @property
    def tau(self):
        return self.tau_out

class CatMixReps(CGModule):
    """
    Module to concatenate mix a list of representation representations using
    :obj:`cormorant.nn.CatReps`, and then linearly mix them using
    :obj:`cormorant.nn.MixReps`.

    Parameters
    ----------
    taus_in : List of :obj:`SO3Tau` (or compatible object).
        List of input tau of representation.
    tau_out : :obj:`SO3Tau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    maxl : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.

    """
    def __init__(self, taus_in, tau_out, maxl=None,
                 real=False, weight_init='randn', gain=1,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)

        self.cat_reps = CatReps(taus_in, maxl=maxl)
        self.mix_reps = MixReps(self.cat_reps.tau, tau_out,
                                real=real, weight_init=weight_init, gain=gain,
                                device=device, dtype=dtype)

        self.tau_out = SO3Tau(self.mix_reps)

    def forward(self, reps_in):
        """
        Concatenate and linearly mix a list of representations.

        Parameters
        ----------
        reps_in : :obj:`list` of :obj:`list` of :obj:`torch.Tensors`
            List of input representations.

        Returns
        -------
        reps_out : :obj:`list` of :obj:`torch.Tensors`
            Representation as a result of combining and mixing input reps.
        """
        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)

        return reps_out

    @property
    def tau(self):
        return self.tau_out
