import torch
from torch.nn import Module, Parameter, ParameterList

from itertools import zip_longest

from .utils import init_mix_reps_weights, mix_rep

############## Edge network scalar concatenation/mixing. ##############

class MixRepsScalar(Module):
    """ Weight mixing module for scalar "representations" that will include a type and error checking. """
    def __init__(self, tau_in, tau_out, weight_init='randn', real=False, gain=1, device=torch.device('cpu'), dtype=torch.float):
        super(MixRepsScalar, self).__init__()

        # Remove extra tailing zeros in input/output type
        while not tau_in[-1]: tau_in.pop()

        if type(tau_out) is int:
            tau_out = [tau_out] * len(tau_in)
        else:
            while not tau_out[-1]: tau_out.pop()

        self.tau_in = list(tau_in)
        self.tau_out = list(tau_out)

        self.real = real
        self.cat_dim = -1 if real else -2

        weights = init_mix_reps_weights(tau_in, tau_out, weight_init, real=real, gain=gain, device=device, dtype=dtype)
        self.weights = ParameterList([Parameter(weight) for weight in weights])

    def forward(self, rep):
        assert([part.shape[self.cat_dim] for part in rep] == self.tau_in), 'Input rep must have same type as initialized tau! rep: {} tau: {}'.format([part.shape[-3] for part in rep], self.tau_in)

        # Note the non-standard order compared to the irrep mixing. (Weights applied from right.)
        # TODO: Move to weight mixing from left.
        return mix_rep(rep, self.weights, real=self.real)


class CatRepsScalar(Module):
    """ Module to concanteate a set of reps with initial type error checking. """
    def __init__(self, taus, real=False, device=torch.device('cpu'), dtype=torch.float):
        super(CatRepsScalar, self).__init__()

        self.taus_in = [list(tau) for tau in taus]
        self.tau_out = [sum(tau_ell) for tau_ell in zip_longest(*taus, fillvalue=0)]

        self.empty = torch.tensor([], device=device, dtype=dtype)

        self.cat_dim = -1 if real else -2

    def forward(self, edge_ops):
        edge_taus = [[part.shape[self.cat_dim] if part.dim() > 1 else 0 for part in edge_op] for edge_op in edge_ops]
        assert(edge_taus == self.taus_in), 'Tau of input reps does not match predefined version! {} {}'.format(edge_taus, self.taus_in)

        return [torch.cat(edges, dim=self.cat_dim) for edges in zip_longest(*edge_ops, fillvalue=self.empty)]


class CatMixRepsScalar(Module):
    """ Module to concanteate a set of reps and then apply a mixing matrix. """
    def __init__(self, taus, tau_out, minl=None, maxl=None, scalars_only=False,
                 weight_init='randn', real=False, gain=1, device=torch.device('cpu'), dtype=torch.float):
        super(CatMixRepsScalar, self).__init__()

        self.cat_reps = CatRepsScalar(taus, real=real, device=device, dtype=dtype)
        tau_cat = self.cat_reps.tau_out
        self.mix_reps = MixRepsScalar(tau_cat, tau_out, weight_init=weight_init, real=real, gain=gain, device=device, dtype=dtype)
        tau_mix = self.mix_reps.tau_out

        self.taus = taus
        self.tau_cat = tau_cat
        self.tau_out = tau_mix

    def forward(self, reps_in):
        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)

        return reps_out

    def scale_weights(self, scale):
        self.mix_reps.scale_weights(scale)
