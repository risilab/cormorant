import torch
import torch.nn as nn
from torch.nn import Module, Parameter, ParameterList

from .utils import init_mix_reps_weights, mix_rep

############# Modules to mix/cat reps #############

class MixReps(Module):
    """ Weight mixing module for Representation that will include a type and error checking. """
    def __init__(self, tau_in, tau_out, weight_init='randn', real=False, gain=1, device=torch.device('cpu'), dtype=torch.float):
        super(MixReps, self).__init__()

        # Remove extra tailing zeros in input/output type
        while not tau_in[-1]: tau_in.pop()

        if type(tau_out) is int:
            tau_out = [tau_out] * len(tau_in)
        else:
            while not tau_out[-1]: tau_out.pop()

        self.tau_in = list(tau_in)
        self.tau_out = list(tau_out)
        self.real = real

        weights = init_mix_reps_weights(tau_out, tau_in, weight_init, real=real, gain=gain, device=device, dtype=dtype)
        self.weights = ParameterList([Parameter(weight) for weight in weights])

    def forward(self, rep):
        assert([part.shape[-3] for part in rep] == self.tau_in), 'Input rep must have same type as initialized tau! rep: {} tau: {}'.format([part.shape[-3] for part in rep], self.tau_in)

        return mix_rep(self.weights, rep, real=self.real)


class CatReps(Module):
    """ Module to concanteate a set of reps with initial type error checking. """
    def __init__(self, taus, minl=None, maxl=None, scalars_only=False):
        super(CatReps, self).__init__()

        if scalars_only:
            if (maxl or minl):
                raise ValueError('Cannot set scalars_only with maxl or minl!')
            maxl = minl = 0
        else:
            if minl is None:
                minl = 0
            if maxl is None:
                maxl = max([len(tau) for tau in taus]) - 1

        assert(minl==0), 'minl=0 not implemented yet!'

        self.maxl = maxl
        self.minl = minl
        self.scalars_only = scalars_only

        self.taus_in = taus
        self.tau_out = [sum([tau[l] for tau in taus if len(tau) >= l+1]) if l >= minl else 0 for l in range(maxl+1)]
        self.all_ls = list(range(self.minl, min(self.maxl+1, len(self.tau_out))))

    def forward(self, reps):
        reps_taus = [[part.shape[-3] for part in rep] for rep in reps]
        reps_ls = [[(part.shape[-2]-1)//2 for part in rep] for rep in reps]
        assert(reps_taus == self.taus_in), 'Tau of input reps does not match predefined version! {} {}'.format(reps_taus, self.taus_in)

        reps_cat = [[part[l] for (part, ls) in zip(reps, reps_ls) if l in ls] for l in self.all_ls]
        return [torch.cat(reps, dim=-3) for reps in reps_cat if len(reps) > 0]


class CatMixReps(Module):
    """ Module to concanteate a set of reps and then apply a mixing matrix. """
    def __init__(self, taus, tau_out, minl=None, maxl=None, scalars_only=False,
                 weight_init='randn', real=False, gain=1, device=torch.device('cpu'), dtype=torch.float):
        super(CatMixReps, self).__init__()

        self.cat_reps = CatReps(taus, minl=minl, maxl=maxl, scalars_only=scalars_only)
        tau_cat = self.cat_reps.tau_out
        self.mix_reps = MixReps(tau_cat, tau_out, weight_init=weight_init, real=real, gain=gain, device=device, dtype=dtype)
        tau_mix = self.mix_reps.tau_out
        if type(tau_out) is int:
            tau_out = tau_mix
        else:
            assert(tau_mix == tau_out), 'Something went wrong with expected type! Could it be minl/maxl? {} {}'.format(tau_mix, tau_out)

        self.taus = taus
        self.tau_cat = tau_cat
        self.tau_out = tau_out

    def forward(self, reps_in):
        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)

        return reps_out

    def scale_weights(self, scale):
        self.mix_reps.scale_weights(scale)
