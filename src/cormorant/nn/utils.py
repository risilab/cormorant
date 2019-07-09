import torch
import torch.nn as nn
from torch.nn import Module, Parameter, ParameterList


### Multiply a representation by a list of scalars

def scalar_mult_rep(scalar, rep):
    scalar_r, scalar_i = scalar.unsqueeze(-2).unbind(-1)
    rep_r, rep_i = rep.unbind(-1)

    return torch.stack([rep_r*scalar_r - rep_i*scalar_i, rep_r*scalar_i + rep_i*scalar_r], dim=-1)

### Mix a representation

def mix_rep(weights, rep, real=False):
    """ Apply an element-wise mixing matrix to the fragment index of each irrep """
    assert(len(weights) == len(rep)), 'Number of weights must match number of reps!'
    if real:
        mixed_rep = [mix_irrep_real(weight, part) for weight, part in zip(weights, rep)]
    else:
        mixed_rep = [mix_irrep_cplx(weight, part) for weight, part in zip(weights, rep)]
    return mixed_rep


def mix_irrep_cplx(w, z):
    """ Mix SO3part with complex weights. """
    wr, wi = w.unbind(-1)
    zr, zi = z.unbind(-1)

    real = torch.matmul(wr, zr) - torch.matmul(wi, zi)
    imag = torch.matmul(wr, zi) + torch.matmul(wi, zr)

    return torch.stack([real, imag], -1)


def mix_irrep_real(w, z):
    """ Mix SO3part with real weights. """
    zr, zi = z.unbind(-1)

    return torch.stack((torch.matmul(w, zr), torch.matmul(w, zi)), -1)

#### Non module concatenate reps

def cat_reps(rep_list, scalars_only=False):
    """ Concatenate a list of SO3 representations. """

    rep_ls = [set((part.shape[-2]-1)//2 for part in rep) for rep in rep_list]
    if scalars_only:
        all_ls = [0]
    else:
        all_ls = set()
        for ls in rep_ls: all_ls.update(ls)

    return [torch.cat([vec[l] for (vec, ls) in zip(rep_list, rep_ls) if l in ls], dim=-3) for l in all_ls]


### Weight initialization

def init_mix_reps_weights(tau1, tau2, weight_init, equal_lengths=True, real=False, gain=1, device=torch.device('cpu'), dtype=torch.float):
    assert(len(tau1) == len(tau2) and equal_lengths), 'Taus must have same maxl if equal length is True! tau1={} tau2={}'.format(tau1, tau2)

    if real:
        weights = [weight_init_func((t1, t2), weight_init, real=real, gain=gain).to(device=device, dtype=dtype) for t1, t2 in zip(tau1, tau2)]
    else:
        weights = [weight_init_func((t1, t2, 2), weight_init, real=real, gain=gain).to(device=device, dtype=dtype) for t1, t2 in zip(tau1, tau2)]

    return weights

def weight_init_func(shape, weight_init, real=False, gain=1):
    if callable(weight_init):
        return weight_init

    """ Initialize weights """
    if weight_init in ['GraphFlow', 'gf']:
        init_func = lambda shape: gain / max(shape) * (torch.randint(0, 10, shape).float() / 10 * torch.pow(-1, torch.randint(0, 2, shape)).float())
    elif weight_init in ['unif', 'uniform', 'rand']:
        init_func = lambda shape: gain / max(shape) * (2*torch.rand(shape)-1)
    elif weight_init in ['norm', 'normal', 'randn']:
        init_func = lambda shape: gain / max(shape) * torch.randn(shape)
    elif weight_init in ['xnorm', 'xavier_normal', 'xav_norm']:
        init_func = lambda shape: torch.nn.init.xavier_normal_(torch.empty(shape), gain)
    elif weight_init in ['xunif', 'xavier_uniform', 'xav_unif']:
        init_func = lambda shape: torch.nn.init.xavier_(torch.empty(shape), gain)
    else:
        raise ValueError('Incorrect initialization type! {}'.format(weight_init))

    return init_func(shape)

# Save reps

def save_grads(reps):
    for part in reps: part.requires_grad_()
    def closure(part):
        def assign_grad(grad):
            if grad is not None: part.add_(grad)
            return None
        return assign_grad
    grads = [torch.zeros_like(part) for part in reps]
    for (part, grad) in zip(reps, grads): part.register_hook(closure(grad))

    return grads

def save_reps(reps_dict, to_save, retain_grad=False):
    if 'reps_out' not in to_save:
        to_save.append('reps_out')

    reps_dict = {key: val for key, val in reps_dict.items() if (key in to_save and len(val) > 0)}

    if retain_grad:
        reps_dict.update({key+'_grad': save_grads(val) for key, val in reps_dict.items()})

    return reps_dict

def broadcastable(tau1, tau2):
    for t1, t2 in zip(tau1[::-1], tau2[::-1]):
        if not (t1 == 1 or t2 == 1 or t1 == t2):
            return False
    return True

def conjugate_rep(rep):
    repc = [part.clone() for part in rep]
    for part in repc:
        part[..., 1] *= -1
    return repc
