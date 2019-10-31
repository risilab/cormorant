import torch
import torch.nn as nn

from cormorant.so3_lib import SO3Tau


class NoLayer(nn.Module):
    """
    Layer that does nothing in the Cormorant architecture.

    This exists just to demonstrate the structure one would want if edge
    features were desired at the input/output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        return None

    @property
    def tau(self):
        return SO3Tau([])

    @property
    def num_scalars(self):
        return 0


# Save reps

def save_grads(reps):
    for part in reps:
        part.requires_grad_()

    def closure(part):
        def assign_grad(grad):
            if grad is not None:
                part.add_(grad)
            return None
        return assign_grad
    grads = [torch.zeros_like(part) for part in reps]
    for (part, grad) in zip(reps, grads):
        part.register_hook(closure(grad))

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
