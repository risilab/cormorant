import torch
import torch.nn as nn

from . import BasicMLP, cat_reps

############# Get Scalars #############

class GetScalars(nn.Module):
    def __init__(self, tau_levels, full_scalars=True, device=torch.device('cpu'), dtype=torch.float):
        super(GetScalars, self).__init__()

        self.device = device
        self.dtype = dtype

        self.maxl = max([len(tau) for tau in tau_levels]) - 1

        signs_tr = [torch.pow(-1, torch.arange(-m, m+1.)) for m in range(self.maxl+1)]
        signs_tr = [torch.stack([s, -s], dim=-1) for s in signs_tr]
        self.signs_tr = [s.view(1, 1, 1, -1, 2).to(device=device, dtype=dtype) for s in signs_tr]

        split_l0 = [tau[0] for tau in tau_levels]
        split_full = [sum(tau) for tau in tau_levels]

        self.full_scalars = full_scalars
        if full_scalars:
            self.num_scalars = sum(split_l0) + sum(split_full)
            self.split = split_l0 + split_full
        else:
            self.num_scalars = sum(split_l0)
            self.split = split_l0

        print('Number of scalars at top:', self.num_scalars)

    def forward(self, reps_levels):

        reps = cat_reps(reps_levels)

        scalars = reps[0]

        if self.full_scalars:
            scalars_tr  = [(sign*part*part.flip(-2)).sum(dim=(-1, -2), keepdim=True) for part, sign in zip(reps, self.signs_tr)]
            scalars_mag = [(part*part).sum(dim=(-1, -2), keepdim=True) for part in reps]

            scalars_full = [torch.cat([s_tr, s_mag], dim=-1) for s_tr, s_mag in zip(scalars_tr, scalars_mag)]

            scalars = [scalars] + scalars_full

            scalars = torch.cat(scalars, dim=-3)

        # print(torch.tensor([p.abs().sum() for p in scalars.split(self.split, dim=2)]))

        return scalars


############# Output of network #############

class OutputLinear(nn.Module):
    def __init__(self, num_scalars, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias

        self.lin = nn.Linear(2*num_scalars, 1, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, scalars, ignore=True):
        s = scalars.shape
        scalars = scalars.view((s[0], s[1], -1)).sum(1)

        predict = self.lin(scalars)

        predict = predict.squeeze(-1)

        return predict


class OutputPMLP(nn.Module):
    """ Iterated MLP of the type used in KLT """
    def __init__(self, num_scalars, num_mixed=64, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_mixed, 1, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, scalars, mask):
        # Reshape scalars appropriately
        scalars = scalars.view(scalars.shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        mask = mask.unsqueeze(-1)
        x = torch.where(mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        predict = predict.squeeze(-1)

        return predict


class OutputMLP(nn.Module):
    """
    Multilayer perceptron.
    """

    def __init__(self, num_scalars, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputMLP, self).__init__()

        self.num_scalars = num_scalars
        self.basic_mlp = BasicMLP(2*num_scalars, 1, num_hidden=num_hidden, layer_width=layer_width, activation=activation, device=device, dtype=dtype)

    def forward(self, scalars, ignore=None):
        scalars = scalars.sum(1)
        scalars = scalars.view((scalars.shape[0], 2*self.num_scalars))

        predict = self.basic_mlp(scalars)

        predict = predict.squeeze(-1)

        return predict
