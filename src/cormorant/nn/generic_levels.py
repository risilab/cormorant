import torch
import torch.nn as nn
from torch.nn import Module, Parameter, ParameterList

#### DotMatrix -- a matrix of dot products as is used in the edge levels ###

class DotMatrix(nn.Module):
    """
    Constructs a matrix of dot-products between scalars of the same representation type.
    Input: Tensor of SO3-vectors psi_i. Each psi has the same tau.
    Output: Matrix of scalars (psi_i cdot psi_j)_c, where c is a channel index with |C| = \sum_\ell tau_\ell.
    """
    def __init__(self, tau_in=None, cat=True, device=torch.device('cpu'), dtype=torch.float):
        super(DotMatrix, self).__init__()
        self.tau_in = tau_in
        self.cat = cat

        if self.tau_in is not None:
            if cat:
                self.tau_out = [sum(tau_in)] * len(tau_in)
            else:
                self.tau_out = [t for t in tau_in]
            self.signs = [torch.tensor(-1.).pow(torch.arange(-ell, ell+1).float()).to(device=device, dtype=dtype).unsqueeze(-1) for ell in range(len(tau_in)+1)]
            self.conj = torch.tensor([1., -1.]).to(device=device, dtype=dtype)
        else:
            self.tau_out = None
            self.signs = None


    def forward(self, reps):
        assert((self.tau_in is None) or [part.shape[-3] for part in reps] == list(self.tau_in)), 'Incorrect input type specified!'

        if self.tau_in is None:
            signs = [torch.tensor(-1.).pow(torch.arange(-ell, ell+1).float()).to(device=reps[0].device, dtype=reps[0].dtype).unsqueeze(-1) for ell in range(len(tau)+1)]
            conj = torch.tensor([1., -1.], dtype=reps[0].dtype, device=reps[0].device)
        else:
            signs = self.signs
            conj = self.conj

        reps1 = [part.unsqueeze(-4) for part in reps]
        reps2 = [part.unsqueeze(-5) for part in reps]

        reps2 = [part.flip(-2)*sign for part, sign in zip(reps2, signs)]

        dot_product_r = [(part1*part2*conj).sum(dim=(-2, -1)) for part1, part2 in zip(reps1, reps2)]
        dot_product_i = [(part1*part2.flip(-1)).sum(dim=(-2, -1)) for part1, part2 in zip(reps1, reps2)]

        dot_products = [torch.stack([prod_r, prod_i], dim=-1) for prod_r, prod_i in zip(dot_product_r, dot_product_i)]

        if self.cat:
            dot_products = torch.cat(dot_products, dim=-2)
            dot_products = [dot_products] * len(reps)

        return dot_products

########### BasicMLP used throughout the network for various reasons ###########

class BasicMLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, num_in, num_out, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(BasicMLP, self).__init__()

        self.num_in = num_in

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_in, layer_width))
        for i in range(num_hidden-1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        self.linear.append(nn.Linear(layer_width, num_out))

        activation_fn = get_activation_fn(activation)

        self.activations = nn.ModuleList()
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        self.to(device=device, dtype=dtype)

    def forward(self, x):

        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))
        x = self.linear[-1](x)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale

def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    else:
        raise ValueError('Activation function {} not implemented!'.format(activation))
    return activation_fn
