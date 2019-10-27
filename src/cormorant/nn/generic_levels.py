import torch
import torch.nn as nn

from cormorant.cg_lib import CGModule
from cormorant.so3_lib import SO3Tau, SO3Scalar


class DotMatrix(CGModule):
    r"""
    Constructs a matrix of dot-products between scalars of the same representation type, as used in the edge levels.

    """
    def __init__(self, tau_in=None, cat=True, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        self.tau_in = tau_in
        self.cat = cat

        if self.tau_in is not None:
            if cat:
                self.tau = SO3Tau([sum(tau_in)] * len(tau_in))
            else:
                self.tau = SO3Tau([t for t in tau_in])
            self.signs = [torch.tensor(-1.).pow(torch.arange(-ell, ell+1).float()).to(device=self.device, dtype=self.dtype).unsqueeze(-1) for ell in range(len(tau_in)+1)]
            self.conj = torch.tensor([1., -1.]).to(device=self.device, dtype=self.dtype)
        else:
            self.tau = None
            self.signs = None

    def forward(self, reps):
        """
        Performs the forward pass.

        Parameters
        ----------
        reps : :class:`SO3Vec <cormorant.so3_lib.SO3Vec>`
            Input SO3 Vector. 
        
        Returns
        -------
        dot_products : :class:`SO3Scalar <cormorant.so3_lib.SO3Scalar>`
            SO3 scalars representing a Matrix of form :math:`(\psi_i \cdot \psi_j)_c`, where c is a channel index with :math:`|C| = \sum_l \tau_l`.
        """
        if self.tau_in is not None and self.tau_in != reps.tau:
            raise ValueError('Initialized tau not consistent with tau from forward! {} {}'.format(self.tau_in, reps.tau))

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

        return SO3Scalar(dot_products)


class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.

    Parameters
    ----------
    num_in : int
        Number of input channels
    num_out : int
        Number of output channels
    num_hidden : int, optional
        Number of hidden layers.
    layer_width : int, optional
        Width of each hidden layer (number of channels).
    activation : string, optional
        Type of nonlinearity to use.
    device : :obj:`torch.device`, optional
        Device to initialize the level to
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize the level to
    """

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

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation
        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

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
