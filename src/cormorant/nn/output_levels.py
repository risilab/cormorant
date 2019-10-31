import torch
import torch.nn as nn

from cormorant.nn import BasicMLP
from cormorant.so3_lib import cat

############# Get Scalars #############

class GetScalarsAtom(nn.Module):
    r"""
    Construct a set of scalar feature vectors for each atom by using the
    covariant atom :class:`SO3Vec` representations at various levels.

    Parameters
    ----------
    tau_levels : :class:`list` of :class:`SO3Tau`
        Multiplicities of the output :class:`SO3Vec` at each level.
    full_scalars : :class:`bool`, optional
        Construct a more complete set of scalar invariants from the full
        :class:`SO3Vec` (``true``), or just use the :math:``\ell=0`` component
        (``false``).
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, tau_levels, full_scalars=True, device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxl = max(len(tau) for tau in tau_levels) - 1

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

    def forward(self, reps_all_levels):
        """
        Forward step for :class:`GetScalarsAtom`

        Parameters
        ----------
        reps_all_levels : :class:`list` of :class:`SO3Vec`
            List of covariant atom features at each level

        Returns
        -------
        scalars : :class:`torch.Tensor`
            Invariant scalar atom features constructed from ``reps_all_levels``
        """

        reps = cat(reps_all_levels)

        scalars = reps[0]

        if self.full_scalars:
            scalars_tr = [(sign*part*part.flip(-2)).sum(dim=(-1, -2), keepdim=True) for part, sign in zip(reps, self.signs_tr)]
            scalars_mag = [(part*part).sum(dim=(-1, -2), keepdim=True) for part in reps]

            scalars_full = [torch.cat([s_tr, s_mag], dim=-1) for s_tr, s_mag in zip(scalars_tr, scalars_mag)]

            scalars = [scalars] + scalars_full

            scalars = torch.cat(scalars, dim=-3)

        return scalars


############# Output of network #############

class OutputLinear(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors. This is performed in a permutation invariant way
    by using a (batch-masked) sum over all atoms, and then applying a
    linear mixing layer to predict a single output.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, bias=True, device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias

        self.lin = nn.Linear(2*num_scalars, 1, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputLinear`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        s = atom_scalars.shape
        atom_scalars = atom_scalars.view((s[0], s[1], -1)).sum(1)  # No masking needed b/c summing over atoms

        predict = self.lin(atom_scalars)

        predict = predict.squeeze(-1)

        return predict


class OutputPMLP(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors.

    This is peformed in a three-step process::

    (1) A MLP is applied to each set of scalar atom-features.
    (2) The environments are summed up.
    (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_mixed=64, activation='leakyrelu', device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_mixed, 1, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputPMLP`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        # Reshape scalars appropriately
        atom_scalars = atom_scalars.view(atom_scalars.shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(atom_scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        atom_mask = atom_mask.unsqueeze(-1)
        x = torch.where(atom_mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        predict = predict.squeeze(-1)

        return predict
