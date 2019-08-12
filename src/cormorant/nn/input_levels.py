import torch
import torch.nn as nn

from cormorant.nn.generic_levels import BasicMLP
from cormorant.nn.position_levels import RadPolyTrig
from cormorant.nn.mask_levels import MaskLevel


############# Input to network #############

class InputLinear(nn.Module):
    def __init__(self, num_in, num_out, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(InputLinear, self).__init__()

        self.num_in = num_in
        self.num_out = num_out

        self.lin = nn.Linear(num_in, 2*num_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, input_scalars, atom_mask, *ignore):
        atom_mask = atom_mask.unsqueeze(-1)

        out = torch.where(atom_mask, self.lin(input_scalars), self.zero)
        out = out.view(input_scalars.shape[0:2] + (self.num_out, 1, 2))

        return out


class InputMPNN(nn.Module):
    def __init__(self, channels_in, channels_out, num_layers=1,
                 soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, cutoff_type=['learn'],
                 channels_mlp=-1, num_hidden=1, layer_width=256,
                 activation='leakyrelu', basis_set=(3, 3),
                 device=torch.device('cpu'), dtype=torch.float):
        super(InputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad

        if channels_mlp < 0:
            channels_mlp = max(channels_in, channels_out)

        # List of channels at each level. The factor of two accounts for
        # the fact that the passed messages are concatenated with the input states.
        channels_lvls = [channels_in] + [channels_mlp]*(num_layers-1) + [2*channels_out]

        self.channels_in = channels_in
        self.channels_mlp = channels_mlp
        self.channels_out = channels_out

        # Set up MLPs
        self.mlps = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.rad_filts = nn.ModuleList()

        for chan_in, chan_out in zip(channels_lvls[:-1], channels_lvls[1:]):
            rad_filt = RadPolyTrig(0, basis_set, chan_in, mix='real', device=device, dtype=dtype)
            # mask = MaskLevel(chan_in, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type, device=device, dtype=dtype)
            mask = MaskLevel(1, hard_cut_rad, soft_cut_rad, soft_cut_width, ['soft', 'hard'], device=device, dtype=dtype)
            mlp = BasicMLP(2*chan_in, chan_out, num_hidden=num_hidden, layer_width=layer_width, device=device, dtype=dtype)

            self.mlps.append(mlp)
            self.masks.append(mask)
            self.rad_filts.append(rad_filt)

        self.dtype = dtype
        self.device = device

    def forward(self, features, atom_mask, edge_mask, norms):
        # Unsqueeze the atom mask to match the appropriate dimensions later
        atom_mask = atom_mask.unsqueeze(-1)

        # Get the shape of the input to reshape at the end
        s = features.shape

        # Loop over MPNN levels. There is no "edge network" here.
        # Instead, there is just masked radial functions, that take
        # the role of the adjacency matrix.
        for mlp, rad_filt, mask in zip(self.mlps, self.rad_filts, self.masks):
            # Construct the learnable radial functions
            rad = rad_filt(norms, edge_mask)
            # Convert to a form that MaskLevel expects
            rad[0] = rad[0].unsqueeze(-1)

            # Mask the position function if desired
            edge = mask(rad, edge_mask, norms)
            # Convert to a form that MatMul expects
            edge = edge[0].squeeze(-1)

            # Now pass messages using matrix multiplication with the edge features
            # Einsum b: batch, a: atom, c: channel, x: to be summed over
            features_mp = torch.einsum('baxc,bxc->bac', edge, features)

            # Concatenate the passed messages with the original features
            features_mp = torch.cat([features_mp, features], dim=-1)

            # Now apply a masked MLP
            features = mlp(features_mp, mask=atom_mask)

        # The output are the MLP features reshaped into a set of complex numbers.
        out = features.view(s[0:2] + (self.channels_out, 1, 2))

        return out
