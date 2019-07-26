import torch
import torch.nn as nn

from ..cg_lib import CGProduct

from ..nn import MaskLevel
from ..nn import CatMixReps, CatMixRepsScalar, DotMatrix

class CormorantEdgeLevel(nn.Module):
    def __init__(self, tau_edge, tau_in, tau_rad, nout,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False,
                 device=torch.device('cpu'), dtype=torch.float):
        super(CormorantEdgeLevel, self).__init__()

        # Set up type of edge network depending on specified input operations
        self.dot_matrix = DotMatrix(tau_in, cat=cat, device=device, dtype=dtype)
        tau_dot = self.dot_matrix.tau_out

        # Set up mixing layer
        edge_taus = [tau_edge, tau_dot, tau_rad]
        self.cat_mix = CatMixRepsScalar(edge_taus, nout, real=False, device=device, dtype=dtype)
        self.tau_out = self.cat_mix.tau_out

        # Set up edge mask layer
        self.mask_layer = MaskLevel(nout, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type, gaussian_mask=gaussian_mask, device=device, dtype=dtype)

        # Standard bookeeping
        self.device = device
        self.dtype = dtype

    def forward(self, edge_in, atom_reps, rad_funcs, base_mask, mask, norms, spherical_harmonics):
        # Caculate the dot product matrix.
        edge_dot = self.dot_matrix(atom_reps)

        # Concatenate and mix the three different types of edge features together
        edge_mix = self.cat_mix([edge_in, edge_dot, rad_funcs])

        # Apply mask to layer -- For now, only can be done after mixing.
        edge_net = self.mask_layer(edge_mix, base_mask, norms)

        return edge_net


class CormorantAtomLevel(nn.Module):
    """
    Basic NBody level initialization.
    """
    def __init__(self, tau_in, tau_pos, maxl, num_channels, level_gain, weight_init,
                 device=torch.device('cpu'), dtype=torch.float):
        super(CormorantAtomLevel, self).__init__()

        self.maxl = maxl
        self.num_channels = num_channels

        self.tau_in = tau_in
        self.tau_pos = tau_pos

        # Operations linear in input reps
        self.cg_aggregate = CGProduct(tau_pos, tau_in, maxl=self.maxl, aggregate=True, device=device, dtype=dtype)
        tau_ag = list(self.cg_aggregate.tau_out)

        self.cg_power = CGProduct(tau_in, tau_in, maxl=self.maxl, device=device, dtype=dtype)
        tau_sq = list(self.cg_power.tau_out)

        self.cat_mix = CatMixReps([tau_ag, tau_in, tau_sq], num_channels, maxl=self.maxl, weight_init=weight_init, gain=level_gain, device=device, dtype=dtype)
        self.tau_out = self.cat_mix.tau_out

        self.taus = {'tau_out': self.tau_out, 'tau_ag': tau_ag, 'tau_sq': tau_sq, 'tau_id': tau_in}

        self.device = device
        self.dtype = dtype

    def forward(self, atom_reps, edge_reps, mask):
        # Aggregate information based upon edge reps
        reps_ag = self.cg_aggregate(edge_reps, atom_reps)

        # CG non-linearity for each atom
        reps_sq = self.cg_power(atom_reps, atom_reps)

        # Concatenate and mix results
        reps_out = self.cat_mix([reps_ag, atom_reps, reps_sq])

        return reps_out
