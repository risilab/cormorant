import torch
import torch.nn as nn

import logging

from cormorant.cg_lib import SphericalHarmonicsRel

from cormorant.models.cormorant_levels import CormorantAtomLevel, CormorantEdgeLevel

from cormorant.nn import RadialFilters
from cormorant.nn import InputLinear, InputMPNN
from cormorant.nn import OutputLinear, OutputPMLP, GetScalars
from cormorant.nn import scalar_mult_rep

def expand_var_list(var, num_cg_levels):
    if type(var) is list:
        var_list = var + (num_cg_levels-len(var))*[var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

class Cormorant(nn.Module):
    def __init__(self, num_cg_levels, maxl, max_sh, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask,
                 top, input, num_mpnn_layers, activation='leakyrelu',
                 device=torch.device('cpu'), dtype=torch.float):
        super(Cormorant, self).__init__()

        logging.info('Initializing network!')

        self.num_cg_levels = num_cg_levels
        self.maxl = maxl
        self.max_sh = max_sh

        self.device = device
        self.dtype = dtype

        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels)

        self.num_channels = num_channels
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.num_species = num_species

        logging.info('hard_cut_rad: {}'.format(hard_cut_rad))
        logging.info('soft_cut_rad: {}'.format(soft_cut_rad))
        logging.info('soft_cut_width: {}'.format(soft_cut_width))
        logging.info('maxl: {}'.format(maxl))
        logging.info('max_sh: {}'.format(max_sh))
        logging.info('num_channels: {}'.format(num_channels))

        # Set up spherical harmonics
        self.spherical_harmonics_rel = SphericalHarmonicsRel(max(self.max_sh), sh_norm='unit', device=device, dtype=dtype)

        # Set up position functions, now independent of spherical harmonics
        self.position_functions = RadialFilters(max_sh, basis_set, num_channels, num_cg_levels, device=device, dtype=dtype)
        tau_pos = self.position_functions.tau

        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]

        input = input.lower()
        if input == 'linear':
            self.input_func = InputLinear(num_scalars_in, num_scalars_out, device=self.device, dtype=self.dtype)
        elif input == 'mpnn':
            self.input_func = InputMPNN(num_scalars_in, num_scalars_out, num_mpnn_layers, soft_cut_rad[0], soft_cut_width[0], hard_cut_rad[0], activation=activation, device=self.device, dtype=self.dtype)
        else:
            raise ValueError('Improper choice of input featurization of network! {}'.format(input))

        tau_in = [num_scalars_out]

        tau_edge = [0]

        logging.info('{} {}'.format(tau_in, tau_edge))

        atom_levels = nn.ModuleList()
        edge_levels = nn.ModuleList()
        for level in range(self.num_cg_levels):
            # First add the edge, since the output type determines the next level
            edge_lvl = CormorantEdgeLevel(tau_edge, tau_in, tau_pos[level], num_channels[level],
                                      cutoff_type, hard_cut_rad[level], soft_cut_rad[level], soft_cut_width[level],
                                      gaussian_mask=gaussian_mask, device=device, dtype=dtype)
            edge_levels.append(edge_lvl)
            tau_edge = edge_lvl.tau_out

            # Now add the NBody level
            atom_lvl = CormorantAtomLevel(tau_in, tau_edge, maxl[level], num_channels[level], level_gain[level], weight_init,
                                        device=device, dtype=dtype)
            atom_levels.append(atom_lvl)
            tau_in = atom_lvl.tau_out

            logging.info('{} {}'.format(tau_in, tau_edge))

        self.atom_levels = atom_levels
        self.edge_levels = edge_levels

        self.tau_levels_all = [level.taus for level in atom_levels]
        self.tau_levels_out = [level.tau_out for level in atom_levels]

        self.scalar_func = GetScalars(self.tau_levels_out, device=device, dtype=dtype)

        num_scalars = self.scalar_func.num_scalars

        top = top.lower()
        if top == 'linear':
            self.top_func = OutputLinear(num_scalars, bias=True, device=self.device, dtype=self.dtype)
        elif top == 'pmlp':
            self.top_func = OutputPMLP(num_scalars, activation=activation, device=self.device, dtype=self.dtype)
        else:
            raise ValueError('Improper choice of top of network! {}'.format(top))

        logging.info('Model initialized. Number of parameters: {}'.format(sum([p.nelement() for p in self.parameters()])))


    def forward(self, data, covariance_test=False):
        input_scalars, atom_mask, atom_positions, edge_mask = self.prepare_input(data)

        spherical_harmonics, norms = self.spherical_harmonics_rel(atom_positions, atom_positions)
        rad_func_levels = self.position_functions(norms, edge_mask * (norms > 0))

        atom_reps = [self.input_func(input_scalars, atom_mask, edge_mask, norms)]
        edge_net = [torch.tensor([]).to(self.device, self.dtype)]

        # Construct iterated multipoles
        atoms_all = []
        edges_all = []

        for idx, (atom_level, edge_level) in enumerate(zip(self.atom_levels, self.edge_levels)):
            edge_net = edge_level(edge_net, atom_reps, rad_func_levels[idx], edge_mask, atom_mask, norms, spherical_harmonics)
            edge_reps = [scalar_mult_rep(edge, sph_harm) for (edge, sph_harm) in zip(edge_net, spherical_harmonics)]
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)
            atoms_all.append(atom_reps)
            edges_all.append(edge_net)

        # Construct scalars for network output
        scalars = self.scalar_func(atoms_all)
        prediction = self.top_func(scalars, atom_mask)

        # Covariance test
        if covariance_test:
            return prediction, atoms_all, atoms_all
        else:
            return prediction

    def prepare_input(self, data):
        charge_power, charge_scale, device, dtype = self.charge_power, self.charge_scale, self.device, self.dtype

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        atom_mask = data['atom_mask'].to(device, torch.uint8)
        edge_mask = data['edge_mask'].to(device, torch.uint8)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        return scalars, atom_mask, atom_positions, edge_mask
