import torch
import torch.nn as nn

import logging

from cormorant.cg_lib import CGModule, SphericalHarmonicsRel

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

class Cormorant(CGModule):
    """
    Basic Cormorant Network

    Parameters
    ----------
    num_cg_levels : int
        Number of cg levels to use.
    """
    def __init__(self, maxl, max_sh, num_cg_levels, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask,
                 top, input, num_mpnn_layers, activation='leakyrelu',
                 device=None, dtype=None):

        logging.info('Initializing network!')

        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels+1)

        super().__init__(maxl=max(maxl+max_sh))
        device, dtype = self.device, self.dtype

        self.num_cg_levels = num_cg_levels
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
        self.spherical_harmonics_rel = SphericalHarmonicsRel(max(max_sh), cg_dict=self.cg_dict, sh_norm='unit')

        # Set up position functions, now independent of spherical harmonics
        self.position_functions = RadialFilters(max_sh, basis_set, num_channels, num_cg_levels, device=device, dtype=dtype)
        tau_pos = self.position_functions.tau

        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]

        self.input_func_atom = None
        self.input_func_edge = None

        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau

        self.cormorant_cg = CormorantMain(maxl, tau_in_atom, tau_in_edge, tau_pos,
                     num_channels, level_gain, weight_init,
                     cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                     cat=True, gaussian_mask=False,
                     device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_atom = self.cormorant_cg.tau_levels_atom
        tau_cg_levels_edge = self.cormorant_cg.tau_levels_edge

        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom, device=device, dtype=dtype)
        self.get_scalars_edge = GetScalarsEdge(tau_cg_levels_atom, device=device, dtype=dtype)
        num_scalars_atom = self.get_scalars_atom.num_scalars
        num_scalars_edge = self.get_scalars_edge.num_scalars

        self.output_layer_atom = None
        self.output_layer_edge = None

        logging.info('Model initialized. Number of parameters: {}'.format(sum([p.nelement() for p in self.parameters()])))

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        data : ?????
            Data input into the layer.
        covariance_test : boolean, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : ?????
            The output of the layer


        """
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        spherical_harmonics, norms = self.spherical_harmonics_rel(atom_positions, atom_positions)
        rad_func_levels = self.position_functions(norms, edge_mask * (norms > 0).byte())

        atom_reps_in = self.input_func_atom(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        atoms_all, edges_all = self.cg_level(atom_reps_in, atom_mask, edge_net_in, edge_mask,
                                             rad_funcs, norms, spherical_harmonics)

        atom_scalars = self.get_scalars_atom(atoms_all)
        edge_scalars = self.get_scalars_edge(edges_all)

        # Construct scalars for network output
        prediction = self.top_func(scalars, atom_mask)

        # Covariance test
        if covariance_test:
            return prediction, atoms_all, atoms_all
        else:
            return prediction

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        scalars : ?????
            ?????
        atom_mask : ?????
            ?????
        atom_positions: ?????
            Positions of the atoms
        edge_mask: ?????
            ?????
        """
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
