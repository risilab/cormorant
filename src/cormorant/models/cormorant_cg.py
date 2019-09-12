import torch
import torch.nn as nn

from cormorant.models import CormorantAtomLevel, CormorantEdgeLevel

from cormorant.nn import MaskLevel, DotMatrix
from cormorant.nn import CatMixReps
from cormorant.cg_lib import CGProduct, CGModule

import logging


class CormorantCG(CGModule):
    def __init__(self, maxl, tau_in_atom, tau_in_edge, tau_pos,
                 num_cg_levels, num_channels,
                 level_gain, weight_init,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False,
                 device=None, dtype=None, cg_dict=None):
        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype = self.device, self.dtype

        tau_atom_in = atom_in.tau if type(tau_in_atom) is CGModule else tau_in_atom
        tau_edge_in = edge_in.tau if type(tau_in_edge) is CGModule else tau_in_edge

        logging.info('{} {}'.format(tau_atom_in, tau_edge_in))

        atom_levels = nn.ModuleList()
        edge_levels = nn.ModuleList()

        tau_atom, tau_edge = tau_atom_in, tau_edge_in

        for level in range(num_cg_levels):
            # First add the edge, since the output type determines the next level
            edge_lvl = CormorantEdgeLevel(tau_atom, tau_edge, tau_pos[level], num_channels[level],
                                      cutoff_type, hard_cut_rad[level], soft_cut_rad[level], soft_cut_width[level],
                                      gaussian_mask=gaussian_mask, cg_dict=self.cg_dict)
            edge_levels.append(edge_lvl)
            tau_edge = edge_lvl.tau

            # Now add the NBody level
            atom_lvl = CormorantAtomLevel(tau_atom, tau_edge, maxl[level], num_channels[level+1], level_gain[level], weight_init,
                                        cg_dict=self.cg_dict)
            atom_levels.append(atom_lvl)
            tau_atom = atom_lvl.tau

            logging.info('{} {}'.format(tau_atom, tau_edge))

        self.atom_levels = atom_levels
        self.edge_levels = edge_levels

        self.tau_levels_atom = [level.tau for level in atom_levels]
        self.tau_levels_edge = [level.tau for level in edge_levels]

    def forward(self, atom_reps, atom_mask, edge_net, edge_mask, rad_funcs, norms, sph_harm):
        """
        Runs a forward pass of the Cormorant CG layers.

        Parameters
        ----------
        atom_reps : :obj:`list` of :obj:`torch.Tensor`
            Input atom representations. List is length `maxl+1`, each with shape
            :math:`(N_{batch}, N_{atom}, N_{channels}, 2*l+1, 2)`
        atom_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom})`.
        edge_net : :obj:`list` of :obj:`torch.Tensor`
            Input edge scalar features. List is length `maxl+1`, each with shape
            :math:`(N_{batch}, N_{atom}, N_{atom}, N_{channels}, 2)`
        edge_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        rad_funcs : :obj:`list` of :obj:`list` of :obj:`torch.Tensor`
            The (possibly learnable) radial filters output from
            :obj:`cormorant.nn.position_functions.RadialFilters`.
        edge_mask : :obj:`torch.Tensor`
            Matrix of the magnitudes of relative position vectors of pairs of atoms.
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        sph_harm : :obj:`list` of :obj:`torch.Tensor`
            Representation of spherical harmonics calculated from the relative
            position vectors of pairs of points. Each tensor has shape
            :math:`(N_{batch}, N_{atom}, N_{atom}, N_{channels}, 2*l+1, 2)`
            for `l` up to `maxl_sh`.

        Returns
        -------
        atoms_all : :obj:`list` of :obj:`list` of :obj:`torch.Tensor`
            The concatenated output of the representations output at each level.
        edges_all : :obj:`list` of :obj:`list` of :obj:`torch.Tensor`
            The concatenated output of the scalar edge network output at each level.
        """
        assert len(self.atom_levels) == len(self.edge_levels) == len(rad_funcs)

        # Construct iterated multipoles
        atoms_all = []
        edges_all = []

        for idx, (atom_level, edge_level) in enumerate(zip(self.atom_levels, self.edge_levels)):
            edge_net = edge_level(edge_net, atom_reps, rad_funcs[idx], edge_mask, atom_mask, norms, sph_harm)
            # edge_reps = [scalar_mult_rep(edge, sph_harm) for (edge, sph_harm) in zip(edge_net, spherical_harmonics)]
            edge_reps = edge_net * sph_harm
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)
            atoms_all.append(atom_reps)
            edges_all.append(edge_net)

        return atoms_all, edges_all
