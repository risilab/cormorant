import logging
import torch.nn as nn

from cormorant.models import CormorantAtomLevel, CormorantEdgeLevel
from cormorant.cg_lib import CGModule


class CormorantCG(CGModule):
    def __init__(self, maxl, max_sh, tau_in_atom, tau_in_edge, tau_pos,
                 num_cg_levels, num_channels,
                 level_gain, weight_init,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False,
                 device=None, dtype=None, cg_dict=None):
        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.max_sh = max_sh

        tau_atom_in = tau_in_atom.tau if type(tau_in_atom) is CGModule else tau_in_atom
        tau_edge_in = tau_in_edge.tau if type(tau_in_edge) is CGModule else tau_in_edge

        logging.info('{} {}'.format(tau_atom_in, tau_edge_in))

        atom_levels = nn.ModuleList()
        edge_levels = nn.ModuleList()

        tau_atom, tau_edge = tau_atom_in, tau_edge_in

        for level in range(num_cg_levels):
            # First add the edge, since the output type determines the next level
            edge_lvl = CormorantEdgeLevel(tau_atom, tau_edge, tau_pos[level], num_channels[level], max_sh[level],
                                          cutoff_type, hard_cut_rad[level], soft_cut_rad[level], soft_cut_width[level],
                                          gaussian_mask=gaussian_mask, device=device, dtype=dtype)
            edge_levels.append(edge_lvl)
            tau_edge = edge_lvl.tau

            # Now add the NBody level
            atom_lvl = CormorantAtomLevel(tau_atom, tau_edge, maxl[level], num_channels[level+1],
                                          level_gain[level], weight_init,
                                          device=device, dtype=dtype, cg_dict=cg_dict)
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
        atom_reps :  SO3 Vector
            Input atom representations.
        atom_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom})`.
        edge_net : SO3 Scalar or None`
            Input edge scalar features.
        edge_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        rad_funcs : :obj:`list` of SO3 Scalars
            The (possibly learnable) radial filters.
        edge_mask : :obj:`torch.Tensor`
            Matrix of the magnitudes of relative position vectors of pairs of atoms.
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        sph_harm : SO3 Vector
            Representation of spherical harmonics calculated from the relative
            position vectors of pairs of points.

        Returns
        -------
        atoms_all : list of SO3 Vectors
            The concatenated output of the representations output at each level.
        edges_all : list of SO3 Scalars
            The concatenated output of the scalar edge network output at each level.
        """
        assert len(self.atom_levels) == len(self.edge_levels) == len(rad_funcs)

        # Construct iterated multipoles
        atoms_all = []
        edges_all = []

        for idx, (atom_level, edge_level, max_sh) in enumerate(zip(self.atom_levels, self.edge_levels, self.max_sh)):
            edge_net = edge_level(edge_net, atom_reps, rad_funcs[idx], edge_mask, norms)
            edge_reps = edge_net * sph_harm[:max_sh+1]
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)

            atoms_all.append(atom_reps)
            edges_all.append(edge_net)

        return atoms_all, edges_all
