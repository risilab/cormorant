import torch
import torch.nn as nn

from cormorant.models import CormorantAtomLevel, CormorantEdgeLevel

from cormorant.nn import MaskLevel, DotMatrix
from cormorant.nn import CatMixReps
from cormorant.cg_lib import CGProduct, CGModule

import logging


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

        tau_atom_in = atom_in.tau if type(tau_in_atom) is CGModule else tau_in_atom
        tau_edge_in = edge_in.tau if type(tau_in_edge) is CGModule else tau_in_edge

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
        
        #### DEBUG ####
        from cormorant.so3_lib import rotations as rot
        from cormorant.so3_lib import SO3WignerD
        print(dir(self))
        print(self.maxl)
        maxl = 10  # SETTING IT BIG ENOUGH FOR TESTING USE... NOT GENERALIZEABLE
        device, dtype = self.device, self.dtype
        D, R, _ = rot.gen_rot(maxl, device=device, dtype=dtype)
        D = SO3WignerD(D).to(device, dtype)
        atom_reps = atom_reps.apply_wigner(D)
        sph_harm_rot = sph_harm.apply_wigner(D)
        print('sph harm is different')
        for si_rot, si in zip(sph_harm_rot, sph_harm):
            print(torch.max(torch.abs(si_rot - si)))

        # atoms_rot_all = []
        # edges_rot_all = []
        ###############

        for idx, (atom_level, edge_level, max_sh) in enumerate(zip(self.atom_levels, self.edge_levels, self.max_sh)):
            # #### DEBUG ####
            if edge_net is not None:
                edge_net_copy = edge_net
            else:
                edge_net_copy = None
            atom_reps_copy = atom_reps.apply_wigner(D)
            print('atom rep copy check')
            for si_rot, si in zip(atom_reps_copy, atom_reps):
                print(torch.max(torch.abs(si_rot - si)))
            ###############

            edge_net = edge_level(edge_net, atom_reps, rad_funcs[idx], edge_mask, norms)
            edge_reps = edge_net * sph_harm
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)

            atoms_all.append(atom_reps)
            edges_all.append(edge_net)

            #### DEBUG ####
            edge_net_rot = edge_level(edge_net_copy, atom_reps_copy, rad_funcs[idx], edge_mask, norms)
            edge_reps_rot = edge_net_rot * sph_harm_rot
            edge_reps_rot_2 = edge_net_rot * sph_harm
            atom_reps_rot = atom_level(atom_reps_copy, edge_reps_rot, atom_mask)
            atom_reps_rot_2 = atom_level(atom_reps_copy, edge_reps_rot_2, atom_mask)

            print('~~Test Covariance Layer %d~~' % (idx +1))
            print('Max Abs Error:')
            print('edge net (before multiplying by spherical harmonics)')
            for k, (a, b) in enumerate(zip(edge_net, edge_net_rot)):
                print("l=%d" % k, torch.max(torch.abs(a - b)))
            print('edge reps (after multiplying by spherical harmonics)')
            for k, (a, b) in enumerate(zip(edge_reps.apply_wigner(D), edge_reps_rot)):
                print("l=%d" % k, torch.max(torch.abs(a - b)))
            
            print('atom reps')
            atom_reps_out_rot = atom_reps.apply_wigner(D)
            for k, (a, b) in enumerate(zip(atom_reps_out_rot, atom_reps_rot)):
                print("l=%d" % k, torch.max(torch.abs(a - b)))
            
            print('atom reps_2')
            for k, (a, b) in enumerate(zip(atom_reps_out_rot, atom_reps_rot_2)):
                print("l=%d" % k, torch.max(torch.abs(a - b)))
            ###############
        #### DEBUG ####
        raise Exception
        ###############

        return atoms_all, edges_all
