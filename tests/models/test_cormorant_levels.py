import pytest
import torch
from cormorant.so3_lib import rotations as rot
from cormorant.so3_lib import SO3Vec

from cormorant.models import CormorantAtomLevel
from ..helper_utils.utils import get_dataloader
from cormorant.cg_lib import SphericalHarmonicsRel
# from cormorant.models import CormorantEdgeLevel


# @pytest.fixture(scope='module')
def build_environment(tau, maxl, num_channels, level_gain=1, weight_init='rand',
                      cg_dict=None):
    datasets, num_species, charge_scale = get_dataloader()
    data = next(iter(datasets))
    device, dtype = data['positions'].device, data['positions'].dtype
    sph_harms = SphericalHarmonicsRel(maxl-1, conj=True, device=device,
                                      dtype=dtype, cg_dict=cg_dict)
    return datasets, data, num_species, charge_scale, sph_harms


def prep_input(data, taus, maxl):
    atom_positions = data['positions']
    atom_scalar_list = [torch.randn(atom_positions.shape[:2] + (taus, 2*l+1, 2)) for l in range(maxl)]
    # atom_scalar_list = [torch.randn(atom_positions.shape[:2] + (num_channels, 1, 2))]
    # atom_scalar_list += [torch.zeros(atom_positions.shape[:2] + (num_channels, 2*l+1, 2)) for l in range(1, maxl)]
    atom_scalars = SO3Vec(atom_scalar_list)
    atom_mask = data['atom_mask']
    edge_mask = data['edge_mask']
    edge_scalars = torch.tensor([])
    return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions


class TestCormorantAtomLevel(object):
    # @pytest.mark.parametrize('tau_atom', [1, 3])
    # @pytest.mark.parametrize('tau_edge', [1, 3])
    @pytest.mark.parametrize('tau', [1, 3])
    @pytest.mark.parametrize('num_channels', [3])
    @pytest.mark.parametrize('maxl', [1, 3])
    def test_covariance(self, tau, num_channels, maxl):
        # setup the environment
        env = build_environment(tau, maxl, num_channels)
        datasets, data, num_species, charge_scale, sph_harms = env
        device, dtype = data['positions'].device, data['positions'].dtype
        D, R, _ = rot.gen_rot(maxl, device=device, dtype=dtype)

        # Build Atom layer
        tlist = [tau] * maxl
        print(tlist)
        atom_lvl = CormorantAtomLevel(tlist, tlist, maxl, num_channels, 1, 'rand',
                                      device=device, dtype=dtype, cg_dict=None)

        # Setup Input
        atom_rep, atom_mask, edge_scalars, edge_mask, atom_positions = prep_input(data, tau, maxl)
        atom_positions_rot = rot.rotate_cart_vec(R, atom_positions)

        # Get nonrotated data
        spherical_harmonics, norms = sph_harms(atom_positions, atom_positions)
        edge_rep_list = [torch.cat([sph_l] * tau, axis=-3) for sph_l in spherical_harmonics]
        edge_reps = SO3Vec(edge_rep_list)
        print(edge_reps.shapes)
        print(atom_rep.shapes)

        # Get Rotated output
        output = atom_lvl(atom_rep, edge_reps, atom_mask)
        output = output.apply_wigner(D)

        # Get rotated outputdata
        atom_rep_rot = atom_rep.apply_wigner(D)
        spherical_harmonics_rot, norms = sph_harms(atom_positions_rot, atom_positions_rot)
        edge_rep_list_rot = [torch.cat([sph_l] * tau, axis=-3) for sph_l in spherical_harmonics_rot]
        edge_reps_rot = SO3Vec(edge_rep_list_rot)
        output_from_rot = atom_lvl(atom_rep_rot, edge_reps_rot, atom_mask)

        for i in range(maxl):
            assert(torch.max(torch.abs(output_from_rot[i] - output[i])) < 1E-5)


# class TestCormorantEdgeLevel(object):
#     @pytest.mark.parametrize('taus', [1, 3])
#     @pytest.mark.parametrize('num_channels', [3])
#     @pytest.mark.parametrize('maxl', [1, 3])
#     def test_covariance(self, taus, num_channels, maxl):
#         env = build_environment(tau, maxl, num_channels)
#         datasets, data, num_species, charge_scale, sph_harms = env
#         device, dtype = data['positions'].device, data['positions'].dtype
#         D, R, _ = rot.gen_rot(maxl, device=device, dtype=dtype)
# 
#         # Build Atom layer
#         tlist = [tau] * maxl
#         print(tlist)
#         edge_lvl = CormorantEdgeLevel(tlist, tlist, maxl, num_channels, 1, 'rand',
#                                       device=device, dtype=dtype, cg_dict=None)
#         return
