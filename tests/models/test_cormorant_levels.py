import pytest
import torch
from cormorant.so3_lib import rotations as rot

from cormorant.models import CormorantAtomLevel
from ..helper_utils.utils import get_dataloader
from cormorant.cg_lib import SphericalHarmonicsRel
# from cormorant.models import CormorantEdgeLevel


# @pytest.fixture(scope='module')
def build_atom_level_environment(tau_atom, tau_edge, maxl, num_channels,
                                 level_gain=1, weight_init='rand', cg_dict=None):
    max_sh = 3
    datasets, num_species, charge_scale = get_dataloader()
    data = next(iter(datasets))
    device, dtype = data['positions'].device, data['positions'].dtype
    sph_harms = SphericalHarmonicsRel(max(max_sh), conj=True, device=device,
                                      dtype=dtype, cg_dict=cg_dict)

    atom_lvl = CormorantAtomLevel(tau_atom, tau_edge, maxl, num_channels,
                                  level_gain, weight_init,
                                  device=device, dtype=dtype, cg_dict=cg_dict)

    return atom_lvl, datasets, data, num_species, charge_scale, sph_harms


def prep_input(data):
    atom_positions = data['positions']
    atom_scalars = data['one_hot']
    atom_mask = data['atom_mask']
    edge_mask = data['edge_mask']
    edge_scalars = torch.tensor([])
    return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions


class TestCormorantAtomLevel(object):
    @pytest.mark.parametrize('tau_atoms', [1, 3])
    @pytest.mark.parametrize('tau_edge', [1, 3])
    @pytest.mark.parametrize('num_channels', [3])
    @pytest.mark.parametrize('natoms', [3, 5])
    @pytest.mark.parametrize('maxl', [1, 3])
    @pytest.mark.parametrize('channels', range(1, 3))
    def test_covariance(self, tau_atoms, tau_edge, num_channels, natoms, maxl, channels):
        # setup the environment
        env = build_atom_level_environment()
        atom_lvl, datasets, data, num_species, charge_scale, sph_harms = env
            
        # Grab the net dataset and generate a rotation matrix.
        data = next(iter(datasets))
        device, dtype = data['positions'].device, data['positions'].dtype
        D, R, _ = rot.gen_rot(maxl, device=device, dtype=dtype)

        batch = 3
        tau_list = [channels]*(maxl+1)

        # Setup Input
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = prep_input(data)
        atom_positions_rot = rot.rotate_cart_vec(R, atom_positions)

        
        # Get nonrotated data
        spherical_harmonics, norms = sph_harms(atom_positions, atom_positions)
        output = atom_lvl(atom_scalars, spherical_harmonics, atom_mask)
        
        # Get nonrotated data
        spherical_harmonics_rot, norms = sph_harms(atom_positions_rot, atom_positions_rot)
        output_rot = atom_lvl(atom_scalars, spherical_harmonics, atom_mask)
        return
