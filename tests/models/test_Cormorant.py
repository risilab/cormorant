import torch
import pytest

from cormorant.models import Cormorant

from ..helper_utils.utils import get_dataloader

class TestCormorant():

    @pytest.mark.parametrize('maxl', [1, 2, [1, 2]])
    @pytest.mark.parametrize('max_sh', [1, 2])
    @pytest.mark.parametrize('num_channels', [1, 2, 5, [1, 2], [2, 1, 3, 4]])
    @pytest.mark.parametrize('level_gain', [1, 10])
    def test_Cormorant(self, maxl, max_sh, num_channels, level_gain):
        datasets, num_species, charge_scale = get_dataloader()

        num_cg_levels = 3
        cutoff_type = ['learn']
        hard_cut_rad = 1.
        soft_cut_rad = 1.
        soft_cut_width = 1.
        weight_init = 'rand'
        charge_power = 2
        basis_set = (3, 3)
        gaussian_mask = False
        top = 'linear'
        input = 'linear'
        num_mpnn_layers = 2

        # First test initialization
        cormorant = Cormorant(maxl, max_sh, num_cg_levels, num_channels, num_species,
                            cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                            weight_init, level_gain, charge_power, basis_set,
                            charge_scale, gaussian_mask,
                            top, input, num_mpnn_layers)

        # Get data and then try pushing through a single example
        data = next(iter(datasets))
        cormorant(data)
