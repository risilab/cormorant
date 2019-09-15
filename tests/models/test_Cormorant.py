import torch
import pytest

from cormorant.models import Cormorant
from cormorant.data.utils import initialize_datasets

class ArgsPlaceholder:
    num_train = 10
    num_test = 10
    num_valid = 10

class TestCormorant():

    def get_dataset(self):
        args = ArgsPlaceholder()
        # Initialize dataloder
        datadir = './data/'
        dataset = 'qm9'
        subset = None

        data_init = initialize_datasets(args, datadir, dataset, subset=subset,
                                force_download=False, subtract_thermo=True)

        args, datasets, num_species, charge_scale = data_init

        return datasets, num_species, charge_scale



    @pytest.mark.parametrize('maxl', [1, 2, [1, 2]])
    @pytest.mark.parametrize('max_sh', [1, 2])
    @pytest.mark.parametrize('num_channels', [1, 2, 5, [1, 2], [2, 1, 3, 4]])
    @pytest.mark.parametrize('level_gain', [1, 10])
    def test_Cormorant(self, maxl, max_sh, num_channels, level_gain):
        num_cg_levels = 3
        num_species = 5
        cutoff_type = ['learn']
        hard_cut_rad = 1.
        soft_cut_rad = 1.
        soft_cut_width = 1.
        weight_init = 'rand'
        charge_power = 2
        basis_set = (3, 3)
        charge_scale = 1
        gaussian_mask = False
        top = 'linear'
        input = 'linear'
        num_mpnn_layers = 2
        cormorant = Cormorant(maxl, max_sh, num_cg_levels, num_channels, num_species,
                            cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                            weight_init, level_gain, charge_power, basis_set,
                            charge_scale, gaussian_mask,
                            top, input, num_mpnn_layers)

        datasets, num_species, charge_scale = self.get_dataset()

        data = next(iter(datasets['train']))

        cormorant(data)
