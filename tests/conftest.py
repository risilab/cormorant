import sys
import os
import numpy as np
import torch
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper_utils'))


@pytest.fixture(scope='session')
def sample_batch():
    np.random.seed(8675309)
    num_atoms = np.array([8, 6, 10, 8])
    max_ntm = np.max(num_atoms)
    batch_size = len(num_atoms)

    num_species = 3
    charge_scale = num_species

    positions = np.zeros((batch_size, max_ntm, 3))
    atom_mask = np.zeros((batch_size, max_ntm))
    edge_mask = np.zeros((batch_size, max_ntm, max_ntm))
    charges = np.zeros((batch_size, max_ntm))
    one_hot = np.zeros((batch_size, max_ntm, num_species))
    for i, n in enumerate(num_atoms):
        positions[i, :n] = np.random.randn(n, 3)
        atom_mask[i, :n] = np.ones(n)
        edge_mask[i, :n, :n] = np.ones(n, n)
        charges_i = np.random.randint(1, num_species+1, size=n)
        charges[i, :n] = charges_i
        one_hot[i, np.arange(n), charges_i-1] = 1.

    data = {'positions': torch.from_numpy(positions).float(),
            'atom_mask': torch.from_numpy(atom_mask).bool(),
            'edge_mask': torch.from_numpy(edge_mask).bool(),
            'one_hot': torch.from_numpy(one_hot).bool(),
            'charges': torch.from_numpy(charges).int(),
            'num_atoms': torch.from_numpy(num_atoms).int()
            }

    return data, num_species, charge_scale
