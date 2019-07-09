import torch
import numpy as np

import logging, os

from torch.utils.data import DataLoader
from cormorant.data.dataset import MolecularDataset
from cormorant.data.prepare import prepare_dataset

def initialize_datasets(datadir, dataset, subset=None, splits=None):
    """
    Initialize datasets.
    """
    
    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(datadir, dataset, subset, splits)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: MolecularDataset(data) for split, data in datasets.items()}

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) == 1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    return datasets, num_species, max_charge
