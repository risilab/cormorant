import torch
import numpy as np

import logging, os

from torch.utils.data import DataLoader
from cormorant.data.dataset import MolecularDataset
from cormorant.data.prepare import prepare_dataset

def initialize_datasets(args, datadir, dataset, subset=None, splits=None, num_pts=None):
    """
    Initialize datasets.
    """
    # Set the number of points based upon the arguments
    if num_pts is None:
        num_pts={'train': args.num_train, 'test': args.num_test, 'valid': args.num_valid}

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

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: MolecularDataset(data, num_pts=num_pts.get(split, -1), included_species=all_species) for split, data in datasets.items()}

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) == 1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge

def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    :datasets: Dictionary of datasets
    :ignore_check: Ignores/overrides checks to make sure every split includes every species included in the entire dataset
    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique() for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0: all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] == 0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error('The number of species is not the same in all datasets!')
        else:
            raise ValueError('Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
