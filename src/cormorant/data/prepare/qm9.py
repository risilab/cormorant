import numpy as np
import torch

import logging, os, urllib

from os.path import join as join
import urllib.request

from cormorant.data.prepare.process import process_xyz_files, process_xyz_gdb9
from cormorant.data.prepare.utils import download_data, is_int, cleanup_file

def download_dataset_qm9(datadir, dataname, splits=None, subtract_thermo=False, exclude=True, cleanup=True):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)

    logging.info('Downloading and processing QM9 dataset. Output will be in directory: {}.'.format(gdb9dir))

    logging.info('Beginning download of GDB9 dataset!')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    logging.info('GDB9 dataset downloaded successfully!')

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_gdb9(gdb9dir, cleanup)

    # Process GDB9 dataset, and return dictionary of splits
    gdb9_data = {}
    for split, split_idx in splits.items():
        gdb9_data[split] = process_xyz_files(gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True)

    # Subtract thermochemical energy if desired.
    if subtract_thermo:
        therm_energy, therm_target = get_thermo_dict(gdb9dir, cleanup)
        raise NotImplementedError('Not finished yet...')

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in gdb9_data.items():
        savedir = join(gdb9dir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    logging.info('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = np.array([int(idx) for idx in excluded_strings if is_int(idx)])

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(len(excluded_idxs))

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array([idx for idx in range(Ngdb9) if idx not in excluded_idxs])

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    train, test, valid = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])[:-1]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_target = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Loop over file of thermochemical energies
    therm_energy = {}
    with open(therm_energy_file) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()
            if split[0] in ['H', 'C', 'N', 'O', 'F']:
                therm_energy[split[0]] = torch.tensor([float(x) for x in split[1:]])

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy, therm_target
