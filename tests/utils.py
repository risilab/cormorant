import torch
import pytest

from torch.utils.data import DataLoader

from cormorant.models import Cormorant
from cormorant.data import initialize_datasets, collate_fn


class ArgsPlaceholder:
    num_train = 10
    num_test = 10
    num_valid = 10

    batch_size = 5
    shuffle = False
    num_workers = 1

def get_dataset():
    args = ArgsPlaceholder()
    # Initialize dataloder
    datadir = './data/'
    dataset = 'qm9'
    subset = None

    data_init = initialize_datasets(args, datadir, dataset, subset=subset,
                            force_download=False, subtract_thermo=True)

    args, datasets, num_species, charge_scale = data_init

    return args, datasets, num_species, charge_scale

def get_dataloader():
    args, datasets, num_species, charge_scale = get_dataset()

    # Construct PyTorch dataloaders from datasets
    dataloader = DataLoader(datasets['train'],
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return dataloader, num_species, charge_scale
