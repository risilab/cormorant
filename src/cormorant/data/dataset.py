import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf

import logging

class MolecularDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, data, included_species=None):

        self.data = data

        if included_species is None:
            included_species = torch.unique(self.data['charges'])

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_pts = len(data['charges'])
        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}
