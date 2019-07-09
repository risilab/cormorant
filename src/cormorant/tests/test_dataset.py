from cormorant.data.utils import initialize_datasets
from cormorant.data.collate import collate_fn

import logging

logging.basicConfig(level=logging.INFO)

datasets, num_species, max_charge = initialize_datasets('/tmp/test', 'qm9')

train = datasets['train']

batch = [train[i] for i in [5, 123, 5436, 43132]]

batch_coll = collate_fn(batch)


breakpoint()
