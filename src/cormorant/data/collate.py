import torch

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]

def collate_fn(batch):
    """
    Collation function for batched data that does simultaneous
    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (batch['charges'].sum(0) > 0)

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    batch['atom_mask'] = atom_mask.to(torch.bool)
    batch['edge_mask'] = edge_mask.to(torch.bool)

    return batch
