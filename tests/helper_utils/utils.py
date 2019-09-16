import torch
from torch.utils.data import DataLoader


class ArgsPlaceholder:
    num_train = 10
    num_test = 10
    num_valid = 10

    batch_size = 5
    shuffle = False
    num_workers = 1


def get_dataset():
    from cormorant.data import initialize_datasets
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
    from cormorant.data import collate_fn
    args, datasets, num_species, charge_scale = get_dataset()

    # Construct PyTorch dataloaders from datasets
    dataloader = DataLoader(datasets['train'],
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)

    return dataloader, num_species, charge_scale


def complex_from_numpy(z, dtype=torch.float, device=torch.device('cpu')):
    """ Take a numpy array and output a complex array of the same size. """
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), -1)


def numpy_from_complex(z):
    """ Take a a complex array and return the commensurate numpy array. """
    zr = torch.numpy(z[..., 0])
    zi = torch.numpy(z[..., 0])
    return zr + 1j * zi
