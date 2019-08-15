from torch import nn

from cormorant.cg_lib import global_cg_dicts

class CGModule(nn.Module):
    """
    Clebsch-Gordan module. This functions identically to a normal PyTorch
    nn.Module, except for it adds the ability to specify a
    Clebsch-Gordan dictionary, and has additional tracking behavior set up
    to allow the CG Dictionary to be compatible with the DataParallel module.

    If `cg_dict` is specified upon instantiation, then the specified
    `cg_dict` is set as the Clebsch-Gordan dictionary for the CG module.

    If `cg_dict` is not specified, and `maxl` is specified, then CGModule
    will attempt to set the local `cg_dict` based upon the global
    `cormorant.cg_lib.global_cg_dicts`. If the dictionary has not been initialized
    with the appropriate `dtype`, `device`, and `maxl`, it will be initialized
    and stored in the `global_cg_dicts`, and then set to the local `cg_dict`.

    In this way, if there are many modules that need `CGDicts`, only a single
    `CGDict` will be initialized and automatically set up. 

    Parameters
    ----------
    cg_dict : CGDict(), optional
        Specify an input CGDict to use for Clebsch-Gordan operations.
    maxl : int, optional
        Maximum weight to initialize the Clebsch-Gordan dictionary.
    device : torch.device, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : torch.device, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.
    """
    def __init__(self, cg_dict=None, maxl=None, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if cg_dict not None:
            self.cg_dict =

    def _cg_dict_to(self, device=None, dtype=None):
        pass

    def to(self, device=None, dtype=None):
        super().to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device=device)
        return self

    def cpu(self):
        super().cpu()
        return self

    def half(self):
        super().half()
        return self

    def float(self):
        super().float()
        return self

    def double(self):
        super().double()
        return self
