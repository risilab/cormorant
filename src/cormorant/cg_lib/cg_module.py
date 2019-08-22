import torch
from torch import nn

from cormorant.cg_lib import CGDict

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
    dtype : torch.dtype, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.
    """
    def __init__(self, cg_dict=None, maxl=None, device=None, dtype=None, *args, **kwargs):
        self._init_device_dtype(device, dtype)
        self._init_cg_dict(cg_dict, maxl)

        super().__init__(*args, **kwargs)

    def _init_device_dtype(self, device, dtype):
        """
        Initialize the default device and data type.

        device : torch.device, optional
            Set device for CGDict and related. If unset defaults to torch.device('cpu').

        dtype : torch.dtype, optional
            Set device for CGDict and related. If unset defaults to torch.float.

        """
        if device is None:
            self._device = torch.device('cpu')
        else:
            self._device = device

        if dtype is None:
            self._dtype = torch.float
        else:
            if not (dtype == torch.half or dtype == torch.float or dtype == torch.double):
                raise ValueError('CG Module only takes internal data types of half/float/double. Got: {}'.format(dtype))
            self._dtype = dtype

    def _init_cg_dict(self, cg_dict, maxl):
        """
        Initialize the Clebsch-Gordan dictionary.

        If cg_dict is set, check the following::
        - The dtype of cg_dict matches with self.
        - The devices of cg_dict matches with self.
        - The desired :maxl: <= :cg_dict.maxl: so that the CGDict will contain
            all necessary coefficients

        If :cg_dict: is not set, but :maxl: is set, get the cg_dict from a
        dict of global CGDict() objects.
        """
        # If cg_dict is defined, check it has the right properties
        if cg_dict is not None:
            if cg_dict.dtype != self.dtype:
                raise ValueError('CGDict dtype ({}) not match CGModule() dtype ({})'.format(cg_dict.dtype, self.dtype))

            if cg_dict.device != self.device:
                raise ValueError('CGDict device ({}) not match CGModule() device ({})'.format(cg_dict.device, self.device))

            if maxl is None:
                Warning('maxl is not defined, setting maxl based upon CGDict maxl ({}!'.format(cg_dict.maxl))

            elif maxl > cg_dict.maxl:
                Warning('CGDict maxl ({}) is smaller than CGModule() maxl ({}). Updating!'.format(cg_dict.maxl, maxl))
                cg_dict.update_maxl(maxl)

            self.cg_dict = cg_dict
            self._maxl = maxl

        # If cg_dict is not defined, but
        elif cg_dict is None and maxl is not None:

            self.cg_dict = CGDict(maxl=maxl, device=self.device, dtype=self.dtype)
            self._maxl = maxl

        else:
            self.cg_dict = None
            self._maxl = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def maxl(self):
        return self._maxl

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device, dtype=dtype)

        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return self

    def __getattr__(self, name):
        """
        PyTorch's nn.Module has a __getattr__/__setattr__ function that throws
        anerror if the user attemps to get/set an attribute that is not part
        of nn.Module
        """
        if name in super().__dict__.keys():
            return super().__getattr__(name)
        elif name in self.__dict__:
                return self.__dict__[name]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                        type(self).__name__, name))


    def __setattr__(self, name, value):
        """
        PyTorch's nn.Module has a __getattr__/__setattr__ function that throws
        anerror if the user attemps to get/set an attribute that is not part
        of nn.Module
        """
        if name in super().__dict__.keys():
            super().__setattr__(name, value)
        else:
            self.__dict__[name] = value

    def cuda(self, device=None):
        if device is None:
            device = torch.device('cuda')
        elif device in range(torch.cuda.device_count()):
            device = torch.device('cuda:{}'.format(device))
        else:
            ValueError('Incorrect choice of device!')

        super().cuda(device=device)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device)

        self._device = device

        return self

    def cpu(self):
        super().cpu()

        if self.cg_dict is not None:
            self.cg_dict.to(device=torch.device('cpu'))

        self._device = torch.device('cpu')

        return self

    def half(self):
        super().half()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.half)

        self._device = torch.half

        return self

    def float(self):
        super().float()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.float)

        self._device = torch.float

        return self

    def double(self):
        super().double()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.double)

        self._device = torch.double

        return self
