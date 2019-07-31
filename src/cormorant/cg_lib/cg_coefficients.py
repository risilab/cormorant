import torch

from .gen_cg import gen_cg_coefffs

import logging
logger = logging.getLogger(__name__)

class CGDict():
    def __init__(self):
        self.cg_dict = {}
        self.initialized = False
        self.transpose = True
        self.split = False
        self.maxl = -1

    def __call__(self, maxl, split=False, transpose=True, dtype=torch.float, device=torch.device('cpu')):
        if self.initialized:
            if not (self.transpose == transpose and (self.split or not split) and self.dtype == dtype and self.device == device):
                logger.warning('WARNING: CGDict options do not match. Initializing new CG dict. Could result in duplicate CG dicts in memory.')
                logger.warning('({} {}) ({} {}) ({} {})'.format(self.transpose, transpose, self.dtype, dtype, self.device, device))
                new_cg_dict = CGDict()
                return new_cg_dict(maxl, split=split, transpose=transpose, dtype=dtype, device=device)

            if self.maxl >= maxl:
                return self
            else:
                self.initialized = False
                self.initialize(maxl, split=split, transpose=transpose, dtype=dtype, device=device)

        cg_mats = gen_cg_coefffs(maxl)
        cg_dict = {((cg_mat.shape[0]-1)//2, (cg_mat.shape[1]-1)//2): cg_mat.view(-1, cg_mat.shape[2]) for cg_mat in cg_mats}

        if split:
            cg_split = {}
            for key, val in cg_dict.items():
                (l1, l2) = key
                split_ = [2*l+1 for l in range(abs(l1-l2), l1+l2+1)]
                cg_split.update({key+(l,): cg_mat.view((2*l1+1)*(2*l2+1), 2*l+1) for (l, cg_mat) in zip(range(abs(l1-l2), l1+l2+1), val.split(split_, dim=-1))})

            cg_dict.update(cg_split)

        if transpose:
            cg_dict = {key: cg_mat.t() for (key, cg_mat) in cg_dict.items()}

        self.cg_dict = {key: cg_mat.to(dtype=dtype, device=device) for (key, cg_mat) in cg_dict.items()}

        self.cg_maxl = maxl
        self.maxl = maxl
        self.initialized = True
        self.transpose = transpose
        self.split = split

        self.dtype = dtype
        self.device = device

        return self

    initialize = __call__

    def to(self, dtype=None, device=None):
        if dtype is None and device is None:
            pass
        elif dtype is None and device is not None:
            self.cg_dict = {key: val.to(device=device) for key, valu in cg_dict.items()}
        elif dtype is not None and device is None:
            self.cg_dict = {key: val.to(dtype=dtype) for key, valu in cg_dict.items()}
        elif dtype is not None and device is not None:
            self.cg_dict = {key: val.to(device=device, dtype=dtype) for key, valu in cg_dict.items()}
        return self

    def keys(self):
        return self.cg_dict.keys()

    def values(self):
        return self.cg_dict.values()

    def items(self):
        return self.cg_dict.items()

    def __getitem__(self, idx):
        assert(self.initialized), 'Must initialize CG coefficients first!'
        return self.cg_dict[idx]

    def __bool__(self):
        return self.initialized
