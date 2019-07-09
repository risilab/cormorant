import torch
from torch.nn import Module, ModuleList, Parameter, ParameterList
from math import sqrt, inf, pi

from . import global_cg_dict


class CGProduct(Module):
    """
    Create new CGproduct object. Takes two lists of type
    [tau^1_{minl1}, tau^1_{minl1+1}, ..., tau^1_{maxl1}],
    [tau^2_{minl2}, tau^2_{minl2+1}, ..., tau^2_{maxl2}],
    and outputs a new SO3 vector of type:
    [tau_{minl}, tau_{minl+1}, ..., tau_{maxl}]
    Each part can have an arbitrary number of batch dimensions. These batch
    dimensions must be broadcastable, unless the option :aggregate=True: is used.

    If :aggregate=True: is used, a "matrix-vector" product is set, a "matrix-vector"
    product of Clebsch-Gordan operators is applied.

    :maxl: Maximum value of l in tensor product to include.
    :minl: Minimum value of l in tensor product to include.

    :aggregate: Perform batched matrix-vector Clebsch-Gordan operation.

    :dtype: Data type to used if :cg_dict: has not be initialized.
    :device: Device to used if :cg_dict: has not be initialized.
    """
    def __init__(self, tau1=None, tau2=None, maxl=inf, minl=0,
                 aggregate=False,
                 cg_dict=None, dtype=torch.float, device=torch.device('cpu')):
        super(CGProduct, self).__init__()

        if (minl > 0):
            raise NotImplementedError('minl > 0 not yet implemented!')

        self.maxl = maxl
        self.minl = minl
        self.aggregate = aggregate

        self.tau1 = tau1
        self.tau2 = tau2

        if not cg_dict:
            self.cg_dict = global_cg_dict(maxl, transpose=True, split=False, dtype=dtype, device=device)
        else:
            self.cg_dict = cg_dict
            assert(cg_dict.transpose==True and cg_dict.dtype == dtype and cg_dict.device == device)

    def forward(self, rep1, rep2):
        tau1 = [p.shape[-3] for p in rep1 if p.nelement() > 0]
        tau2 = [p.shape[-3] for p in rep2 if p.nelement() > 0]
        assert len(set(tau1).union(set(tau2))) == 1, 'The number of fragments must be same for each part! {} {}'.format(tau1, tau2)

        return cg_product(self.cg_dict, rep1, rep2, maxl=self.maxl, minl=self.minl, aggregate=self.aggregate)

    @property
    def tau_out(self):
        if not(self.tau1) or not(self.tau2):
            raise ValueError('Module not intialized with input type!')
        tau1 = [1 if t > 0 else 0 for t in self.tau1]
        tau2 = [1 if t > 0 else 0 for t in self.tau2]
        nchan = set([t for t in self.tau1+self.tau2 if t > 0]).pop()
        tau_out = cg_product_tau(tau1, tau2, maxl=self.maxl)
        return [nchan*t for t in tau_out]

    @property
    def tau1(self):
        try:
            return self._tau1
        except AttributeError:
            return None

    @tau1.setter
    def tau1(self, tau1):
        if tau1 is None:
            self._tau1 = None
        elif type(tau1) in [list, tuple]:
            self._tau1 = list(tau1)
        else:
            raise ValueError('Tau must be None, list, or tuple! {}'.format(tau1))
        self.check_taus()

    @property
    def tau2(self):
        try:
            return self._tau2
        except AttributeError:
            return None

    @tau2.setter
    def tau2(self, tau2):
        if tau2 is None:
            self._tau2 = None
        elif type(tau2) in [list, tuple]:
            self._tau2 = list(tau2)
        else:
            raise ValueError('Tau must be None, list, or tuple! {}'.format(tau2))
        self.check_taus()

    def check_taus(self):
        if self.tau2 and self.tau1:
            chan1 = set([t for t in self.tau1 if t > 0])
            chan2 = set([t for t in self.tau2 if t > 0])
            assert(chan1 == chan2 and len(chan1) == 1), 'Can only have single non-zero channel in tau1 and tau2! {} {}'.format(self.tau1, self.tau2)


def cg_product(cg_dict, rep1, rep2, maxl=inf, minl=0, aggregate=False):
    """
    Explicit function to calculate the Clebsch-Gordan product. See the documentation for CGProduct for more information.
    """
    assert(cg_dict.transpose), 'This uses transposed CG coefficients!'
    L1 = (rep1[-1].shape[-2] - 1)//2
    L2 = (rep2[-1].shape[-2] - 1)//2

    maxL = min(L1 + L2, maxl)

    new_rep = [[] for _ in range(maxL + 1)]

    for part1 in rep1:
        for part2 in rep2:
            l1, l2 = (part1.shape[-2] - 1)//2, (part2.shape[-2] - 1)//2
            lmin, lmax = max(abs(l1 - l2), minl), min(l1 + l2, maxL)
            if lmin > lmax: continue

            cg_mat = cg_dict[(l1, l2)][:(lmax+1)**2 - (lmin)**2, :]

            # Loop over atom irreps accumulating each.
            irrep_prod = complex_kron_product(part1, part2, aggregate=aggregate)
            cg_decomp = torch.matmul(cg_mat, irrep_prod)

            split = [2*l+1 for l in range(lmin, lmax+1)]
            cg_decomp = torch.split(cg_decomp, split, dim=-2)

            for idx, l in enumerate(range(lmin, lmax+1)):
                new_rep[l].append(cg_decomp[idx])

    new_rep = [torch.cat(part, dim=-3) for part in new_rep if len(part) > 0]

    return new_rep


def complex_kron_product(z1, z2, aggregate=False):
    """
    Take two complex matrix tensors z1 and z2, and take their tensor product.
    z1: batch1 x M1 x N1 x 2
    z2: batch2 x M2 x N2 x 2
    :aggregate: Apply aggregation/point-wise convolutional filter. Must have batch1 = B x A x A, batch2 = B x A
    :return: batch x (M1 x M2) x (N1 x N2) x 2
    """
    s1 = z1.shape
    s2 = z2.shape
    assert(len(s1) >= 3), 'Must have batch dimension!'
    assert(len(s2) >= 3), 'Must have batch dimension!'

    b1, b2 = s1[:-3], s2[:-3]
    s1, s2 = s1[-3:], s2[-3:]
    if not aggregate:
        assert(b1 == b2), 'Batch sizes must be equal! {} {}'.format(b1, b2)
        b = b1
    else:
        if (len(b1) == 3) and (len(b2) == 2):
            assert(b1[0] == b2[0]), 'Batch sizes must be equal! {} {}'.format(b1, b2)
            assert(b1[2] == b2[1]), 'Neighborhood sizes must be equal! {} {}'.format(b1, b2)

            z2 = z2.unsqueeze(1)
            b2 = z2.shape[:-3]
            b = b1

            agg_sum_dim = 2

        elif (len(b1) == 2) and (len(b2) == 3):
            assert(b2[0] == b1[0]), 'Batch sizes must be equal! {} {}'.format(b1, b2)
            assert(b2[2] == b1[1]), 'Neighborhood sizes must be equal! {} {}'.format(b1, b2)

            z1 = z1.unsqueeze(1)
            b1 = z1.shape[:-3]
            b = b2

            agg_sum_dim = 2

        else:
            raise ValueError('Batch size error! {} {}'.format(b1, b2))

    # Treat the channel index like a "batch index".
    assert(s1[0] == s2[0]), 'Number of channels must match! {} {}'.format(s1[0], s2[0])

    s12 = b + (s1[0], s1[1]*s2[1], s1[2]*s2[2])

    s10 = b1 + (s1[0],) + torch.Size([s1[1], 1, s1[2], 1])
    s20 = b2 + (s1[0],) + torch.Size([1, s2[1], 1, s2[2]])

    z = (z1.view(s10) * z2.view(s20))
    z = z.contiguous().view(s12)

    if aggregate:
        # Aggregation is sum over aggregation sum dimension defined above
        z = z.sum(agg_sum_dim, keepdim=False)

    zrot = torch.tensor([[1, 0], [0, 1], [0, 1], [-1, 0]], dtype=z.dtype, device=z.device)
    z = torch.matmul(z, zrot)

    return z

def cg_product_tau(tau1, tau2, maxl=inf):
    tau1 = list(tau1)
    tau2 = list(tau2)

    L1, L2 = len(tau1) - 1, len(tau2) - 1
    L = min(L1 + L2, maxl)

    tau = [0]*(L+1)

    for l1 in range(L1+1):
        for l2 in range(L2+1):
            lmin, lmax = abs(l1-l2), min(l1+l2, maxl)
            for l in range(lmin, lmax+1):
                tau[l] += tau1[l1]*tau2[l2]

    return tuple(tau)
