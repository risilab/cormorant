import torch
from torch.nn import Module, ModuleList, Parameter, ParameterList
from math import sqrt, inf, pi

from . import CGModule

from . import SO3Tau, cg_product_tau

class CGProduct(CGModule):
    r"""
    Create new CGproduct object. Inherits from CGModule, and has access
    to the CGDict related features.

    Takes two lists of type

    .. math::

        [tau^1_{minl2}, tau^2_{minl2+1}, ..., tau^2_{maxl2}],

        [tau^2_{minl2}, tau^2_{minl2+1}, ..., tau^2_{maxl2}],
    and outputs a new SO3 vector of type:

    .. math::

        [tau_{minl}, tau_{minl+1}, ..., tau_{maxl}]

    Each part can have an arbitrary number of batch dimensions. These batch
    dimensions must be broadcastable, unless the option :aggregate=True: is used.

    Parameters
    ----------
    rep1 : list of torch.Tensors
        SO3Vector
    rep2 : list of torch.Tensors
        SO3Vector

    minl : int
        Minimum weight to include in CG Product
    maxl : int
        Minimum weight to include in CG Product
    aggregate : bool, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a SO3Vector as a filter.
    cg_dict : CGDict, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxl.
    device : torch.device, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : torch.dtype, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.

    """
    def __init__(self, tau1=None, tau2=None,
                 aggregate=False,
                 minl=0, maxl=inf, cg_dict=None, dtype=None, device=None):

        self.aggregate = aggregate

        if (maxl == inf) and cg_dict:
            maxl = cg_dict.maxl
        elif (maxl == inf) and (tau1 and tau2):
            maxl = max(len(tau1), len(tau2))
        elif (maxl == inf):
            raise ValueError('maxl is not defined, and was unable to retrieve get maxl from cg_dict or tau1 and tau2')

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

        self.set_taus(tau1, tau2)

        if (minl > 0):
            raise NotImplementedError('minl > 0 not yet implemented!')
        else:
            self.minl = 0

    def forward(self, rep1, rep2):
        if self.tau1 is not None and self.tau1 != SO3Tau.from_rep(rep1):
            raise ValueError('Input rep1 does not have predefined tau!')

        if self.tau2 is not None and self.tau2 != SO3Tau.from_rep(rep2):
            raise ValueError('Input rep2 does not have predefined tau!')

        return cg_product(self.cg_dict, rep1, rep2, maxl=self.maxl, minl=self.minl, aggregate=self.aggregate)

    @property
    def tau_out(self):
        if not(self.tau1) or not(self.tau2):
            raise ValueError('Module not intialized with input type!')
        tau_out = cg_product_tau(self.tau1, self.tau2, maxl=self.maxl)
        return tau_out

    @property
    def tau1(self):
        return self._tau1

    @property
    def tau2(self):
        return self._tau2

    def set_taus(self, tau1=None, tau2=None):
        self._tau1 = SO3Tau(tau1) if tau1 else None
        self._tau2 = SO3Tau(tau2) if tau2 else None

        if self._tau1 and self._tau2:
            if not self.tau1.channels or (self.tau1.channels != self.tau2.channels):
                raise ValueError('The number of fragments must be same for each part! '
                                 '{} {}'.format(self.tau1, self.tau2))



def cg_product(cg_dict, rep1, rep2, maxl=inf, minl=0, aggregate=False):
    """
    Explicit function to calculate the Clebsch-Gordan product.
    See the documentation for CGProduct for more information.

    rep1 : list of torch.Tensors
        First SO3Vector in the CG product
    rep2 : list of torch.Tensors
        First SO3Vector in the CG product
    minl : int
        Minimum weight to include in CG Product
    maxl : int
        Minimum weight to include in CG Product
    aggregate : bool, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a SO3Vector as a filter.
    cg_dict : CGDict, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxl.
    """
    tau1 = SO3Tau.from_rep(rep1)
    tau2 = SO3Tau.from_rep(rep2)
    assert tau1.channels and (tau1.channels == tau2.channels), 'The number of fragments must be same for each part! {} {}'.format(tau1, tau2)

    ells1 = [(part.shape[-2] - 1)//2 for part in rep1]
    ells2 = [(part.shape[-2] - 1)//2 for part in rep2]

    L1 = max(ells1)
    L2 = max(ells2)

    if (cg_dict.maxl < maxl) or (cg_dict.maxl < L1) or (cg_dict.maxl < L2):
        raise ValueError('CG Dictionary maxl ({}) not sufficiently large for (maxl, L1, L2) = ({} {} {})'.format(cg_dict.maxl, maxl, L1, L2))
    assert(cg_dict.transpose), 'This operation uses transposed CG coefficients!'

    maxL = min(L1 + L2, maxl)

    new_rep = [[] for _ in range(maxL + 1)]

    for l1, part1 in zip(ells1, rep1):
        for l2, part2 in zip(ells2, rep2):
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
