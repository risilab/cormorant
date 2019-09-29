import torch
from math import sqrt, pi

from cormorant.cg_lib import CGModule
from cormorant.cg_lib import cg_product

from cormorant.so3_lib import SO3Vec


class SphericalHarmonics(CGModule):
    r"""
    Calculate a list of spherical harmonics :math:`Y^\ell_m(\hat{\bf r})`
    for a :class:`torch.Tensor` of cartesian vectors :math:`{\bf r}`.

    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict`, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """
    def __init__(self, maxl, normalize=True, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos):
        r"""
        Calculate the Spherical Harmonics for a set of cartesian position vectors.

        Parameters
        ----------
        pos : :class:`torch.Tensor`
            Input tensor of cartesian vectors

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output list of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        return spherical_harmonics(self.cg_dict, pos, self.maxl,
                                   self.normalize, self.conj, self.sh_norm)


class SphericalHarmonicsRel(CGModule):
    r"""
    Calculate a matrix of spherical harmonics

    .. math::
        \Upsilon_{ij} = Y^\ell_m(\hat{\bf r}_{ij})

    based upon the difference

    .. math::
        {\bf r}_{ij} = {\bf r}^{(1)}_i - {\bf r}^{(2)}_j.

    in two lists of cartesian vectors  :math:`{\bf r}^{(1)}_i`
    and :math:`{\bf r}^{(2)}_j`.


    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict` or None, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """
    def __init__(self, maxl, normalize=False, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos1, pos2):
        r"""
        Calculate the Spherical Harmonics for a matrix of differences of cartesian
        position vectors `pos1` and `pos2`.

        Note that `pos1` and `pos2` must agree in every dimension except for
        the second-to-last one.

        Parameters
        ----------
        pos1 : :class:`torch.Tensor`
            First tensor of cartesian vectors :math:`{\bf r}^{(1)}_i`.
        pos2 : :class:`torch.Tensor`
            Second tensor of cartesian vectors :math:`{\bf r}^{(2)}_j`.

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output matrix of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        return spherical_harmonics_rel(self.cg_dict, pos1, pos2, self.maxl,
                                       self.normalize, self.conj, self.sh_norm)


def spherical_harmonics(cg_dict, pos, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the Spherical Harmonics. See documentation of
    :class:`SphericalHarmonics` for details.
    """
    s = pos.shape[:-1]

    pos = pos.view(-1, 3)

    if normalize:
        norm = pos.norm(dim=-1, keepdim=True)
        mask = (norm > 0)
        # pos /= norm
        # pos[pos == inf] = 0
        pos = torch.where(mask, pos / norm, torch.zeros_like(pos))

    psi0 = torch.full(s + (1,), sqrt(1/(4*pi)), dtype=pos.dtype, device=pos.device)
    psi0 = torch.stack([psi0, torch.zeros_like(psi0)], -1)
    psi0 = psi0.view(-1, 1, 1, 2)

    sph_harms = [psi0]
    if maxsh >= 1:
        psi1 = pos_to_rep(pos, conj=conj)
        psi1 *= sqrt(3/(4*pi))
        sph_harms.append(psi1)

    if maxsh >= 2:
        new_psi = psi1
        for l in range(2, maxsh+1):
            new_psi = cg_product(cg_dict, [new_psi], [psi1], minl=0, maxl=l, ignore_check=True)[-1]
            # Use equation Y^{m1}_{l1} \otimes Y^{m2}_{l2} = \sqrt((2*l1+1)(2*l2+1)/4*\pi*(2*l3+1)) <l1 0 l2 0|l3 0> <l1 m1 l2 m2|l3 m3> Y^{m3}_{l3}
            # cg_coeff = CGcoeffs[1*(CGmaxL+1) + l-1][5*(l-1)+1, 3*(l-1)+1] # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            cg_coeff = cg_dict[(1, l-1)][5*(l-1)+1, 3*(l-1)+1]  # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            new_psi *= sqrt((4*pi*(2*l+1))/(3*(2*l-1))) / cg_coeff
            sph_harms.append(new_psi)
    sph_harms = [part.view(s + part.shape[1:]) for part in sph_harms]

    if sh_norm == 'qm':
        pass
    elif sh_norm == 'unit':
        sph_harms = [part*sqrt((4*pi)/(2*ell+1)) for ell, part in enumerate(sph_harms)]
    else:
        raise ValueError('Incorrect choice of spherial harmonic normalization!')

    return SO3Vec(sph_harms)


def spherical_harmonics_rel(cg_dict, pos1, pos2, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the relative Spherical Harmonics. See documentation of
    :class:`SphericalHarmonicsRel` for details.
    """
    rel_pos = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)
    rel_norms = rel_pos.norm(dim=-1, keepdim=True)

    rel_sph_harm = spherical_harmonics(cg_dict, rel_pos, maxsh, normalize=normalize,
                                       conj=conj, sh_norm=sh_norm)

    return rel_sph_harm, rel_norms.squeeze(-1)


def pos_to_rep(pos, conj=False):
    r"""
    Convert a tensor of cartesian position vectors to an l=1 spherical tensor.

    Parameters
    ----------
    pos : :class:`torch.Tensor`
        A set of input cartesian vectors. Can have arbitrary batch dimensions
         as long as the last dimension has length three, for x, y, z.
    conj : :class:`bool`, optional
        Return the complex conjugated representation.


    Returns
    -------
    psi1 : :class:`torch.Tensor`
        The input cartesian vectors converted to a l=1 spherical tensor.

    """
    pos_x, pos_y, pos_z = pos.unbind(-1)

    # Only the y coordinates get mapped to imaginary terms
    if conj:
        pos_y *= -1

    pos_m = torch.stack([pos_x, -pos_y], -1)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], -1)
    pos_p = torch.stack([-pos_x, -pos_y], -1)/sqrt(2.)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=-2).unsqueeze(-3)

    return psi1


def rep_to_pos(rep):
    r"""
    Convert a tensor of l=1 spherical tensors to cartesian position vectors.

    Warning
    -------
    The input spherical tensor must satisfy :math:`F_{-m} = (-1)^m F_{m}^*`,
    so the output cartesian tensor is explicitly real. If this is not satisfied
    an error will be thrown.

    Parameters
    ----------
    rep : :class:`torch.Tensor`
        A set of input l=1 spherical tensors.
        Can have arbitrary batch dimensions as long
        as the last dimension has length three, for m = -1, 0, +1.

    Returns
    -------
    pos : :class:`torch.Tensor`
        The input l=1 spherical tensors converted to cartesian vectors.

    """
    rep_m, rep_0, rep_p = rep.unbind(-2)

    pos_x = (-rep_p + rep_m)/sqrt(2.)
    pos_y = (-rep_p - rep_m)/sqrt(2.)
    pos_z = rep_0

    imag_part = [pos_x[..., 1].abs().mean(), pos_y[..., 0].abs().mean(), pos_z[..., 1].abs().mean()]
    if (any([p > 1e-6 for p in imag_part])):
        raise ValueError('Imaginary part not zero! {}'.format(imag_part))

    pos = torch.stack([pos_x[..., 0], pos_y[..., 1], pos_z[..., 0]], dim=-1)

    return pos
