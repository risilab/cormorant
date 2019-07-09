import torch
from torch.nn import Module
from math import sqrt, inf, pi

from . cg_ops import cg_product
from . import global_cg_dict

class SphericalHarmonics(Module):
    def __init__(self, max_sh, normalize=False, sh_norm='qm',
                 cg_dict=None, dtype=torch.float, device=torch.device('cpu')):
        super(SphericalHarmonics, self).__init__()

        self.max_sh = max_sh
        self.normalize = normalize

        self.device = device
        self.dtype = dtype

        if not cg_dict:
            self.cg_dict = global_cg_dict(max_sh, transpose=True, split=False, dtype=dtype, device=device)
        else:
            self.cg_dict = cg_dict
            assert(cg_dict.transpose==True and cg_dict.dtype == dtype and cg_dict.device == device)

        self.sh_norm = sh_norm

    def forward(self, pos):
        return spherical_harmonics(self.cg_dict, pos, self.max_sh, self.normalize, self.sh_norm)


class SphericalHarmonicsRel(Module):
    def __init__(self, max_sh, sh_norm='qm',
                 cg_dict=None, dtype=torch.float, device=torch.device('cpu')):

        super(SphericalHarmonicsRel, self).__init__()

        self.max_sh = max_sh

        self.device = device
        self.dtype = dtype

        if not cg_dict:
            self.cg_dict = global_cg_dict(max_sh, transpose=True, split=False, dtype=dtype, device=device)
        else:
            self.cg_dict = cg_dict
            assert(cg_dict.transpose==True and cg_dict.dtype == dtype and cg_dict.device == device)

        self.sh_norm = sh_norm

    def forward(self, pos1, pos2, mask=None):
        return spherical_harmonics_rel(self.cg_dict, pos1, pos2, self.max_sh, mask, self.sh_norm)


def spherical_harmonics(cg_dict, pos, maxsh, normalize=False, sh_norm='qm'):
    # global CGcoeffs, CGmaxL
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
        psi1 = pos_to_rep(pos)
        psi1 *= sqrt(3/(4*pi))
        sph_harms.append(psi1)

    if maxsh >= 2:
        new_psi = psi1
        for l in range(2, maxsh+1):
            new_psi = cg_product(cg_dict, [new_psi], [psi1], minl=0, maxl=l)[-1]
            ### Use equation Y^{m1}_{l1} \otimes Y^{m2}_{l2} = \sqrt((2*l1+1)(2*l2+1)/4*\pi*(2*l3+1)) <l1 0 l2 0|l3 0> <l1 m1 l2 m2|l3 m3> Y^{m3}_{l3}
            # cg_coeff = CGcoeffs[1*(CGmaxL+1) + l-1][5*(l-1)+1, 3*(l-1)+1] # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            cg_coeff = cg_dict[(1, l-1)][5*(l-1)+1, 3*(l-1)+1] # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            new_psi *= sqrt((4*pi*(2*l+1))/(3*(2*l-1))) / cg_coeff
            sph_harms.append(new_psi)
    sph_harms = [part.view(s + part.shape[1:]) for part in sph_harms]

    if sh_norm == 'qm':
        pass
    elif sh_norm == 'unit':
        sph_harms = [part*sqrt((4*pi)/(2*ell+1)) for ell, part in enumerate(sph_harms)]
    else:
        raise ValueError('Incorrect choice of spherial harmonic normalization!')

    return sph_harms


def spherical_harmonics_rel(cg_dict, pos1, pos2, maxsh, mask=None, sh_norm='qm'):
    rel_pos = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)
    rel_norms = rel_pos.norm(dim=-1, keepdim=True)
    # if mask:
    #     zeros = torch.zeros(1, dtype=rel_pos.dtype, device=rel_pos.device)
    #     rel_units = torch.where(mask, rel_pos / rel_norms, zeros)
    # else:
    #     rel_units = rel_pos / rel_norms
    #     rel_units[rel_pos == 0.] = 0

    rel_units = rel_pos

    rel_sph_harm = spherical_harmonics(cg_dict, rel_units, maxsh, normalize=True, sh_norm=sh_norm)

    return rel_sph_harm, rel_norms.squeeze(-1)

def pos_to_rep(pos):
    pos_x, pos_y, pos_z = pos.unbind(-1)

    pos_m = torch.stack([pos_x, -pos_y], -1)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], -1)
    pos_p = torch.stack([-pos_x, -pos_y], -1)/sqrt(2.)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=-2).unsqueeze(-3)

    return psi1

def rep_to_pos(rep):
    rep_m, rep_0, rep_p = rep.unbind(-2)

    pos_x = (-rep_p + rep_m)/sqrt(2.)
    pos_y = (-rep_p - rep_m)/sqrt(2.)
    pos_z = rep_0

    imag_part = [pos_x[..., 1].abs().mean(), pos_y[..., 0].abs().mean(), pos_z[..., 1].abs().mean()]
    assert(all([p < 1e-6 for p in imag_part])), 'Imaginary part not zero! {}'.format(imag_part)

    pos = torch.stack([pos_x[..., 0], pos_y[..., 1], pos_z[..., 0]], dim=-1)

    return pos
