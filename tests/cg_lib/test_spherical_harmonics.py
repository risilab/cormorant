import torch
import pytest
import scipy
import numpy as np

from cormorant.cg_lib import CGDict
from cormorant.cg_lib import spherical_harmonics, spherical_harmonics_rel
from utils import complex_from_numpy

# Test cg_product runs and aggregate=True works

class TestSphericalHarmonics():

    # Compare with SciyPy
    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('batch', [(1,), (2,), (5,), (1,1), (2,1), (1,2), (1, 1, 1), (2, 2, 2), (2, 3, 3)])
    def test_spherical_harmonics_vs_scipy(self, maxl, batch):
        cg_dict = CGDict(maxl=maxl, dtype=torch.double)

        pos = torch.rand(batch + (3,), dtype=torch.double)

        sh = spherical_harmonics(cg_dict, pos, maxl, normalize=True, sh_norm='qm')

        sh_sp = sph_harms_from_scipy(pos, maxl)

        for part1, part2 in zip(sh, sh_sp):
            assert torch.allclose(part1, part2)

    # Compare with SciyPy
    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('batch', [(1,), (2,), (5,), (1,1), (2,1), (1,2), (1, 1, 1), (2, 2, 2), (2, 3, 3)])
    @pytest.mark.parametrize('natoms1', [1, 2, 5])
    @pytest.mark.parametrize('natoms2', [1, 2, 5])
    def test_spherical_rel_harmonics_vs_scipy(self, maxl, batch, natoms1, natoms2):
        cg_dict = CGDict(maxl=maxl, dtype=torch.double)

        pos1 = torch.rand(batch + (natoms1, 3), dtype=torch.double)
        pos2 = torch.rand(batch + (natoms2, 3), dtype=torch.double)

        sh, norms = spherical_harmonics_rel(cg_dict, pos1, pos2, maxl, sh_norm='qm')

        sh_sp, norms_sp = sph_harms_rel_from_scipy(pos1, pos2, maxl)

        for l, (part1, part2) in enumerate(zip(sh, sh_sp)):
            if l == 0: continue
            assert torch.allclose(part1, part2)

        assert torch.allclose(norms, norms_sp)

def sph_harms_rel_from_scipy(pos1, pos2, maxl):
    """ Calculate the relative spherical harmonics using SciPy's special function reoutine """
    s1 = pos1.shape
    s2 = pos2.shape

    sbatch = s1[:-2]

    assert s1[:-2] == s2[:-2], 'Batch sizes unequal! {} {}'.format(s1, s2)

    pos1 = pos1.view((-1, s1[-2], 3))
    pos2 = pos2.view((-1, s2[-2], 3))

    s12 = (pos1.shape[0], pos1.shape[1], pos2.shape[1])

    norms = torch.zeros(s12, dtype=pos1.dtype)
    sph_harms = [torch.zeros(s12 + (1, 2*l+1, 2), dtype=pos1.dtype) for l in range(maxl + 1)]

    for bidx in range(s12[0]):
        for aidx1 in range(s12[1]):
            for aidx2 in range(s12[2]):
                vec1 = pos1[bidx, aidx1]
                vec2 = pos2[bidx, aidx2]
                if (vec1 == vec2).all(): continue
                sh_temp = sph_harms_from_scipy(vec1-vec2, maxl)
                norms[bidx, aidx1, aidx2] = (vec1-vec2).norm()
                for l in range(maxl + 1):
                    sph_harms[l][bidx, aidx1, aidx2, 0, :, :] = sh_temp[l]

    sph_harms = [part.view(sbatch + (s1[-2], s2[-2], 1, 2*l+1, 2)) for l, part in enumerate(sph_harms)]

    norms = norms.view(sbatch + (s1[-2], s2[-2]))

    return sph_harms, norms


def sph_harms_from_scipy(pos, maxl):
    """ Calculate the spherical harmonics using SciPy's special function reoutine """
    s = pos.shape

    pos = pos.view(-1, 3)
    norm = pos.norm(dim=-1).unsqueeze(-1)

    x, y, z = (pos/norm).unbind(-1)
    phi = torch.atan2(y, x)
    theta = torch.acos(z)

    phi, theta = phi.view(-1, 1).numpy(), theta.view(-1, 1).numpy()

    sph_harms = []
    for l in range(maxl + 1):
        sph_harm_l = scipy.special.sph_harm(np.arange(-l, l+1).reshape(1, -1), l, phi, theta)
        sph_harm_l = complex_from_numpy(sph_harm_l, dtype=torch.double).reshape(s[:-1] + (1, 2*l+1, 2,))

        sph_harms.append(sph_harm_l)

    return sph_harms
