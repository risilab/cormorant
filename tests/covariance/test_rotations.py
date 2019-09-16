import torch
import pytest

from cormorant.so3_lib import so3_torch, SO3Tau, SO3Vec, SO3WignerD
from cormorant.so3_lib import rotations as rot
from cormorant.cg_lib import spherical_harmonics, CGDict
from utils import numpy_from_complex, complex_from_numpy


class TestRotations():

    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('channels', [1, 2])
    @pytest.mark.parametrize('batch', [[1], [2], (1, 1)])
    def test_apply_euler(self, batch, channels, maxl):
        tau = SO3Tau([channels] * (maxl+1))
        vec = SO3Vec.rand(batch, tau, dtype=torch.double)
        wigner = SO3WignerD.euler(maxl, dtype=torch.double)
        so3_torch.apply_wigner(vec, wigner)

    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('channels', range(1, 3))
    def test_spherical_harmonics(self, channels, maxl):
        # D, R, angles = rot.gen_rot(maxl, dtype=torch.double)
        import numpy as np
        D, R, angles = rot.gen_rot(maxl, angles=[np.pi/2, 0, 0], dtype=torch.double)
        D = SO3WignerD(D)
        D_numpy = np.round(numpy_from_complex(D[1]), decimals=8)

        # pos = torch.randn((channels, 3), dtype=torch.double)
        pos = torch.eye(3, dtype=torch.double)
        posr = rot.rotate_cart_vec(R, pos)

        cg_dict = CGDict(maxl, dtype=torch.double)

        sph_harms = spherical_harmonics(cg_dict, pos, maxl, sh_norm='qm')
        scipy_sph_harms_1 = sph_harms_from_scipy(pos, maxl)[1]
        print('original')
        print(sph_harms[1])
        print(sph_harms.channels, 'channels')
        print('Scipy:')
        print(scipy_sph_harms_1)
        print('-----')
        sph_harmsr = spherical_harmonics(cg_dict, posr, maxl, sh_norm='qm')
        scipy_sph_harms_1_r = sph_harms_from_scipy(posr, maxl)[1]
        print('rotated')
        print(sph_harmsr[1])
        print(scipy_sph_harms_1_r)

        sph_harmsd = so3_torch.apply_wigner(sph_harms, D)
        print('-----')
        print('rotated')
        print(np.round(numpy_from_complex(sph_harmsd), decimals=8))
        print('scipy')
        print(D_numpy.dot(scipy_sph_harms_1))

        # diff = (sph_harmsr - sph_harmsd).abs()
        diff = [(p1 - p2).abs().max() for p1, p2 in zip(sph_harmsr, sph_harmsd)]
        print(diff)
        assert all([d < 1e-6 for d in diff])


def sph_harms_from_scipy(pos, maxl):
    """ Calculate the spherical harmonics using SciPy's special function routine """
    import numpy as np
    from scipy.special import sph_harm as scipy_sph_harm
    s = pos.shape

    pos = pos.view(-1, 3)
    norm = pos.norm(dim=-1).unsqueeze(-1)

    x, y, z = (pos/norm).unbind(-1)
    phi = torch.atan2(y, x)
    theta = torch.acos(z)

    phi, theta = phi.view(-1, 1).numpy(), theta.view(-1, 1).numpy()

    sph_harms = []
    for l in range(maxl + 1):
        sph_harm_l = scipy_sph_harm(np.arange(-l, l+1).reshape(1, -1), l, phi, theta)
        sph_harms.append(sph_harm_l)
    return sph_harms
