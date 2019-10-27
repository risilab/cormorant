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
        so3_torch.apply_wigner(wigner, vec)

    @pytest.mark.parametrize('maxl', range(4))
    @pytest.mark.parametrize('channels', range(1, 3))
    @pytest.mark.parametrize('conj', [True, False])
    def test_spherical_harmonics(self, maxl, channels, conj):
        D, R, angles = rot.gen_rot(maxl, dtype=torch.double)
        D = SO3WignerD(D)

        if not conj:
            R = R.t()

        pos = torch.randn((channels, 3), dtype=torch.double)
        posr = rot.rotate_cart_vec(R, pos)

        cg_dict = CGDict(maxl, dtype=torch.double)

        sph_harms = spherical_harmonics(cg_dict, pos, maxl, conj=conj)
        sph_harmsr = spherical_harmonics(cg_dict, posr, maxl, conj=conj)

        dir = 'left' if conj else 'right'

        sph_harmsd = so3_torch.apply_wigner(D, sph_harms, dir=dir)

        # diff = (sph_harmsr - sph_harmsd).abs()
        diff = [(p1 - p2).abs().max() for p1, p2 in zip(sph_harmsr, sph_harmsd)]
        print(diff)
        assert all([d < 1e-6 for d in diff])
