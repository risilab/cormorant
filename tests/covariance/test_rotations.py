import torch
import pytest

from cormorant.so3_lib import so3_torch, SO3Tau, SO3Vec, SO3WignerD
from cormorant.so3_lib import rotations as rot

from cormorant.cg_lib import spherical_harmonics, CGDict


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
    @pytest.mark.parametrize('channels', range(3))
    def test_spherical_harmonics(self, channels, maxl):
        D, R, angles = rot.gen_rot(maxl)

        pos = torch.tensor((channels, 3), dtype=torch.double)
        posr = rot.rotate_cart_vec(pos, R)

        cg_dict = CGDict(maxl, dtype=torch.double)

        sph_harms = spherical_harmonics(cg_dict, pos, maxl)
        sph_harmsr = spherical_harmonics(cg_dict, posr, maxl)

        sph_harmsd = so3_torch.apply_wigner(sph_harms, D)
