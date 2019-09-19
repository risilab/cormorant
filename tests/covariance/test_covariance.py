import torch
import pytest

from cormorant.so3_lib import so3_torch, SO3Tau
from cormorant.so3_lib import SO3Vec, SO3Scalar, SO3Weight, SO3WignerD
from cormorant.so3_lib import rotations as rot

from cormorant.cg_lib import CGProduct, CGDict

from ..helper_utils.utils import get_dataloader

def gen_rotated_data(dataloader, maxl):
    D, R = rot.gen_rot(maxl)

    data_rotout = data

    data_rotin = {key: val.clone() if type(val) is torch.Tensor else None for key, val in data.items()}
    data_rotin['positions'] = rot.rotate_cart_vec(R, data_rotin['positions'])

    return data_rotin, data_rotout

class TestCovariance():

    @pytest.mark.parametrize('batch', [(1,), (3,), (1, 1), (3, 3)])
    @pytest.mark.parametrize('maxl1', range(4))
    @pytest.mark.parametrize('maxl2', range(4))
    @pytest.mark.parametrize('maxl', range(4))
    @pytest.mark.parametrize('channels', [1, 5])
    def test_CGProduct(self, batch, maxl1, maxl2, maxl, channels):
        maxl_all = max(maxl1, maxl2, maxl)
        D, R, _ = rot.gen_rot(maxl_all)

        cg_dict = CGDict(maxl=maxl_all, dtype=torch.double)
        cg_prod = CGProduct(maxl=maxl, dtype=torch.double, cg_dict=cg_dict)

        tau1 = SO3Tau([channels] * (maxl1+1))
        tau2 = SO3Tau([channels] * (maxl2+1))

        vec1 = SO3Vec.randn(tau1, batch, dtype=torch.double)
        vec2 = SO3Vec.randn(tau2, batch, dtype=torch.double)

        vec1i = vec1.apply_wigner(D, dir='left')
        vec2i = vec2.apply_wigner(D, dir='left')

        vec_prod = cg_prod(vec1, vec2)
        veci_prod = cg_prod(vec1i, vec2i)

        vecf_prod = vec_prod.apply_wigner(D, dir='left')

        # diff = (sph_harmsr - sph_harmsd).abs()
        diff = [(p1 - p2).abs().max() for p1, p2 in zip(veci_prod, vecf_prod)]
        print(diff)
        assert all([d < 1e-6 for d in diff])


        # assert (veci_prod - vecf_prod).allclose()
