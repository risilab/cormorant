import torch
import pytest

from cormorant.so3_lib import so3_torch, SO3Vec, SO3Scalar, SO3Weight
from cormorant.so3_lib import rotations as rot

from cormorant.cg_lib import CGProduct

from ..utils import get_dataloader

def gen_rotated_data(dataloader, maxl):
    D, R = rot.gen_rot(maxl)

    data_rotout = data

    data_rotin = {key: val.clone() if type(val) is torch.Tensor else None for key, val in data.items()}
    data_rotin['positions'] = rot.rotate_cart_vec(R, data_rotin['positions'])

    return data_rotin, data_rotout

class TestCovariance():

    def test_CGProduct(self, maxl1, maxl2, maxl, channels1, channels2):
        D, R = rot.gen_rot(maxl)

        wigner = SO3WignerD(D)

        cg_dict = CGDict(maxl=maxl, dtype=torch.double)
        cg_prod = CGProduct(maxl=maxl, dtype=torch.double, cg_dict=cg_dict)

        tau1 = SO3Tau([channels1] * (maxl1+1))
        tau2 = SO3Tau([channels2] * (maxl2+1))

        vec1 = SO3Vec.randn(tau1)
        vec2 = SO3Vec.randn(tau2)

        vec1r = vec1.apply_wigner(D)
        vec2r = vec2.apply_wigner(D)

        vec_prod = cg_prod(vec1, vec2)
        vecr_prod = cg_prod(vec1r, vec2r)
