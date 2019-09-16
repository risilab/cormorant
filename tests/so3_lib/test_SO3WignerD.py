import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar, SO3WignerD
from cormorant.so3_lib import rotations as rot

class TestSO3WignerD():

    @pytest.mark.parametrize('maxl', range(3))
    def test_SO3WignerD_init(self, maxl):
        data = [torch.rand((2*l+1, 2*l+1, 2)) for l in range(maxl+1)]

        test_wigner_d = SO3WignerD(data)


    @pytest.mark.parametrize('maxl', range(3))
    def test_SO3WignerD_euler(self, maxl):

        test_wigner_d = SO3WignerD.euler(maxl)

    @pytest.mark.parametrize('maxl', range(3))
    def test_SO3WignerD_unitary(self, maxl):

        D = SO3WignerD.euler(maxl)

        Dp = SO3WignerD([rot.dagger(d) for d in D])

        for d1, d2 in zip(D, Dp):
            d1r, d1i = d1.unbind(-1)
            d2r, d2i = d2.unbind(-1)

            DDpr = torch.matmul(d1r, d2r) - torch.matmul(d1i, d2i)
            DDpi = torch.matmul(d1i, d2r) + torch.matmul(d1r, d2i)

            eye = torch.eye(DDpr.shape[0], dtype=DDpr.dtype)
            assert (DDpr - eye).abs().max() < 1e-6
            assert DDpi.abs().max() < 1e-6
