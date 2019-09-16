import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar, SO3WignerD
from cormorant.so3_lib import rotations as rot

class TestSO3WignerD():

    @pytest.mark.parametrize('maxl', range(3))
    def test_SO3WignerD_euler(self, maxl):
        data = [torch.rand((2*l+1, 2*l+1, 2) for l in range(maxl+1))]

        test_wigner_d = SO3WignerD(data)


    @pytest.mark.parametrize('maxl', range(3))
    def test_SO3WignerD_euler(self, maxl):

        test_wigner_d = SO3WignerD.euler(maxl)
