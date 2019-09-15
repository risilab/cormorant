import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar, so3_torch

class TestSO3Torch():

    @pytest.mark.parametrize('batch1', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('batch2', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('batch3', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('maxl1', [0, 2, 3])
    @pytest.mark.parametrize('maxl2', [2, 3])
    @pytest.mark.parametrize('maxl3', [2, 3])
    @pytest.mark.parametrize('channels1', [1, 2])
    @pytest.mark.parametrize('channels2', [1, 2])
    @pytest.mark.parametrize('channels3', [1, 2])
    def test_SO3Vec_cat(self, batch1, batch2, batch3, channels1, channels2, channels3, maxl1, maxl2, maxl3):
        tau1 = [channels1] * (maxl1+1)
        tau2 = [channels2] * (maxl2+1)
        tau3 = [channels3] * (maxl2+1)

        tau12 = SO3Tau.cat([tau1, tau2])
        tau123 = SO3Tau.cat([tau1, tau2, tau3])

        vec1 = SO3Vec.randn(tau1, batch1)
        vec2 = SO3Vec.randn(tau2, batch2)
        vec3 = SO3Vec.randn(tau3, batch3)

        if batch1 == batch2:
            vec12 = so3_torch.cat([vec1, vec2])

            assert vec12.tau == tau12
        else:
            with pytest.raises(RuntimeError):
                vec12 = so3_torch.cat([vec1, vec2])

        if batch1 == batch2 == batch3:
            vec123 = so3_torch.cat([vec1, vec2, vec3])

            assert vec123.tau == tau123
        else:
            with pytest.raises(RuntimeError):
                vec12 = so3_torch.cat([vec1, vec2, vec3])
