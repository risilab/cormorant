import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar, SO3Weight

from cormorant.so3_lib import mix

@pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
@pytest.mark.parametrize('maxl', range(3))
@pytest.mark.parametrize('channels1', range(1, 3))
@pytest.mark.parametrize('channels2', range(1, 3))
def test_mix_SO3Vec(batch, maxl, channels1, channels2):

    tau_in = [channels1]*(maxl+1)
    tau_out = [channels2]*(maxl+1)

    test_vec = SO3Vec.rand(batch, tau_in)
    test_weight = SO3Weight.rand(tau_in, tau_out)

    print(test_vec.shapes, test_weight.shapes)
    mix(test_weight, test_vec)

@pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
@pytest.mark.parametrize('maxl', range(3))
@pytest.mark.parametrize('channels1', range(1, 3))
@pytest.mark.parametrize('channels2', range(1, 3))
def test_mix_SO3Scalar(batch, maxl, channels1, channels2):

    tau_in = [channels1]*(maxl+1)
    tau_out = [channels2]*(maxl+1)

    test_scalar = SO3Scalar.rand(batch, tau_in)
    test_weight = SO3Weight.rand(tau_in, tau_out)

    print(test_scalar.shapes, test_weight.shapes)
    mix(test_weight, test_scalar)
