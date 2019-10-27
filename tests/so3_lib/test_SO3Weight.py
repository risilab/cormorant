import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Weight, SO3Vec, SO3Scalar

rand_vec = lambda batch, tau: SO3Vec([torch.rand(batch + (2*l+1, t, 2)) for l, t in enumerate(tau)])
rand_weight_list = lambda tau1, tau2: [torch.rand((t2, t1, 2)) for t1, t2 in zip(tau1, tau2)]
rand_weight = lambda tau1, tau2: SO3Weight(rand_weight_list(tau1, tau2))

class TestSO3Weight():

    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('channels1', range(1, 3))
    @pytest.mark.parametrize('channels2', range(1, 3))
    def test_SO3Weight_init(self, maxl, channels1, channels2):

        tau1_list = SO3Tau([channels1]*(maxl+1))
        tau2_list = SO3Tau([channels2]*(maxl+1))

        weight_list = rand_weight_list(tau1_list, tau2_list)

        weight = SO3Weight(weight_list)

        assert isinstance(weight, SO3Weight)
        assert weight.tau_in == SO3Tau(tau1_list)
        assert weight.tau_out == SO3Tau(tau2_list)
