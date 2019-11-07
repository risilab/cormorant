import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Scalar, SO3Scalar

rand_scalar = lambda batch, tau: SO3Scalar([torch.rand(batch + (t, 2)) for l, t in enumerate(tau)])

class TestSO3Scalar():

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', [0, 2])
    @pytest.mark.parametrize('channels', range(1, 3))
    def test_SO3Scalar_init_channels(self, batch, maxl, channels):

        tau_list = [channels]*(maxl+1)

        test_vec = SO3Scalar.rand(batch, tau_list)

        assert test_vec.tau == tau_list

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', [0, 2])
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_init_arb_tau(self, batch, maxl, channels):

        tau_list = torch.randint(1, channels+1, [maxl+1])

        test_vec = SO3Scalar.rand(batch, tau_list)

        assert test_vec.tau == tau_list


    @pytest.mark.parametrize('batch', [[(1,), (2,)], [(1,), (1, 2)]])
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_check_batch_fail(self, batch, channels):

        maxl = len(batch)

        tau = torch.randint(1, channels+1, [maxl+1])

        rand_scalar = [torch.rand(b + (t, 2)) for l, (b, t) in enumerate(zip(batch, tau))]

        if len(set(batch)) == 1:
            SO3Scalar(rand_scalar)
        else:
            with pytest.raises(ValueError) as e:
                SO3Scalar(rand_scalar)

    @pytest.mark.parametrize('batch', [[(1,), (2,)], [(1,), (1, 2)], [(1, 1), (2, 2,)], [(1, 1), (1, 1)]])
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_check_batch_fail(self, batch, channels):

        maxl = len(batch) - 1

        tau = torch.randint(1, channels+1, [maxl+1])

        rand_scalar = [torch.rand(b + (t, 2)) for l, (b, t) in enumerate(zip(batch, tau))]

        if len(set(batch)) == 1:
            SO3Scalar(rand_scalar)
        else:
            with pytest.raises(ValueError) as e:
                SO3Scalar(rand_scalar)


    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_check_cplx_fail(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)
        rand_scalar = [torch.rand(batch + (t, 1, 1)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Scalar(rand_scalar)

        rand_scalar = [torch.rand(batch + (t, 1, 3)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Scalar(rand_scalar)


    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_mul_scalar(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Scalar([torch.rand(batch + (t, 2)) for l, t in enumerate(tau)])

        vec1 = 2 * vec0
        assert all(torch.allclose(2*part0, part1) for part0, part1 in zip(vec0, vec1))

        vec1 = vec0 * 2.0
        assert all(torch.allclose(2*part0, part1) for part0, part1 in zip(vec0, vec1))

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_mul_list(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Scalar([torch.rand(batch + (t, 2)) for l, t in enumerate(tau)])

        scalar = [torch.rand(1).item() for _ in vec0]

        vec1 = scalar * vec0
        assert all(torch.allclose(s*part0, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

        vec1 = vec0 * scalar
        assert all(torch.allclose(part0*s, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Scalar_add_list(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Scalar([torch.rand(batch + (t, 2)) for l, t in enumerate(tau)])

        scalar = [torch.rand(1).item() for _ in vec0]

        vec1 = scalar + vec0
        assert all(torch.allclose(s + part0, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

        vec1 = vec0 + scalar
        assert all(torch.allclose(part0 + s, part1) for part0, s, part1 in zip(vec0, scalar, vec1))
