import torch
import pytest

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar

rand_vec = lambda batch, tau: SO3Vec([torch.rand(batch + (2*l+1, t, 2)) for l, t in enumerate(tau)])

class TestSO3Vec():

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('channels', range(1, 3))
    def test_SO3Vec_init_channels(self, batch, maxl, channels):

        tau_list = [channels]*(maxl+1)

        test_vec = rand_vec(batch, tau_list)

        assert test_vec.tau == tau_list

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_init_arb_tau(self, batch, maxl, channels):

        tau_list = torch.randint(1, channels+1, [maxl+1])

        test_vec = rand_vec(batch, tau_list)

        assert test_vec.tau == tau_list


    @pytest.mark.parametrize('batch', [[(1,), (2,)], [(1,), (1, 2)]])
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_check_batch_fail(self, batch, channels):

        maxl = len(batch)

        tau = torch.randint(1, channels+1, [maxl+1])

        rand_vec = [torch.rand(b + (2*l+1, t, 2)) for l, (b, t) in enumerate(zip(batch, tau))]

        if len(set(batch)) == 1:
            SO3Vec(rand_vec)
        else:
            with pytest.raises(ValueError) as e:
                SO3Vec(rand_vec)

    @pytest.mark.parametrize('batch', [[(1,), (2,)], [(1,), (1, 2)], [(1, 1), (2, 2,)], [(1, 1), (1, 1)]])
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_check_batch_fail(self, batch, channels):

        maxl = len(batch) - 1

        tau = torch.randint(1, channels+1, [maxl+1])

        rand_vec = [torch.rand(b + (2*l+1, t, 2)) for l, (b, t) in enumerate(zip(batch, tau))]

        if len(set(batch)) == 1:
            SO3Vec(rand_vec)
        else:
            with pytest.raises(ValueError) as e:
                SO3Vec(rand_vec)

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_check_rep_fail(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)
        rand_vec = [torch.rand(batch + (2*l+2, t, 2)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Vec(rand_vec)

        rand_vec = [torch.rand(batch + (1, t, 2)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Vec(rand_vec)

        rand_vec = [torch.rand(batch + (t, t, 2)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Vec(rand_vec)

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_check_cplx_fail(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)
        rand_vec = [torch.rand(batch + (2*l+1, t, 1)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Vec(rand_vec)

        rand_vec = [torch.rand(batch + (2*l+1, t, 3)) for l, t in enumerate(tau)]

        with pytest.raises(ValueError) as e:
            SO3Vec(rand_vec)


    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_mul_scalar(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Vec([torch.rand(batch + (2*l+1, t, 2)) for l, t in enumerate(tau)])

        vec1 = 2 * vec0
        assert all(torch.allclose(2*part0, part1) for part0, part1 in zip(vec0, vec1))

        vec1 = vec0 * 2.0
        assert all(torch.allclose(2*part0, part1) for part0, part1 in zip(vec0, vec1))

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_mul_list(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Vec([torch.rand(batch + (2*l+1, t, 2)) for l, t in enumerate(tau)])

        scalar = [torch.rand(1).item() for _ in vec0]

        vec1 = scalar * vec0
        assert all(torch.allclose(s*part0, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

        vec1 = vec0 * scalar
        assert all(torch.allclose(part0*s, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

    @pytest.mark.parametrize('batch', [(1,), (2,), (7,), (1,1), (2, 2), (7, 7)])
    @pytest.mark.parametrize('maxl', range(1, 4))
    @pytest.mark.parametrize('channels', range(1, 4))
    def test_SO3Vec_add_list(self, batch, maxl, channels):
        tau = [channels] * (maxl+1)

        vec0 = SO3Vec([torch.rand(batch + (2*l+1, t, 2)) for l, t in enumerate(tau)])

        scalar = [torch.rand(1).item() for _ in vec0]

        vec1 = scalar + vec0
        assert all(torch.allclose(s + part0, part1) for part0, s, part1 in zip(vec0, scalar, vec1))

        vec1 = vec0 + scalar
        assert all(torch.allclose(part0 + s, part1) for part0, s, part1 in zip(vec0, scalar, vec1))
