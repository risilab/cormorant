import torch
import pytest

from cormorant.cg_lib import CGModule, CGDict
from cormorant.cg_lib import SO3Tau

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append([torch.device('cuda')])

class TestSO3Tau():

    # Test initialization
    @pytest.mark.parametrize('maxl', [0, 1, 2])
    @pytest.mark.parametrize('num_channels', [0, 1, 2])
    def test_init(self, maxl, num_channels):
        tau = SO3Tau([num_channels]*(maxl+1))

        assert type(tau) == SO3Tau
        assert list(tau) == [num_channels]*(maxl+1)

        tau = SO3Tau(tau)

        assert type(tau) == SO3Tau
        assert list(tau) == [num_channels]*(maxl+1)

    # Test Concatenation
    def test_cat(self):
        tau1 = SO3Tau([1, 2, 3])
        tau2 = SO3Tau([1, 1])
        tau3 = SO3Tau([0, 0, 2])

        tau = SO3Tau.cat([tau1, tau2])
        assert list(tau) == [2, 3, 3]

        assert type(tau) == SO3Tau

        print(tau)

        tau = (tau1 & tau2)
        assert list(tau) == [2, 3, 3]

        tau1 &= tau2
        assert list(tau1) == [2, 3, 3]

        tau123 = (tau1 & tau2) & tau3

        assert SO3Tau.cat([tau1, tau2, tau3]) == tau123

    # Test list-like addition
    def test_add(self):
        tau1 = SO3Tau([1, 2, 3])
        tau2 = SO3Tau([1, 1])

        tau = tau1 + tau2
        assert type(tau) == SO3Tau
        assert list(tau) == list(tau1) + list(tau2)

        tau = tuple(tau1) + tau2
        assert type(tau) == SO3Tau
        assert list(tau) == list(tau1) + list(tau2)

        tau = list(tau1) + tau2
        assert type(tau) == SO3Tau
        assert list(tau) == list(tau1) + list(tau2)

        tau1p = SO3Tau(tau1)
        tau1p += tau2
        assert type(tau1p) == SO3Tau
        assert list(tau1p) == list(tau1) + list(tau2)

        tau = sum([SO3Tau([3, 2, 1]), [1], (2, 3)])
        assert type(tau) == SO3Tau
        assert list(tau) == [3, 2, 1, 1, 2, 3]

    def test_channels(self):
        tau = SO3Tau([3, 2, 1])

        assert tau.channels == None

        tau = SO3Tau([3]*2)

        assert tau.channels == 3

    @pytest.mark.parametrize('batch', [(1,), (3,), (1, 1), (3, 3)])
    @pytest.mark.parametrize('tau0', [(1,), (1, 2), (2, 2)])
    def test_from_rep(self, batch, tau0):
        rand_rep = lambda tau, batch: [torch.rand(batch + (t, 2*l+1, 2)).double() for l, t in enumerate(tau)]

        rep = rand_rep(tau0, batch)
        tau = SO3Tau.from_rep(rep)

        assert type(tau) == SO3Tau
        assert list(tau) == list(tau0)
