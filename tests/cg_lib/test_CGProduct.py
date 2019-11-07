import torch
import pytest

from cormorant.cg_lib import CGProduct, cg_product
from cormorant.cg_lib import CGDict
from cormorant.cg_lib import cg_product_tau
from cormorant.so3_lib import SO3Tau

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append([torch.device('cuda')])

# Test CG Product Initialization

class TestCGProductInitialization():

    # Test to see CGProduct crashes when maxl is not set
    def test_no_maxl(self):
        with pytest.raises(ValueError) as e_info:
            cg_prod = CGProduct()

    # Test to see CGProduct does not when maxl is not set, but cg_dict is
    @pytest.mark.parametrize('maxl', [1, 2])
    def test_no_maxl_w_cg_dict(self, maxl):
        cg_dict = CGDict(maxl=maxl)
        cg_prod = CGProduct(cg_dict=cg_dict)

        assert cg_prod.cg_dict is not None
        assert cg_prod.maxl is not None

    # Check the cg_dict device works correctly if maxl is set.
    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    def test_cg_prod_cg_dict_dtype(self, maxl, dtype):

        cg_prod = CGProduct(maxl=maxl, dtype=dtype)
        assert cg_prod.dtype == torch.float if dtype is None else dtype
        assert cg_prod.device == torch.device('cpu')
        assert cg_prod.maxl == maxl
        assert cg_prod.cg_dict
        assert cg_prod.cg_dict.maxl == maxl

    ########## Check initialization.

    # Check the cg_dict device works correctly if maxl is set.
    @pytest.mark.parametrize('maxl', [None, 0, 2])
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    def test_cg_prod_set_from_cg_dict(self, maxl, dtype):

        cg_dict = CGDict(maxl=1, dtype=torch.float)

        if dtype in [torch.half, torch.double]:
            # If data type in CGProduct does not match CGDict, throw an errror
            with pytest.raises(ValueError):
                cg_prod = CGProduct(maxl=maxl, dtype=dtype, cg_dict=cg_dict)
        else:
            cg_prod = CGProduct(maxl=maxl, dtype=dtype, cg_dict=cg_dict)

            assert cg_prod.dtype == torch.float if dtype is None else dtype
            assert cg_prod.device == torch.device('cpu')
            assert cg_prod.maxl == maxl if maxl is not None else 1
            assert cg_prod.cg_dict
            assert cg_prod.cg_dict.maxl == max(1, maxl) if maxl is not None else 1

    # Check the cg_dict device works correctly if maxl is set.
    @pytest.mark.parametrize('maxl1', [0, 2])
    @pytest.mark.parametrize('maxl2', [0, 2])
    @pytest.mark.parametrize('chan1', [1, 2])
    @pytest.mark.parametrize('chan2', [1, 2])
    @pytest.mark.parametrize('set_tau1', [True, False])
    @pytest.mark.parametrize('set_tau2', [True, False])
    def test_cg_prod_tau_check(self, maxl1, maxl2, chan1, chan2, set_tau1, set_tau2):
        rand_rep = lambda tau, nbatch: [torch.rand(nbatch + (t, 2*l+1, 2)).double() for l, t in enumerate(tau)]

        tau1 = [chan1] * (maxl1 + 1)
        tau2 = [chan2] * (maxl2 + 1)

        rep1 = rand_rep(tau1, (2,))
        rep2 = rand_rep(tau2, (2,))

        tau1_in = tau1 if set_tau1 else None
        tau2_in = tau2 if set_tau2 else None

        if (set_tau1 and set_tau2) and chan1 != chan2:
            with pytest.raises(ValueError) as e:
                cg_prod = CGProduct(tau1_in, tau2_in, maxl=2)
            return
        else:
            cg_prod = CGProduct(tau1_in, tau2_in, maxl=2)

        if set_tau1 and set_tau2:
            tau_out = cg_prod.tau_out
        else:
            with pytest.raises(ValueError) as e:
                tau_out = cg_prod.tau_out



# Test actual CG Product properties inherited from CG Module

class TestCGProductInheritedFromCGModule():

    ########### Test device/data types

    # Check that the module and cg_dict's data types can be moved correctly.
    @pytest.mark.parametrize('dtype1', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('dtype2', [torch.half, torch.float, torch.double])
    def test_cg_prod_to_dtype(self, dtype1, dtype2):

        cg_prod = CGProduct(maxl=1, dtype=dtype1)

        cg_prod.to(dtype=dtype2)
        assert cg_prod.dtype == dtype2
        assert cg_prod.cg_dict.dtype == dtype2
        assert all([t.dtype == dtype2 for t in cg_prod.cg_dict.values()])

    # Check that the module and cg_dict's devices can be moved correctly.
    # WARNING: Will not do anything useful if CUDA is not available.
    @pytest.mark.parametrize('device1', devices)
    @pytest.mark.parametrize('device2', devices)
    def test_cg_prod_to_device(self, device1, device2):

        cg_prod = CGProduct(maxl=1, device=device1)

        cg_prod.to(device=device2)
        assert cg_prod.device == device2
        assert cg_prod.cg_dict.device == device2
        assert all([t.device == device2 for t in cg_prod.cg_dict.values()])

    # Check that the module and cg_dict's data types can be moved correctly with standard .to() syntax.
    @pytest.mark.parametrize('dtype1', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('dtype2', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('device1', devices)
    @pytest.mark.parametrize('device2', devices)

    def test_cg_prod_to(self, dtype1, dtype2, device1, device2):

        cg_prod = CGProduct(maxl=1, dtype=dtype1, device=device1)

        cg_prod.to(device2, dtype2)
        assert cg_prod.dtype == dtype2
        assert cg_prod.cg_dict.dtype == dtype2
        assert all([t.dtype == dtype2 for t in cg_prod.cg_dict.values()])
        assert cg_prod.device == device2
        assert cg_prod.cg_dict.device == device2
        assert all([t.device == device2 for t in cg_prod.cg_dict.values()])

        # Check that .half() work as expected
        @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
        def test_cg_prod_half(self, maxl, dtype):

            cg_prod = CGProduct(maxl=maxl, dtype=dtype)
            cg_prod.half()
            assert cg_prod.dtype == torch.half
            assert cg_prod.cg_dict.dtype == torch.half
            assert all([t.device == torch.half for t in cg_prod.cg_dict.values()])

        # Check that .float() work as expected
        @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
        def test_cg_prod_float(self, maxl, dtype):

            cg_prod = CGProduct(maxl=maxl, dtype=dtype)
            cg_prod.float()
            assert cg_prod.dtype == torch.float
            assert cg_prod.cg_dict.dtype == torch.float
            assert all([t.device == torch.float for t in cg_prod.cg_dict.values()])

        # Check that .double() work as expected
        @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
        def test_cg_prod_double(self, maxl, dtype):

            cg_prod = CGProduct(maxl=maxl, dtype=dtype)
            cg_prod.double()
            assert cg_prod.dtype == torch.double
            assert cg_prod.cg_dict.dtype == torch.double
            assert all([t.device == torch.double for t in cg_prod.cg_dict.values()])

        # Check that .cpu() work as expected
        @pytest.mark.parametrize('device', devices)
        def test_cg_prod_cpu(self, maxl, device):

            cg_prod = CGProduct(maxl=maxl, device=device)
            cg_prod.cpu()
            assert cg_prod.device == torch.device('cpu')
            assert cg_prod.cg_dict.device == torch.device('cpu')
            assert all([t.device == torch.device('cpu') for t in cg_prod.cg_dict.values()])

        # Check that .cuda() work as expected
        @pytest.mark.parametrize('device', devices)
        def test_cg_prod_cuda(self, maxl, device):

            if not torch.cuda.is_available():
                return

            cg_prod = CGProduct(maxl=maxl, device=device)
            cg_prod.cuda()
            assert cg_prod.device == torch.device('cuda')
            assert cg_prod.cg_dict.device == torch.device('cuda')
            assert all([t.device == torch.device('cuda') for t in cg_prod.cg_dict.values()])
