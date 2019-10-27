import torch
import pytest

from torch.nn import Parameter

from cormorant.cg_lib import CGModule, CGDict

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append([torch.device('cuda')])


class TestCGModule():

    # Test to see that an uninitialized CG dictionary doesn't do anything.
    def test_cg_mod_nodict(self):
        cg_mod = CGModule()
        assert cg_mod.maxl is None
        assert not cg_mod.cg_dict
        assert cg_mod.device == torch.device('cpu')
        assert cg_mod.dtype == torch.float

    # Check the device works if maxl is not defined.
    @pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double, torch.long])
    def test_cg_mod_device(self, dtype):

        if dtype == torch.long:
            with pytest.raises(ValueError):
                cg_mod = CGModule(dtype=dtype)
        else:
            cg_mod = CGModule(dtype=dtype)
            assert cg_mod.dtype == dtype
            assert cg_mod.device == torch.device('cpu')
            assert cg_mod.maxl is None
            assert cg_mod.cg_dict is None

    # Check the cg_dict device works correctly if maxl is set.
    @pytest.mark.parametrize('maxl', range(3))
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    def test_cg_mod_cg_dict_dtype(self, maxl, dtype):

        cg_mod = CGModule(maxl=maxl, dtype=dtype)
        assert cg_mod.dtype == torch.float if dtype is None else dtype
        assert cg_mod.device == torch.device('cpu')
        assert cg_mod.maxl == maxl
        assert cg_mod.cg_dict
        assert cg_mod.cg_dict.maxl == maxl

    # ######### Check initialization.
    # Check the cg_dict device works correctly if maxl is set.
    @pytest.mark.parametrize('maxl', [None, 0, 1, 2])
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    def test_cg_mod_set_from_cg_dict(self, maxl, dtype):

        cg_dict = CGDict(maxl=1, dtype=torch.float)

        if dtype in [torch.half, torch.double]:
            # If data type in CGModule does not match CGDict, throw an errror
            with pytest.raises(ValueError):
                cg_mod = CGModule(maxl=maxl, dtype=dtype, cg_dict=cg_dict)
        else:
            cg_mod = CGModule(maxl=maxl, dtype=dtype, cg_dict=cg_dict)

            assert cg_mod.dtype == torch.float if dtype is None else dtype
            assert cg_mod.device == torch.device('cpu')
            assert cg_mod.maxl == maxl if maxl is not None else 1
            assert cg_mod.cg_dict
            assert cg_mod.cg_dict.maxl == max(1, maxl) if maxl is not None else 1

    # ########## Test device/data types

    # Check that the module and cg_dict's data types can be moved correctly.
    @pytest.mark.parametrize('dtype1', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('dtype2', [torch.half, torch.float, torch.double])
    def test_cg_mod_to_dtype(self, dtype1, dtype2):

        cg_mod = CGModule(maxl=1, dtype=dtype1)

        cg_mod.to(dtype=dtype2)
        assert cg_mod.dtype == dtype2
        assert cg_mod.cg_dict.dtype == dtype2
        assert all([t.dtype == dtype2 for t in cg_mod.cg_dict.values()])

    # Check that the module and cg_dict's devices can be moved correctly.
    # WARNING: Will not do anything useful if CUDA is not available.
    @pytest.mark.parametrize('device1', devices)
    @pytest.mark.parametrize('device2', devices)
    def test_cg_mod_to_device(self, device1, device2):

        cg_mod = CGModule(maxl=1, device=device1)

        cg_mod.to(device=device2)
        assert cg_mod.device == device2
        assert cg_mod.cg_dict.device == device2
        assert all([t.device == device2 for t in cg_mod.cg_dict.values()])

    # Check that the module and cg_dict's data types can be moved correctly with standard .to() syntax.
    @pytest.mark.parametrize('dtype1', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('dtype2', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('device1', devices)
    @pytest.mark.parametrize('device2', devices)
    def test_cg_mod_to(self, dtype1, dtype2, device1, device2):

        cg_mod = CGModule(maxl=1, dtype=dtype1, device=device1)

        cg_mod.to(device2, dtype2)
        assert cg_mod.dtype == dtype2
        assert cg_mod.cg_dict.dtype == dtype2
        assert all([t.dtype == dtype2 for t in cg_mod.cg_dict.values()])
        assert cg_mod.device == device2
        assert cg_mod.cg_dict.device == device2
        assert all([t.device == device2 for t in cg_mod.cg_dict.values()])

    # Check that .half() work as expected
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('maxl', [2])
    def test_cg_mod_half(self, maxl, dtype):

        cg_mod = CGModule(maxl=maxl, dtype=dtype)
        print(cg_mod.dtype, dtype)
        cg_mod.half()
        print(cg_mod.dtype, dtype)
        assert cg_mod.dtype == torch.half
        assert cg_mod.cg_dict.dtype == torch.half
        assert all([t.dtype == torch.half for t in cg_mod.cg_dict.values()])

    # Check that .float() work as expected
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('maxl', [2])
    def test_cg_mod_float(self, maxl, dtype):

        cg_mod = CGModule(maxl=maxl, dtype=dtype)
        cg_mod.float()
        assert cg_mod.dtype == torch.float
        assert cg_mod.cg_dict.dtype == torch.float
        assert all([t.dtype == torch.float for t in cg_mod.cg_dict.values()])

    # Check that .double() work as expected
    @pytest.mark.parametrize('dtype', [None, torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('maxl', [2])
    def test_cg_mod_double(self, maxl, dtype):

        cg_mod = CGModule(maxl=maxl, dtype=dtype)
        cg_mod.double()
        assert cg_mod.dtype == torch.double
        assert cg_mod.cg_dict.dtype == torch.double
        assert all([t.dtype == torch.double for t in cg_mod.cg_dict.values()])

    # Check that .cpu() work as expected
    @pytest.mark.parametrize('device', devices)
    @pytest.mark.parametrize('maxl', [2])
    def test_cg_mod_cpu(self, maxl, device):

        cg_mod = CGModule(maxl=maxl, device=device)
        cg_mod.cpu()
        assert cg_mod.device == torch.device('cpu')
        assert cg_mod.cg_dict.device == torch.device('cpu')
        assert all([t.device == torch.device('cpu') for t in cg_mod.cg_dict.values()])

    # Check that .cuda() work as expected
    @pytest.mark.parametrize('device', devices)
    @pytest.mark.parametrize('maxl', [2])
    def test_cg_mod_cuda(self, maxl, device):

        if not torch.cuda.is_available():
            return

        cg_mod = CGModule(maxl=maxl, device=device)
        cg_mod.cuda()
        assert cg_mod.device == torch.device('cuda')
        assert cg_mod.cg_dict.device == torch.device('cuda')
        assert all([t.device == torch.device('cuda') for t in cg_mod.cg_dict.values()])

    def test_register_parameter(self):
        class BasicCGModule(CGModule):
            def __init__(self):
                super().__init__()
                x = Parameter(torch.tensor(0.))
                self.register_parameter('x', x)

                self.y = Parameter(torch.tensor(1.))

        basic_cg = BasicCGModule()

        params = [key for key, val in basic_cg.named_parameters()]

        assert 'x' in params
        assert 'y' in params
