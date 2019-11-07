import torch
import pytest

from cormorant.cg_lib import CGDict

class TestCGDict():

    # Test to see that an uninitialized CG dictionary doesn't do anything.
    def test_cg_dict_uninit(self):
        cg_dict = CGDict()

        assert(not cg_dict)

        with pytest.raises(ValueError) as e_info:
            cg_dict[(0, 0)]

    @pytest.mark.parametrize('maxl', [0, 1, 2])
    def test_cg_dict_init(self, maxl):
        cg_dict = CGDict(maxl=maxl)

        assert len(cg_dict.keys()) == (maxl+1)**2

        for key, val in cg_dict.items():
            assert val*val.t()

    @pytest.mark.parametrize('maxl', [0, 1, 2])
    def test_cg_dict_init(self, maxl):
        cg_dict = CGDict(maxl=maxl)

        assert set(cg_dict.keys()) == {(l1, l2) for l1 in range(maxl+1) for l2 in range(maxl+1)}

    @pytest.mark.parametrize('maxl', [1])
    @pytest.mark.parametrize('dtype', [torch.float, torch.double])
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_cg_dict_device_dtype(self, maxl, device, dtype):

        if (device == torch.device('cuda')) and (not torch.cuda.is_available()):
            with pytest.raises(AssertionError) as e_info:
                cg_dict = CGDict(maxl=maxl, device=device, dtype=dtype)
        else:
            cg_dict = CGDict(maxl=maxl, device=device, dtype=dtype)

    @pytest.mark.parametrize('maxl', [1])
    @pytest.mark.parametrize('dtype1', [torch.float, torch.double])
    @pytest.mark.parametrize('dtype2', [torch.float, torch.double])
    def test_cg_dict_to(self, maxl, dtype1, dtype2):

        cg_dict = CGDict(maxl=maxl, dtype=dtype1)

        assert cg_dict.dtype == cg_dict[(0,0)].dtype
        assert cg_dict.dtype == dtype1

        cg_dict.to(dtype=dtype2)

        assert cg_dict.dtype == cg_dict[(0,0)].dtype
        assert cg_dict.dtype == dtype2

    @pytest.mark.parametrize('maxl1', [0, 1, 3])
    @pytest.mark.parametrize('maxl2', [0, 1, 3])
    def test_cg_dict_update_maxl(self, maxl1, maxl2):

        maxl = max(maxl1, maxl2)

        cg_dict = CGDict(maxl=maxl1)
        cg_dict.update_maxl(maxl2)

        assert cg_dict.maxl == maxl
        assert set(cg_dict.keys()) == {(l1, l2) for l1 in range(maxl+1) for l2 in range(maxl+1)}
