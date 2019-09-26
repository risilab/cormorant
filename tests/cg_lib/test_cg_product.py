import torch
import pytest

from cormorant.cg_lib import CGProduct, cg_product, cg_product_tau
from cormorant.cg_lib import CGDict
from cormorant.so3_lib import SO3Tau, SO3Vec
import cormorant.so3_lib.rotations as rot

# Test cg_product runs and aggregate=True works

class TestCGProduct():

    @pytest.mark.parametrize('maxl1', range(2))
    @pytest.mark.parametrize('maxl2', range(2))
    @pytest.mark.parametrize('chan', [1, 2, 5])
    @pytest.mark.parametrize('batch', [(1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)])
    @pytest.mark.parametrize('maxl_dict', range(2))
    @pytest.mark.parametrize('maxl_prod', range(2))
    def test_cg_product_dict_maxl(self, maxl_dict, maxl_prod, maxl1, maxl2, chan, batch):
        cg_dict = CGDict(maxl=maxl_dict, dtype=torch.double)

        tau1, tau2 = [chan]*(maxl1+1), [chan]*(maxl2+1)

        rep1 = SO3Vec.rand(batch, tau1, dtype=torch.double)
        rep2 = SO3Vec.rand(batch, tau2, dtype=torch.double)

        if all(maxl_dict >= maxl for maxl in [maxl_prod, maxl1, maxl2]):
            cg_prod = cg_product(cg_dict, rep1, rep2, maxl=maxl_prod)

        else:
            with pytest.raises(ValueError) as e_info:
                cg_prod = cg_product(cg_dict, rep1, rep2, maxl=maxl_prod)

                tau_out = cg_prod.tau
                tau_pred = cg_product_tau(tau1, tau2)

                # Test to make sure the output type matches the expected output type
                assert list(tau_out) == list(tau_pred)

            assert str(e_info.value).startswith('CG Dictionary maxl')


    @pytest.mark.parametrize('maxl1', range(2))
    @pytest.mark.parametrize('maxl2', range(2))
    @pytest.mark.parametrize('chan', [1, 2, 5])
    @pytest.mark.parametrize('batch', [1, 2, 5])
    @pytest.mark.parametrize('atom1', [1, 2, 5])
    @pytest.mark.parametrize('atom2', [1, 2, 4])
    @pytest.mark.parametrize('maxl_dict', range(2))
    @pytest.mark.parametrize('maxl_prod', range(2))
    def test_cg_aggregate(self, maxl_dict, maxl_prod, maxl1, maxl2, chan, batch, atom1, atom2):
        if any(maxl_dict < maxl for maxl in [maxl_prod, maxl1, maxl2]):
            return

        cg_dict = CGDict(maxl=maxl_dict, dtype=torch.double)
        rand_rep = lambda tau, batch: [torch.rand(batch + (t, 2*l+1, 2)).double() for l, t in enumerate(tau)]

        tau1, tau2 = [chan]*(maxl1+1), [chan]*(maxl2+1)

        batch1 = (batch, atom1, atom2)
        batch2 = (batch, atom2)

        rep1 = rand_rep(tau1, batch1)
        rep2 = rand_rep(tau2, batch2)

        # Calculate CG Aggregate and compare it to explicit calculation
        cg_agg = cg_product(cg_dict, rep1, rep2, maxl=maxl_prod, aggregate=True)

        cg_agg_explicit = [torch.zeros_like(p) for p in cg_agg]
        for bidx in range(batch):
        	for aidx1 in range(atom1):
        	       for aidx2 in range(atom2):
                		rep1_sub = [p[bidx, aidx1, aidx2] for p in rep1]
                		rep2_sub = [p[bidx, aidx2] for p in rep2]
                		cg_out = cg_product(cg_dict, rep1_sub, rep2_sub, maxl=maxl_prod, aggregate=False)
                		out_ell = [(p.shape[-2]-1)//2 for p in cg_out]
                		for ell in out_ell:
                			cg_agg_explicit[ell][bidx, aidx1] += cg_out[ell]

        for part1, part2 in zip(cg_agg, cg_agg_explicit):
            assert torch.allclose(part1, part2)

        cg_agg = cg_product(cg_dict, rep2, rep1, maxl=maxl_prod, aggregate=True)

        cg_agg_explicit = [torch.zeros_like(p) for p in cg_agg]
        for bidx in range(batch):
        	for aidx1 in range(atom1):
        	       for aidx2 in range(atom2):
                		rep1_sub = [p[bidx, aidx1, aidx2] for p in rep1]
                		rep2_sub = [p[bidx, aidx2] for p in rep2]
                		cg_out = cg_product(cg_dict, rep2_sub, rep1_sub, maxl=maxl_prod, aggregate=False)
                		out_ell = [(p.shape[-2]-1)//2 for p in cg_out]
                		for ell in out_ell:
                			cg_agg_explicit[ell][bidx, aidx1] += cg_out[ell]

        for part1, part2 in zip(cg_agg, cg_agg_explicit):
            assert torch.allclose(part1, part2)

def gen_rot(angles, maxl):
    alpha, beta, gamma = angles
    D = rot.WignerD_list(maxl, alpha, beta, gamma, dtype=torch.double)
    R = rot.EulerRot(alpha, beta, gamma)

    return D, R

# Test CG product covariance
class TestCGProductCovariance():

    @pytest.mark.parametrize('maxl_cg', range(2))
    @pytest.mark.parametrize('maxl1', range(2))
    @pytest.mark.parametrize('maxl2', range(2))
    @pytest.mark.parametrize('channels', [1, 2, 5])
    @pytest.mark.parametrize('batch', [1, 2, 5])
    def test_cg_product_covariance(self, maxl_cg, maxl1, maxl2, channels, batch):
        maxl = max(maxl_cg, maxl1, maxl2)

        cg_dict = CGDict(maxl=maxl, dtype=torch.double)
        rand_rep = lambda tau, batch: [torch.rand(batch + (t, 2*l+1, 2)).double() for l, t in enumerate(tau)]

        angles = torch.rand(3)
        D, R = gen_rot(angles, maxl)

        tau1 = [channels] * (maxl1 + 1)
        tau2 = [channels] * (maxl2 + 1)

        rep1 = rand_rep(tau1, (batch,))
        rep2 = rand_rep(tau2, (batch,))

        cg_prod = cg_product(cg_dict, rep1, rep2, maxl=maxl_cg)

        cg_prod_rot_out = rot.rotate_rep(D, cg_prod)

        rep1_rot = rot.rotate_rep(D, rep1)
        rep2_rot = rot.rotate_rep(D, rep2)

        cg_prod_rot_in = cg_product(cg_dict, rep1_rot, rep2_rot, maxl=maxl_cg)

        for part1, part2 in zip(cg_prod_rot_out, cg_prod_rot_in):
            assert torch.allclose(part1, part2)
