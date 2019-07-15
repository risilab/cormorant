import torch

from .gen_cg_qutip import clebsch

def gen_cg_coefffs(maxl):
    cg_mats = []

    for l1 in range(maxl+1):
        for l2 in range(maxl+1):
            lmin, lmax = abs(l1 - l2), l1 + l2
            N1, N2 = 2*l1+1, 2*l2+1
            N = N1*N2
            cg_mat = torch.zeros((N1, N2, N), dtype=torch.double)
            for l in range(lmin, lmax+1):
                l_off = l*l - lmin*lmin
                for m1 in range(-l1, l1+1):
                    for m2 in range(-l2, l2+1):
                        for m in range(-l, l+1):
                            if m == m1 + m2:
                                cg_mat[l1+m1, l2+m2, l+m+l_off] = clebsch(l1, l2, l, m1, m2, m)

            cg_mats.append(cg_mat)

    return cg_mats
