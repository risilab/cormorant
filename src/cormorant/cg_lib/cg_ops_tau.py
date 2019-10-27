from math import inf

from cormorant.so3_lib import SO3Tau


def cg_product_tau(tau1, tau2, maxl=inf):
    """
    Calulate output multiplicity of the CG Product of two SO3 Vectors
    given the multiplicty of two input SO3 Vectors.

    Parameters
    ----------
    tau1 : :class:`list` of :class:`int`, :class:`SO3Tau`.
        Multiplicity of first representation.

    tau2 : :class:`list` of :class:`int`, :class:`SO3Tau`.
        Multiplicity of second representation.

    maxl : :class:`int`
        Largest weight to include in CG Product.

    Return
    ------

    tau : :class:`SO3Tau`
        Multiplicity of output representation.

    """
    tau1 = SO3Tau(tau1)
    tau2 = SO3Tau(tau2)

    L1, L2 = tau1.maxl, tau2.maxl
    L = min(L1 + L2, maxl)

    tau = [0]*(L+1)

    for l1 in range(L1+1):
        for l2 in range(L2+1):
            lmin, lmax = abs(l1-l2), min(l1+l2, maxl)
            for l in range(lmin, lmax+1):
                tau[l] += tau1[l1]

    return SO3Tau(tau)
