import torch

def mul_zscalar_zirrep(scalar, part, cdim=-2, zdim=-1):
    """
    Multiply the part of a :obj:`SO3Scalar` and a part of a :obj:`SO3Vec`.

    Parameters
    ----------
    scalar : :obj:`torch.Tensor`
        A tensor of scalars to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Vec` to multiply by scalars.

    """
    scalar_r, scalar_i = scalar.unsqueeze(cdim).unbind(zdim)
    part_r, part_i = rep.unbind(zdim)

    return torch.stack([part_r*scalar_r - part_i*scalar_i, part_r*scalar_i + part_i*scalar_r], dim=zdim)

def mul_zscalar_zscalar(scalar1, scalar2, zdim=-1):
    """
    Complex multiply the part of a :obj:`SO3Scalar` and a part of a
    different :obj:`SO3Scalar`.

    Parameters
    ----------
    scalar1 : :obj:`torch.Tensor`
        First tensor of scalars to multiply.
    scalar2 : :obj:`torch.Tensor`
        Second tensor of scalars to multiply.
    zdim : :obj:`int`
        Dimension for which complex multiplication is defined.


    """
    scalar1_r, scalar1_i = scalar1.unbind(zdim)
    scalar2_r, scalar2_i = scalar2.unbind(zdim)

    return torch.stack([scalar1_r*scalar2_r - scalar1_i*scalar2_i,
                        scalar1_r*scalar2_i + scalar1_i*scalar2_r], dim=zdim)
