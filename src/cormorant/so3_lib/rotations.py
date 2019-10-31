import torch
import numpy as np

# from cormorant.so3_lib import so3_wigner_d
#
# SO3WignerD = so3_wigner_d.SO3WignerD

# TODO: Update legacy code to use SO3Vec/SO3WignerD interfaces
# TODO: Convert to PyTorch objects to allow for GPU parallelism and autograd support

# Explicitly construct functions for the 3D cartesian rotation matrices


# def Rx(theta):
#     """
#     Rotation Matrix for rotations on the x axis.
#
#     Parameters
#     ----------
#     theta : double
#         Angle over which to rotate.
#
#     Returns
#     -------
#     Rmat : :obj:`torch.Tensor`
#         The associated rotation matrix.
#     """
#     return torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=torch.double)


# def Ry(theta, device=None, dtype=None):
def Ry(theta):
    """
    Rotation Matrix for rotations on the y axis.

    Parameters
    ----------
    theta : double
        Angle over which to rotate.

    Returns
    -------
    Rmat : :obj:`torch.Tensor`
        The associated rotation matrix.
    """
    return torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.double)


def Rz(theta):
    """
    Rotation Matrix for rotations on the z axis. Syntax is the same as with Ry.
    """
    return torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.double)


def EulerRot(alpha, beta, gamma):
    """
    Constructs a Rotation Matrix from Euler angles.

    Parameters
    ----------
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle

    Returns
    -------
    Rmat : :obj:`torch.Tensor`
        The associated rotation matrix.
    """
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)


def gen_rot(maxl, angles=None, device=None, dtype=None):
    """
    Generate a rotation matrix corresponding to a Cartesian and also a Wigner-D
    representation of a specific  rotation. If `angles` is :obj:`None`, will
    generate random rotation.

    Parameters
    ----------
    maxl : :obj:`int`
        Maximum weight to include in the Wigner D-matrix list
    angles : :obj:`list` of :obj:`float` or compatible, optional
        Three Euler angles (alpha, beta, gamma) to parametrize the rotation.
    device : :obj:`torch.device`, optional
        Device of the output tensor
    dtype : :obj:`torch.dtype`, optional
        Data dype of the output tensor

    Returns
    -------
    D : :obj:`list` of :obj:`torch.Tensor`
        List of Wigner D-matrices from `l=0` to `l=maxl`
    R : :obj:`torch.Tensor`
        Rotation matrix that will perform the same cartesian rotation
    angles : :obj:`tuple`
        Euler angles that defines the input rotation
    """
    # TODO : Output D as SO3WignerD
    if angles is None:
        alpha, beta, gamma = np.random.rand(3) * 2*np.pi
        beta = beta / 2
        angles = alpha, beta, gamma
    else:
        assert len(angles) == 3
        alpha, beta, gamma = angles
    D = WignerD_list(maxl, alpha, beta, gamma, device=device, dtype=dtype)
    # R = EulerRot(alpha, beta, gamma, device=device, dtype=dtype)
    R = EulerRot(alpha, beta, gamma).to(device=device, dtype=dtype)

    return D, R, angles


def rotate_cart_vec(R, vec):
    """ Rotate a Cartesian vector by a Euler rotation matrix. """
    return torch.einsum('ij,...j->...i', R, vec)  # Broadcast multiplication along last axis.


def rotate_part(D, z, dir='left'):
    """ Apply a WignerD matrix using complex broadcast matrix multiplication. """
    Dr, Di = D.unbind(-1)
    zr, zi = z.unbind(-1)

    if dir == 'left':
        matmul = lambda D, z: torch.einsum('ij,...kj->...ki', D, z)
    elif dir == 'right':
        matmul = lambda D, z: torch.einsum('ji,...kj->...ki', D, z)
    else:
        raise ValueError('Must apply Wigner rotation from dir=left/right! got dir={}'.format(dir))

    return torch.stack((matmul(Dr, zr) - matmul(Di, zi),
                        matmul(Di, zr) + matmul(Dr, zi)), -1)


def rotate_rep(D_list, rep, dir='left'):
    """ Apply a WignerD rotation part-wise to a representation. """
    ls = [(part.shape[-2]-1)//2 for part in rep]
    D_maxls = (D_list[-1].shape[-2]-1)//2
    assert((D_maxls >= max(ls))), 'Must have at least one D matrix for each rep! {} {}'.format(D_maxls, len(rep))

    D_list = [D_list[l] for l in ls]
    return [rotate_part(D, part, dir=dir) for (D, part) in zip(D_list, rep)]


def dagger(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(1, 1, 2)
    D = (D*conj).permute((1, 0, 2))
    return D


def create_J(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    # Jx = (Jp + Jm) / complex(2, 0)
    # Jy = -(Jp - Jm) / complex(0, 2)
    Jz = np.diag(-np.arange(-j, j+1))
    Id = np.eye(2*j+1)

    return Jp, Jm, Jz, Id


def create_Jy(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    Jy = -(Jp - Jm) / complex(0, 2)

    return Jy


def create_Jx(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    Jx = (Jp + Jm) / complex(2, 0)

    return Jx


def littled(j, beta):
    Jy = create_Jy(j)

    evals, evecs = np.linalg.eigh(Jy)
    evecsh = evecs.conj().T
    evals_exp = np.diag(np.exp(complex(0, -beta)*evals))

    d = np.matmul(np.matmul(evecs, evals_exp), evecsh)

    return d


def WignerD(j, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=None):
    """
    Calculates the Wigner D matrix for a given degree and Euler Angle.

    Parameters
    ----------
    j : int
        Degree of the representation.
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle
    numpy_test : bool, optional
        ?????
    device : :obj:`torch.device`, optional
        Device of the output tensor
    dtype : :obj:`torch.dtype`, optional
        Data dype of the output tensor

    Returns
    -------
    D =


    """
    if device is None:
        device = torch.device('cpu')
    d = littled(j, beta)

    Jz = np.arange(-j, j+1)
    Jzl = np.expand_dims(Jz, 1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = np.exp(complex(0, -alpha)*Jzl)
    right = np.exp(complex(0, -gamma)*Jz)

    D = left * d * right

    if not numpy_test:
        D = complex_from_numpy(D, dtype=dtype, device=device)

    return D


def WignerD_list(jmax, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=None):
    """

    """
    if device is None:
        device = torch.device('cpu')
    return [WignerD(j, alpha, beta, gamma, numpy_test=numpy_test, dtype=dtype, device=device) for j in range(jmax+1)]


def complex_from_numpy(z, dtype=torch.float, device=None):
    """ Take a numpy array and output a complex array of the same size. """
    if device is None:
        device = torch.device('cpu')
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), -1)
