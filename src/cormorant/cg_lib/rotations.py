import torch
import numpy as np

# 3D cartesian rotation matrices
Rx = lambda theta: torch.Tensor([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
Ry = lambda theta: torch.Tensor([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
Rz = lambda theta: torch.Tensor([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
EulerRot = lambda alpha, beta, gamma: Rz(alpha) @ Ry(beta) @ Rz(gamma)


def rotate_cart_vec(R, vec, autoconvert=True):
    """ Rotate a Cartesian vector by a Euler rotation matrix. """
    if autoconvert:
        R = R.to(vec.device, vec.dtype)
    return torch.matmul(vec, R) # Broadcast multiplication along last axis.


def rotate_part(D, z, autoconvert=True):
    """ Apply a WignerD matrix using complex broadcast matrix multiplication. """
    if autoconvert:
        D = D.to(z.device, z.dtype)
    Dr, Di = D.unbind(-1)
    zr, zi = z.unbind(-1)

    return torch.stack((torch.matmul(zr, Dr) - torch.matmul(zi, Di),
                        torch.matmul(zr, Di) + torch.matmul(zi, Dr)), -1)


def rotate_so3part(D, z, side='left', autoconvert=True, conjugate=False):
    """ Apply a WignerD matrix using complex broadcast matrix multiplication. """
    if autoconvert:
        D = D.to(z.device, z.dtype)
    if conjugate:
        D = dagger(D)
    Dr, Di = D.unbind(-1)
    zr, zi = z.unbind(-1)

    if side == 'left':
        return torch.stack((torch.matmul(zr, Dr) - torch.matmul(zi, Di),
                            torch.matmul(zr, Di) + torch.matmul(zi, Dr)), -1)
    elif side == 'right':
        return torch.stack((torch.matmul(Dr, zr) - torch.matmul(Di, zi),
                            torch.matmul(Di, zr) + torch.matmul(Dr, zi)), -1)
    else:
        raise ValueError('Must chose side: left/right.')

def dagger(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(1, 1, 2)
    D = (D*conj).permute((1, 0, 2))
    return D

def rotate_rep(D_list, rep):
    """ Apply a WignerD rotation part-wise to a representation. """
    ls = [(part.shape[-2]-1)//2 for part in rep]
    D_maxls = (D_list[-1].shape[-2]-1)//2
    assert((D_maxls >= max(ls))), 'Must have at least one D matrix for each rep! {} {}'.format(D_maxls, len(rep))

    D_list = [D_list[l] for l in ls]
    return [rotate_part(D, part) for (D, part) in zip(D_list, rep)]


def rotate_so3rep(D_list, rep, side='left', conjugate=False):
    """ Apply a part-wise left/right sided WignerD rotation to a SO3 (matrix) representation. """
    ls = [(part.shape[-2]-1)//2 for part in rep]
    D_maxls = (D_list[-1].shape[-2]-1)//2
    assert((D_maxls >= max(ls))), 'Must have at least one D matrix for each rep! {} {}'.format(D_maxls, len(rep))

    D_list = [D_list[l] for l in ls]
    return [rotate_so3part(D, part, side=side, conjugate=conjugate) for (D, part) in zip(D_list, rep)]


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


def WignerD(j, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu')):
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


def WignerD_list(jmax, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu')):
    return [WignerD(j, alpha, beta, gamma, numpy_test=numpy_test, dtype=dtype, device=device) for j in range(jmax+1)]


def complex_from_numpy(z, dtype=torch.float, device=torch.device('cpu')):
    """ Take a numpy array and output a complex array of the same size. """
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), -1)
