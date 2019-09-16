import torch
import numpy as np

# TODO: Update legacy code to use SO3Vec/SO3WignerD interfaces
# TODO: Convert to PyTorch objects to allow for GPU parallelism and autograd support

# Explicitly construct 3D cartesian rotation matrices
Rx = lambda theta: torch.tensor([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]], dtype=torch.double)
Ry = lambda theta: torch.tensor([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]], dtype=torch.double)
Rz = lambda theta: torch.tensor([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]], dtype=torch.double)
EulerRot = lambda alpha, beta, gamma: Rz(alpha) @ Ry(beta) @ Rz(gamma)


def gen_rot(maxl, angles=None):
	"""
	Generate a rotation matrix corresponding to a Cartesian and also a Wigner-D
	representation of a specific  rotation. If `angles` is :obj:`None`, will
	generate random rotation.

	Parameters
	----------
	maxl : :obj:`int`
		Maximum weight to include in the Wigner D-matrix list
	angles : :obj:`list` of :obj:`float` or compatible
		Three Euler angles (alpha, beta, gamma) to parametrize the rotation.
	"""
	if angles is None:
		alpha, beta, gamma = np.random.rand(3) * 2*np.pi
		beta = beta / 2
	else:
		assert len(angles) == 3
		alpha, beta, gamma = angles
	D = WignerD_list(maxl, alpha, beta, gamma)
	R = EulerRot(alpha, beta, gamma)

	return D, R


def rotate_cart_vec(R, vec):
	""" Rotate a Cartesian vector by a Euler rotation matrix. """
	return torch.matmul(vec, R) # Broadcast multiplication along last axis.


def rotate_part(D, z):
	""" Apply a WignerD matrix using complex broadcast matrix multiplication. """
	Dr, Di = D.unbind(-1)
	zr, zi = z.unbind(-1)

	matmul = lambda D, z : torch.einsum('ij,...jk->...ik', D, z)

	return torch.stack((matmul(Dr, zr) - matmul(Di, zi),
						matmul(Di, zr) + matmul(Dr, zi)), -1)


def rotate_rep(D_list, rep):
	""" Apply a WignerD rotation part-wise to a representation. """
	ls = [(part.shape[-2]-1)//2 for part in rep]
	D_maxls = (D_list[-1].shape[-2]-1)//2
	assert((D_maxls >= max(ls))), 'Must have at least one D matrix for each rep! {} {}'.format(D_maxls, len(rep))

	D_list = [D_list[l] for l in ls]
	return [rotate_part(D, part) for (D, part) in zip(D_list, rep)]


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
