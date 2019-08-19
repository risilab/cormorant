import torch
import numpy as np
from scipy.special import factorial

import logging
logger = logging.getLogger(__name__)

class CGDict():
    """
    A dictionary of Clebsch-Gordan coefficients to be used in CG operations.

    The module has a dict-like interface with keys :math:`(l_1, l_2)` for
    :math:`l_1, l_2 \\leq l_{\rm max}`. Each value is a matrix of shape
    :math:`D x D`, where :math:`D = (2l_1+1)(2l_2+1)`. The matrix has elements
    :math:`\\langle l_1, m_1, l_2, m_2 | l, m \\rangle`.

    Parameters
    ==========
    maxl: int
        Maximum weight for which to calculate the Clebsch-Gordan coefficients.
        This refers to the maximum weight for the `input tensors`, not the
        output tensors.

    transpose: bool
        Transpose the CG coefficient matrix for each :(l1, l2):. Cannot be
        modified after instantiation.

    device: torch.device
        Device of CG dictionary.

    dtype: torch.dtype
        Data type of CG dictionary.
    """

    def __init__(self, maxl=None, transpose=True, dtype=torch.float, device=torch.device('cpu')):

        self.dtype = dtype
        self.device = device
        self._transpose = transpose
        self._maxl = None
        self._cg_dict = {}

        if maxl is not None:
            self.update_maxl(maxl)

    @property
    def transpose(self):
        """
        Use "transposed" version of CG coefficients.
        """
        return self._transpose

    @property
    def maxl(self):
        """
        Maximum weight for CG coefficients.
        """
        return self._maxl

    def update_maxl(self, new_maxl):
        """
        Update maxl to a new (possibly larger) value. If the new_maxl is
        larger than the current maxl, new CG coefficients should be calculated
        and the cg_dict will be updated.

        Otherwise, do nothing.

        Parameters
        ==========
        new_maxl: int
            New maximum weight.

        Return
        ======
        self: CGDict()
            Returns self with a possibly updated self.cg_dict.
        """
        # If self is already initialized, and maxl is sufficiently large, do nothing
        if self and (self.maxl >= new_maxl):
            return self

        # If self is false, old_maxl = 0 (uninitialized).
        old_maxl = self.maxl if self else 0

        # Otherwise, update the CG coefficients.
        cg_dict_new = gen_cg_dict(new_maxl, transpose=self.transpose, existing_keys=self._cg_dict.keys())

        # Ensure elements of new CG dict are on correct device.
        cg_dict_new = {key: val.to(device=self.device, dtype=self.dtype) for key, val in cg_dict_new.items()}

        # Now update the CG dict, and also update maxl
        self._cg_dict.update(cg_dict_new)
        self._maxl = new_maxl

        return self


    def to(self, dtype=None, device=None):
        """
        Convert CGDict() to a new device/dtype.

        Parameters
        ==========
        device : torch.device, optional
            Device to move the cg_dict to.
        dtype : torch.dtype, optional
            Data type to convert the cg_dict to.
        """
        if dtype is None and device is None:
            pass
        elif dtype is None and device is not None:
            self._cg_dict = {key: val.to(device=device) for key, val in self._cg_dict.items()}
            self.device = device
        elif dtype is not None and device is None:
            self._cg_dict = {key: val.to(dtype=dtype) for key, val in self._cg_dict.items()}
            self.dtype = dtype
        elif dtype is not None and device is not None:
            self._cg_dict = {key: val.to(device=device, dtype=dtype) for key, val in self._cg_dict.items()}
            self.device, self.dtype = device, dtype
        return self

    def keys(self):
        return self._cg_dict.keys()

    def values(self):
        return self._cg_dict.values()

    def items(self):
        return self._cg_dict.items()

    def __getitem__(self, idx):
        if  self.maxl is None:
            raise ValueError('CGDict() not initialized. Either set maxl, or use update_maxl()')
        return self._cg_dict[idx]

    def __bool__(self):
        """
        Check to see if CGDict has been properly initialized, since :maxl=-1: initially.
        """
        return self.maxl is not None


def gen_cg_dict(maxl, transpose=False, existing_keys={}):
    """
    Generate all Clebsch-Gordan coefficients for a weight up to maxl.

    Parameters
    ==========
    maxl: int
        Maximum weight to generate CG coefficients.

    Return
    ======
    cg_dict:
        Dictionary of CG basis transformation matrices with keys :(l1, l2):,
        and matrices that convert a tensor product of irreps of type :l1: and :l2:
        into a direct sum of irreps :l: from :abs(l1-l2): to :l1+l2:
    """
    cg_dict = {}

    for l1 in range(maxl+1):
        for l2 in range(maxl+1):
            if (l1, l2) in existing_keys:
                continue

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

            cg_mat = cg_mat.view(N, N)
            if transpose:
                cg_mat = cg_mat.transpose(0, 1)
            cg_dict[(l1, l2)] = cg_mat

    return cg_dict


# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

def clebsch(j1, j2, j3, m1, m2, m3):
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) * factorial(j1 + m1) *
                factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C
