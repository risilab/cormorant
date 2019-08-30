import sys

# First need to import the CG dictionary
from cormorant.cg_lib.cg_dict import CGDict

# First need to import the CG dictionary
from cormorant.cg_lib.cg_module import CGModule

# Import some basic complex utilities
from cormorant.cg_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar

#Import SO3Tensor and related classes
from cormorant.cg_lib.so3_tau import SO3Tau

# Set up
from cormorant.cg_lib.so3_tensor_base import SO3TensorBase
from cormorant.cg_lib import so3_torch
from cormorant.cg_lib.so3_torch import add, mul, cat, mix, cat_mix
from cormorant.cg_lib.so3_tensor import SO3Tensor


from cormorant.cg_lib.so3_scalar import SO3Scalar
from cormorant.cg_lib.so3_vec import SO3Vec

# Import tau calculation for cg_ops
from cormorant.cg_lib.cg_ops_tau import cg_product_tau

# Now for your regularly scheduled imports
from cormorant.cg_lib.cg_ops import CGProduct, cg_product

from cormorant.cg_lib.spherical_harmonics import spherical_harmonics, spherical_harmonics_rel, pos_to_rep, rep_to_pos
from cormorant.cg_lib.spherical_harmonics import SphericalHarmonics, SphericalHarmonicsRel

from cormorant.cg_lib.rotations import WignerD, WignerD_list, littled, rotate_part, rotate_rep
from cormorant.cg_lib.rotations import Rx, Ry, Rz, EulerRot, rotate_cart_vec, create_J
