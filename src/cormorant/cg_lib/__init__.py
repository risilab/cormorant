import sys

# First need to import the CG dictionary
from cormorant.cg_lib.cg_dict import CGDict

# Module (not instance)-wide CG-dictionary solution based upon: https://stackoverflow.com/a/35904211
this = sys.modules[__name__]
this.global_cg_dict = CGDict()

# First need to import the CG dictionary
from cormorant.cg_lib.cg_module import CGModule

#Import SO3Tau
from cormorant.cg_lib.so3tau import SO3Tau, cg_product_tau

# Now for your regularly scheduled imports
from cormorant.cg_lib.cg_ops import CGProduct, cg_product

from cormorant.cg_lib.spherical_harmonics import spherical_harmonics, spherical_harmonics_rel, pos_to_rep, rep_to_pos
from cormorant.cg_lib.spherical_harmonics import SphericalHarmonics, SphericalHarmonicsRel

from cormorant.cg_lib.rotations import WignerD, WignerD_list, littled, rotate_part, rotate_rep
from cormorant.cg_lib.rotations import Rx, Ry, Rz, EulerRot, rotate_cart_vec, create_J
