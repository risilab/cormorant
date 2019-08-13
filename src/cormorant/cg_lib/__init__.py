import sys

# Module (not instance)-wide CG-dictionary solution based upon: https://stackoverflow.com/a/35904211
from .cg_coefficients import CGDict
this = sys.modules[__name__]
this.global_cg_dict = CGDict()

# Now for your regularly scheduled imports
from cormorant.cg_lib.cg_ops import CGProduct, cg_product, cg_product_tau

from cormorant.cg_lib.spherical_harmonics import spherical_harmonics, spherical_harmonics_rel, pos_to_rep, rep_to_pos
from cormorant.cg_lib.spherical_harmonics import SphericalHarmonics, SphericalHarmonicsRel

from cormorant.cg_lib.rotations import WignerD, WignerD_list, littled, rotate_part, rotate_rep
from cormorant.cg_lib.rotations import Rx, Ry, Rz, EulerRot, rotate_cart_vec, create_J
