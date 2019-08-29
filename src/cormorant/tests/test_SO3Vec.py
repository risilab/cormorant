import torch
import pytest

from cormorant.nn import CatReps, MixReps, CatMixReps
from cormorant.cg_lib import SO3Tau, SO3Vec

class TestSO3Vec():

    def test_so3_vec(self):
        
        SO3Vec()
