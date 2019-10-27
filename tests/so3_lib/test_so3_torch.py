import torch
import pytest
import numpy as np

from cormorant.so3_lib import SO3Tau, SO3Vec, SO3Scalar, so3_torch
from utils import numpy_from_complex


class TestSO3Torch():
    @pytest.mark.parametrize('batch1', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('batch2', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('batch3', [(1,), (2,), (2, 2)])
    @pytest.mark.parametrize('maxl1', [0, 2, 3])
    @pytest.mark.parametrize('maxl2', [2, 3])
    @pytest.mark.parametrize('maxl3', [2, 3])
    @pytest.mark.parametrize('channels1', [1, 2])
    @pytest.mark.parametrize('channels2', [1, 2])
    @pytest.mark.parametrize('channels3', [1, 2])
    def test_SO3Vec_cat(self, batch1, batch2, batch3, channels1, channels2, channels3, maxl1, maxl2, maxl3):
        tau1 = [channels1] * (maxl1+1)
        tau2 = [channels2] * (maxl2+1)
        tau3 = [channels3] * (maxl2+1)

        tau12 = SO3Tau.cat([tau1, tau2])
        tau123 = SO3Tau.cat([tau1, tau2, tau3])

        vec1 = SO3Vec.randn(tau1, batch1)
        vec2 = SO3Vec.randn(tau2, batch2)
        vec3 = SO3Vec.randn(tau3, batch3)

        if batch1 == batch2:
            vec12 = so3_torch.cat([vec1, vec2])

            assert vec12.tau == tau12
        else:
            with pytest.raises(RuntimeError):
                vec12 = so3_torch.cat([vec1, vec2])

        if batch1 == batch2 == batch3:
            vec123 = so3_torch.cat([vec1, vec2, vec3])

            assert vec123.tau == tau123
        else:
            with pytest.raises(RuntimeError):
                vec12 = so3_torch.cat([vec1, vec2, vec3])


class TestMultiplication():
    @pytest.mark.parametrize('maxl', [1, 3])
    @pytest.mark.parametrize('num_middle', [1, 2])
    def test_so3_scalar_so3_scalar_mul(self, maxl, num_middle):
        middle_dims = (3,) * num_middle
        scalar_size = (2,) + middle_dims + (4, 2)
        scalar1 = SO3Scalar([torch.randn(scalar_size) for i in range(maxl)])
        scalar1_numpy = [numpy_from_complex(ti) for ti in scalar1]
        scalar2 = SO3Scalar([torch.randn(scalar_size) for i in range(maxl)])
        scalar2_numpy = [numpy_from_complex(ti) for ti in scalar2]

        true_complex_product = [part1 * part2 for (part1, part2) in zip(scalar1_numpy, scalar2_numpy)]
        so3scalar_product = scalar1 * scalar2
        so3scalar_product_numpy = [numpy_from_complex(ti) for ti in so3scalar_product]
        for exp_prod, true_prod in zip(so3scalar_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

    @pytest.mark.parametrize('maxl', [1, 3])
    @pytest.mark.parametrize('num_middle', [1, 2])
    def test_so3_scalar_so3_vector_mul(self, maxl, num_middle):
        middle_dims = (3,) * num_middle
        scalar = SO3Scalar([torch.randn((2,) + middle_dims + (4, 2)) for i in range(maxl)])
        scalar_numpy = [numpy_from_complex(ti) for ti in scalar]
        vector = SO3Vec([torch.randn((2,) + middle_dims + (4, 2 * i+1, 2)) for i in range(maxl)])
        vector_numpy = [numpy_from_complex(ti) for ti in vector]

        true_complex_product = [np.expand_dims(part1, -1) * part2 for (part1, part2) in zip(scalar_numpy, vector_numpy)]
        so3sv_product = vector * scalar
        so3sv_product_numpy = [numpy_from_complex(ti) for ti in so3sv_product]
        for exp_prod, true_prod in zip(so3sv_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

        so3sv_product = scalar * vector
        so3sv_product_numpy = [numpy_from_complex(ti) for ti in so3sv_product]
        for exp_prod, true_prod in zip(so3sv_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

    def test_so3_vector_so3_vector_mul(self):
        maxl = 2
        middle_dims = (3, 3)
        vector1 = SO3Vec([torch.randn((2,) + middle_dims + (4, 2 * l+1, 2)) for l in range(maxl)])
        vector1_numpy = [numpy_from_complex(ti) for ti in vector1]
        vector2 = SO3Vec([torch.randn((2,) + middle_dims + (4, 2 * l+1, 2)) for l in range(maxl)])
        vector2_numpy = [numpy_from_complex(ti) for ti in vector2]

        true_complex_product = [part1 * part2 for (part1, part2) in zip(vector1_numpy, vector2_numpy)]
        with pytest.warns(RuntimeWarning):
            so3sv_product = vector1 * vector2
        so3sv_product_numpy = [numpy_from_complex(ti) for ti in so3sv_product]
        for exp_prod, true_prod in zip(so3sv_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

    @pytest.mark.parametrize('type_iterable', [list, tuple])
    @pytest.mark.parametrize('type_so3', [SO3Scalar, SO3Vec])
    def test_so3_vector_iterable_mul(self, type_iterable, type_so3):
        maxl = 2
        other_iterable = torch.randn(maxl)
        other_iterable_numpy = other_iterable.numpy()
        other_iterable = type_iterable([a for a in other_iterable])
        so3_object = type_so3([torch.randn((3, 3, 4, 2 * l+1, 2)) for l in range(maxl)])
        so3_object_numpy = [numpy_from_complex(ti) for ti in so3_object]

        true_complex_product = [part1 * part2 for (part1, part2) in zip(other_iterable_numpy, so3_object_numpy)]
        so3_product = so3_object * other_iterable
        so3_product_numpy = [numpy_from_complex(ti) for ti in so3_product]
        for exp_prod, true_prod in zip(so3_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

        so3_product = other_iterable * so3_object
        so3_product_numpy = [numpy_from_complex(ti) for ti in so3_product]
        for exp_prod, true_prod in zip(so3_product_numpy, true_complex_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)
    
    @pytest.mark.parametrize('type_so3', [SO3Scalar, SO3Vec])
    @pytest.mark.parametrize('other_object', [2, 0., -3.2])
    def test_so3_vector_number_mul(self, type_so3, other_object):
        maxl = 2
        so3_object = type_so3([torch.randn((3, 3, 4, 2 * l+1, 2)) for l in range(maxl)])
        so3_object_numpy = [numpy_from_complex(ti) for ti in so3_object]

        true_product = [other_object * part for part in so3_object_numpy]
        so3_product = so3_object * other_object
        so3_product_numpy = [numpy_from_complex(ti) for ti in so3_product]
        for exp_prod, true_prod in zip(so3_product_numpy, true_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)

        so3_product = other_object * so3_object
        so3_product_numpy = [numpy_from_complex(ti) for ti in so3_product]
        for exp_prod, true_prod in zip(so3_product_numpy, true_product):
            assert(np.sum(np.abs(exp_prod - true_prod)) < 1E-6)
