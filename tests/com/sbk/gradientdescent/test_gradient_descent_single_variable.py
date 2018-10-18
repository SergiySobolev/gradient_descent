import unittest

from com.sbk.func.func_1 import calc_gradient_1, func_1
from com.sbk.gradientdescent.gradient_descent import gradient_descent_single_variable


class TestGradientDescentSingleVariable(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum_with_default_start_approximation(self):
        x, f, c = gradient_descent_single_variable(func_calc_gradient=calc_gradient_1, func=func_1, learning_rate=0.01)
        self.assertAlmostEqual(x, 2.53506948945, places=2)
        self.assertAlmostEqual(f, -6.696234472468862, places=2)

    def test_should_find_minimum_with_start_approximation_1(self):
        x, f, c = gradient_descent_single_variable(func_calc_gradient=calc_gradient_1, func=func_1, start_x=1.8)
        self.assertAlmostEqual(x, 2.5358338694353826, places=4)
        self.assertAlmostEqual(f, -6.696234472468862, places=4)

    def test_should_find_minimum_with_start_approximation_4(self):
        x, f, c = gradient_descent_single_variable(func_calc_gradient=calc_gradient_1, func=func_1, start_x=2.7)
        self.assertAlmostEqual(x, 2.5358338694353826, places=4)
        self.assertAlmostEqual(f, -6.696234472468862, places=4)
