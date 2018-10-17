import unittest

from com.sbk.func.func_1 import calc_gradient, func
from com.sbk.gradientdescent.gradient_descent import gradient_descent_single_variable


class TestGradientDescentSingleVariable(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum_with_default_start_approximation(self):
        x, f = gradient_descent_single_variable(func_calc_gradient=calc_gradient, func=func, learning_rate=0.0001)
        self.assertAlmostEqual(x, 2.53506948945, places=4)
        self.assertAlmostEqual(f, -6.696234472468862, places=4)

    def test_should_find_minimum_with_start_approximation_1(self):
        x, f = gradient_descent_single_variable(func_calc_gradient=calc_gradient, func=func, start_x=1.3)
        self.assertAlmostEqual(x, 2.5358338694353826, places=4)
        self.assertAlmostEqual(f, -6.696234472468862, places=4)

    def test_should_find_minimum_with_start_approximation_4(self):
        x, f = gradient_descent_single_variable(func_calc_gradient=calc_gradient, func=func, start_x=2.7)
        self.assertAlmostEqual(x, 2.5358338694353826, places=4)
        self.assertAlmostEqual(f, -6.696234472468862, places=4)
