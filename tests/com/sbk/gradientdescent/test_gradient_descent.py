import unittest

from com.sbk.gradientdescent.gradient_descent import gradient_descent, calc_gradient


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum_with_default_start_approximation(self):
        x, f = gradient_descent(func_calc_gradient=calc_gradient)
        self.assertAlmostEqual(x, 2.469303230007481, places=2)
        self.assertAlmostEqual(f, -6.664790734663696, places=2)

    def test_should_find_minimum_with_start_approximation_1(self):
        x, f = gradient_descent(func_calc_gradient=calc_gradient, start_x = 1.3)
        self.assertAlmostEqual(x, 2.469303230007481, places=2)
        self.assertAlmostEqual(f, -6.664790734663696, places=2)

    def test_should_find_minimum_with_start_approximation_4(self):
        x, f = gradient_descent(func_calc_gradient=calc_gradient, start_x = 2.7, tolerance=0.000001, learning_rate = -0.01)
        #self.assertAlmostEqual(x, 2.469303230007481, places=2)
        #self.assertAlmostEqual(f, -6.664790734663696, places=2)
