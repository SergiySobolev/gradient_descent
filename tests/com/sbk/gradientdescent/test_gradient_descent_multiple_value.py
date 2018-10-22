import unittest

from com.sbk.func.func_1 import calc_gradient_2, func_2
from com.sbk.gradientdescent.gradient_descent import is_points_not_close_enough, gradient_descent_multiple_variable


class TestGradientDescentMultipleVariable(unittest.TestCase):

    def setUp(self):
        pass

    def test_is_points_not_close_enough_1_1(self):
        self.assertTrue(is_points_not_close_enough(1, 2, 0.1))

    def test_is_points_not_close_enough_1_2(self):
        self.assertFalse(is_points_not_close_enough((1, 2), (1.01, 2.01), 0.02))

    def test_is_points_not_close_enough_1_3(self):
        self.assertTrue(is_points_not_close_enough((1, 2), (1.01, 2.01), 0.0193))

    def test_should_find_minimum_with_default_start_approximation(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_2, func=func_2,
                                                     tolerance=0.0000001)
        self.assertAlmostEqual(x[0], 5, places=4)
        self.assertAlmostEqual(x[1], 7, places=4)
        self.assertAlmostEqual(f, 0, places=8)

    def test_should_find_minimum_with_default_1(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_2,
                                                     func=func_2, start_x=(-4, -8), tolerance=0.0000001)
        self.assertAlmostEqual(x[0], 5, places=4)
        self.assertAlmostEqual(x[1], 7, places=4)
        self.assertAlmostEqual(f, 0, places=8)

    def test_should_find_minimum_with_learning_rate_0_03(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_2, func=func_2,
                                                     start_x=(-4, -8), learning_rate=(0.03, 0.03), tolerance=0.0000001)
        self.assertAlmostEqual(x[0], 5, places=4)
        self.assertAlmostEqual(x[1], 7, places=4)
        self.assertAlmostEqual(f, 0, places=8)
        self.assertLess(c.__len__(), 300)
        print(func_2(c))
