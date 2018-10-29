import unittest

from com.sbk.func.beale_function import calc_gradient_beale, beale_function
from com.sbk.gradientdescent.gradient_descent import gradient_descent_multiple_variable


class TestBealeFunctionMinimum(unittest.TestCase):

    def test_calc_gradient_beale_start_point_4_5(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_beale,
                                                     func=beale_function,
                                                     tolerance=0.00000005,
                                                     start_x=(-4, -5),
                                                     learning_rate=(0.00001, 0.00001))
        self.assertAlmostEqual(x[0], 3, places=1)
        self.assertAlmostEqual(x[1], 0.5, places=1)
        self.assertAlmostEqual(f, 0, places=4)

    def test_calc_gradient_beale_start_point_5_5(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_beale,
                                                     func=beale_function,
                                                     tolerance=0.000000001,
                                                     start_x=(5, 5),
                                                     learning_rate=(0.00001, 0.00001))
        self.assertAlmostEqual(x[0], 3.27, places=1)
        self.assertAlmostEqual(x[1], 0.56, places=1)
        self.assertAlmostEqual(f, 0.0092, places=4)

    def test_calc_gradient_beale_start_point_4_1(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_beale,
                                                     func=beale_function,
                                                     tolerance=0.000000001,
                                                     start_x=(4, -1.5),
                                                     learning_rate=(0.00001, 0.00001))
        self.assertAlmostEqual(x[0], 3.22, places=1)
        self.assertAlmostEqual(x[1], 0.54, places=1)
        self.assertAlmostEqual(f, 0.0062, places=4)

    def test_calc_gradient_beale_start_point_4_2(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_beale,
                                                     func=beale_function,
                                                     tolerance=0.00000005,
                                                     start_x=(-2, -1),
                                                     learning_rate=(0.00001, 0.00001))
        self.assertAlmostEqual(x[0], 3, places=1)
        self.assertAlmostEqual(x[1], 0.5, places=1)
        self.assertAlmostEqual(f, 0, places=4)
