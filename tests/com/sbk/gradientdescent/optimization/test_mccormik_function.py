import unittest

from com.sbk.func.mccormik_function import mccormik_function, calc_gradient_mccormik_function
from com.sbk.gradientdescent.gradient_descent import gradient_descent_multiple_variable


class TestMccormicFunctionMinimum(unittest.TestCase):

    def test_calc_gradient_mccormik_1(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_mccormik_function,
                                                     func=mccormik_function,
                                                     tolerance=0.0001,
                                                     start_x=(-1, -2),
                                                     learning_rate=(0.01, 0.01))
        self.assertAlmostEqual(x[0], -0.54719, places=1)
        self.assertAlmostEqual(x[1], -1.54719, places=1)
        self.assertAlmostEqual(f, -1.9132, places=4)

    def test_calc_gradient_mccormik_2(self):
        x, f, c = gradient_descent_multiple_variable(func_calc_gradient=calc_gradient_mccormik_function,
                                                     func=mccormik_function,
                                                     tolerance=0.0001,
                                                     start_x=(-2.5, 3.5),
                                                     learning_rate=(0.01, 0.01))
        self.assertAlmostEqual(x[0], -0.54719, places=1)
        self.assertAlmostEqual(x[1], -1.54719, places=1)
        self.assertAlmostEqual(f, -1.9132, places=4)
