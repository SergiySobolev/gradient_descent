import unittest

import numpy as np

from com.sbk.gradientdescent.gradient_descent import compute_cost_function, compute_gradient


class TestComputeCostFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_cost_function_1(self):
        v = compute_cost_function(np.asarray([2, 3]),
                                  np.transpose([[1, 2, 3, 4, 5]]),
                                  np.asarray([3, 6, 9, 12, 15]))
        self.assertAlmostEqual(v, 2)

    def test_compute_cost_function_2(self):
        v = compute_cost_function(np.asarray([10, 20]),
                                  np.transpose([[3, 4, 5]]),
                                  np.asarray([5, 6, 7]))
        self.assertAlmostEqual(v, 3648.3333333333)

    def test_compute_cost_function_3(self):
        v = compute_cost_function(np.asarray([2, 3, 4]),
                                  np.transpose([[3, 4, 5, 6, 7], [3, 4, 5, 8, 7]]),
                                  np.asarray([6, 8, 10, 12, 14]))
        self.assertAlmostEqual(v, 447.1)

    def test_compute_cost_function_4(self):
        v = compute_cost_function(np.asarray([2, 3, 4, 5]),
                                  np.transpose([[3, 4, 5, 6, 7], [3, 4, 5, 8, 7], [3, 4, 5, 3, 7]]),
                                  np.asarray([6, 8, 10, 12, 14]))
        self.assertAlmostEqual(v, 1370.1)

    def test_compute_gradient_2(self):
        v = compute_gradient(np.asarray([2, 3, 4]),
                             np.transpose([[3, 4, 5, 6, 7], [3, 4, 5, 6, 7]]),
                             np.asarray([6, 8, 10, 12, 14]))
        np.allclose(v, np.asarray([27.0, 145.0, 145.0]))

    def test_compute_gradient_3(self):
        v = compute_gradient(np.asarray([2, 3, 4, 5]),
                             np.transpose([[3, 4, 5, 6, 7], [3, 4, 5, 8, 7], [3, 4, 5, 3, 7]]),
                             np.asarray([6, 8, 10, 12, 14]))
        np.allclose(v, np.asarray([50.6, 271.6, 293.6, 238.6]))
