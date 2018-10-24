import unittest

from com.sbk.gradientdescent.gradient_descent import compute_cost_function


class TestComputeCostFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_cost_function_1(self):
        self.assertEqual(compute_cost_function([2, 3], [1, 2, 3, 4, 5], [3, 6, 9, 12, 15]), 2)

    def test_compute_cost_function_2(self):
        self.assertAlmostEqual(compute_cost_function([10, 20], [3, 4, 5], [5, 6, 7]), 3648.3333333333)

    def test_compute_cost_function_3(self):
        v = compute_cost_function([2, 3, 4],
                                  [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7]], [6, 8, 10, 12, 14])
        self.assertAlmostEqual(v, 389.5)

    def test_compute_cost_function_4(self):
        v = compute_cost_function([2, 3, 4, 5],
                                  [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]], [6, 8, 10, 12, 14])
        self.assertAlmostEqual(v, 1452)
