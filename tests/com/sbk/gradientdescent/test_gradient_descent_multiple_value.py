import unittest

import numpy as np

from com.sbk.func.func_1 import func_2


class TestGradientDescentMultipleVariable(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum_with_default_start_approximation(self):
        t1 = np.arange(-15, 10, 0.5)
        t2 = np.arange(-15, 10, 0.5)
        x, y = np.meshgrid(t1, t2)
        z = func_2(x, y)
        print(z)
