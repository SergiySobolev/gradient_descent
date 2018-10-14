import unittest
import numpy as np

from com.sbk.gradientdescent.gradient_descent import gradient_descent


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum(self):
        l = -3
        r = 7
        n = 1000

        x = np.linspace(start=l, stop=r, num=n)
        y = (x - 2) ** 2

        v = gradient_descent(x,y)
        self.assertEqual(v, 10)