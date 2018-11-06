import unittest

import numpy as np
import pandas as pd

from com.sbk.gradientdescent.gradient_descent import mini_batch_gradient_descent


class TestStochasticGradientDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_mini_batch_gradient_descent_1(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta, min_cost_value = mini_batch_gradient_descent(data, alpha=0.01, max_iter=1000, batch_size=10)
        self.assertTrue(0.9 < theta[0] < 1.1)
        self.assertTrue(0.4 < theta[1] < 0.53)
        self.assertTrue(0.4 < min_cost_value < 0.6)

    def test_mini_batch_gradient_descent_2(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data1.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta, min_cost_value = mini_batch_gradient_descent(data, alpha=0.01, max_iter=1000, batch_size=10)
        self.assertTrue(0.6 < theta[0] < 1)
        self.assertTrue(-0.3 < theta[1] < 0.05)
        self.assertTrue(20 < min_cost_value < 30)

    def test_mini_batch_gradient_descent_3(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data2.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta, min_cost_value = mini_batch_gradient_descent(data, alpha=0.01, max_iter=1000, batch_size=30)
        np.allclose(theta, np.asarray([0.98531158, 0.90480338, 0.90480338]))
