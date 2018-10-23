import unittest

import numpy as np
import pandas as pd

from com.sbk.gradientdescent.gradient_descent import batch_gradient_descent


class TestBatchGradientDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_batch_gradient_descent_1(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta0, theta1 = batch_gradient_descent(data)
        self.assertAlmostEqual(theta0, 0.9665143757)
        self.assertAlmostEqual(theta1, 0.47481285077)
