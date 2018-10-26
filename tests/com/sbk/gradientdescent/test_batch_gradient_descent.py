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
        self.assertAlmostEqual(theta0, 0.9742431489767754)
        self.assertAlmostEqual(theta1, 0.47458109262638415)

    def test_batch_gradient_descent_2(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data1.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta0, theta1 = batch_gradient_descent(data)
        self.assertAlmostEqual(theta0, 0.871737325073544)
        self.assertAlmostEqual(theta1, -0.09512333650940695)

    def test_batch_gradient_descent_3(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data1.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta0, theta1 = batch_gradient_descent(data, max_iter=1000)
        self.assertAlmostEqual(theta0, 0.8621108429929154)
        self.assertAlmostEqual(theta1, -0.2180970258304527)

    def test_batch_gradient_descent_4(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data2.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta = batch_gradient_descent(data)
        np.allclose(theta, np.asarray([0.98531158, 0.90480338, 0.90480338]))

    def test_batch_gradient_descent_5(self):
        uploaded_data = pd.read_csv(
            'https://raw.githubusercontent.com/SergiySobolev/gradient_descent/master/data/data2.csv',
            delimiter=',')
        data = np.asarray(uploaded_data)
        theta = batch_gradient_descent(data, max_iter=10000)
        np.allclose(theta, np.asarray([0.98531158, 0.90480338, 0.90480338]))
