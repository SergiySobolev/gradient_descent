import unittest

import numpy as np

from com.sbk.func.func_1 import func_2, build_table_for_multi_variable_func


class TestGradientDescentSingleVariable(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_find_minimum_with_default_start_approximation(self):
        t1 = np.arange(-5, 15, 0.1)
        t2 = np.arange(-5, 15, 0.1)
        table = build_table_for_multi_variable_func(t1, t2, func_2)
        print(table)
