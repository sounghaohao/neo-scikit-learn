# -*- coding: utf-8 -*-
import unittest

from sklearn import datasets

from neo_scikit_learn.linear_model import LinearRegression


class LinearRegressionTestCase(unittest.TestCase):
    def setUp(self):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.lr = LinearRegression()

    def test_fit(self):
        self.assertEqual(len(self.lr.fit(self.X, self.y)), self.X.shape[1])

    def runTest(self):
        self.test_fit()

    def tearDown(self):
        del self.X
        del self.y
        del self.lr
