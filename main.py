# -*- coding: utf-8 -*-
import unittest

from neo_scikit_learn_test.linear_model import LinearRegressionTestCase


def suite():
    suite = unittest.TestSuite()
    suite.addTest(LinearRegressionTestCase())
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
