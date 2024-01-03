"""
@author : Andy
@date: 2023/1/14 19:20
@description: Test judge x is equidistant in the file
    named newton_difference_interpolation_py
"""

import numpy as np
from data_interpolation.newton_difference_interpolation import NewtonDifferenceInterpolation


def test_1():
    xx = np.linspace(0, 2 * np.pi, 3, endpoint=True)
    yy = np.sin(xx)
    return xx, yy


def test_2():
    xx = [1, 2, 3, 4, 5]
    yy = [1, 1, 1, 1, 1]
    return xx[::-1], yy


if __name__ == "__main__":
    x, y = test_2()
    print(x)
    ndi = NewtonDifferenceInterpolation(x, y)
