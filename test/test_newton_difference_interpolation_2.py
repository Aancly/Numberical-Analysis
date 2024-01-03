"""
@author : Andy
@date: 2023/1/14 22:13
@description: Test diff_mat for forward and backward
"""

import numpy as np
from data_interpolation.newton_difference_interpolation import NewtonDifferenceInterpolation

x = np.linspace(0, 2 * np.pi, 5, endpoint=True)
y = np.sin(x)
x0 = np.array([np.pi / 2, 2.158, 3.58, 4.789])


def test_f():
    ndi_f = NewtonDifferenceInterpolation(x, y, "forward")
    ndi_f.__difference_matrix__()
    print(ndi_f.diff_mat)


def test_b():
    ndi_b = NewtonDifferenceInterpolation(x, y, "backward")
    ndi_b.__difference_matrix__()
    print(ndi_b.diff_mat)


if __name__ == "__main__":
    test_b()
