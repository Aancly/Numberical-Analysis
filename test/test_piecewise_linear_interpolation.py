"""
@author: Andy
@date: 2023/01/17 17:55
@description: final test for piecewise linear interpolation
"""

import numpy as np
from data_interpolation.piecewise_linear_interpolation import PiecewiseLinearInterpolation


def run():
    x = np.linspace(0, 2 * np.pi, 100)
    y = 2 * np.exp(-x) * np.sin(x)
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])

    pli = PiecewiseLinearInterpolation(x, y)
    pli.fit_interp()
    print("插值多项式系数:\n", pli.linear_coefficient)
    y0 = pli.cal_interp_x0(x0)
    print("插值点的值: ", y0)
    print("精确值: ", 2 * np.exp(-x0) * np.sin(x0))
    pli.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
