"""
@author: Andy
@date: 2023/01/15 23:05
@description: test hermite interpolation
"""

import numpy as np
from data_interpolation.hermite_interpolation import HermiteInterpolation


def run():
    x = np.linspace(0, 2 * np.pi, 5)
    y = 2 * np.exp(-x) * np.sin(x)
    dy = 2 * np.exp(-x) * (np.cos(x) - np.sin(x))
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])

    hp = HermiteInterpolation(x, y, dy)
    hp.fit_interp()
    print("插值多项式系数: ", hp.poly_coefficient)
    print("系数所对应的阶数: ", hp.coefficient_order)
    y0 = hp.cal_interp_x0(x0)
    print("插值点的值: ", y0)
    print("精确值: ", 2 * np.exp(-x0) * np.sin(x0))
    hp.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
