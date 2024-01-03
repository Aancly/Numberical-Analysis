"""
@author: Andy
@date: 2023/01/19 17:26
@description: finally test cubic_spline_interpolation.py
"""

import numpy as np
from data_interpolation.cubic_spline_interpolation import CubicSplineInterpolation


def run():
    x0 = np.arange(0, 10.25, 0.25, )
    print(x0)
    x = np.array([0, 1, 2.5, 3.6, 5, 7, 8.1, 10])
    y = np.sin(x)

    csi = CubicSplineInterpolation(x, y, "periodic", None)
    csi.fit_interp()
    print("插值多项式系数:\n", csi.poly_coefficient)
    y0 = csi.cal_interp_x0(x0)
    print("插值点的值: ", y0)
    print("精  确  值: ", np.sin(x0))
    csi.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
