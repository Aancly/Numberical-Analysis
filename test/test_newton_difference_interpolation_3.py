"""
@author : Andy
@date: 2023/1/14 22:13
@description: finally test
"""

import numpy as np
from data_interpolation.newton_difference_interpolation import NewtonDifferenceInterpolation


def run():
    """test Newton difference interpolation"""
    x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    y = np.sin(x)
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])
    ndi = NewtonDifferenceInterpolation(x, y, "forward")
    ndi.fit_interp()
    print("牛顿差分插值多项式:")
    print(ndi.polynomial)
    print("牛顿差分插值多项式系数向量与对应阶次")
    print(ndi.poly_coefficient)
    print(ndi.coefficient_order)
    y0 = ndi.cal_interp_x0(x0)
    print("所求插值点的值为：", y0)
    print("所求插值点的精确值为：", np.sin(x0))
    ndi.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
