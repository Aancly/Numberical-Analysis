"""
@author: Andy
@date: 2023/1/13
@description: Runge Phenomenon
"""
from data_interpolation.lagrange_interpolation import LagrangeInterpolation
import numpy as np


def run():
    n = 10  # 节点个数
    k = np.arange(0, n + 1)
    x = -5 + 10 * k / n
    y = 1 / (1 + x ** 2)
    x0 = np.array([4.8])
    lag_interp = LagrangeInterpolation(x=x, y=y)
    lag_interp.fit_interp()
    y0 = lag_interp.cal_interp_x0(x0)
    print("Lagrange Interpolation:")
    print(lag_interp.polynomial)
    print("Lagrange Interpolation Coefficient and Order:")
    print(lag_interp.poly_coefficient)
    print(lag_interp.coefficient_order)
    print("所求插值点的值为：")
    print(y0)
    print("所求插值点的精确值为：")
    print(1 / (1 + x0 ** 2))
    lag_interp.plt_interpolation(x0, y0)


if __name__ == '__main__':
    run()
