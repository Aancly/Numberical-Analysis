"""
@author: Andy
@data: 2023/1/13 23:05
@description: Newton difference quotient building test
@more: simply Lagrange polynomial and Newton difference quotient polynomial
    It's excited that those polynomials are the same, So they have same polynomial remainder
"""

import numpy as np
import pandas as pd
from data_interpolation.newton_difference_quotient import NewtonDifferenceQuotient

x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
y = np.sin(x)
x0 = np.array([np.pi / 2, 2.158, 3.58, 4.789])
ndq = NewtonDifferenceQuotient(x=x, y=y)


def run_1():
    """ test diff quotient table"""
    tmp_diff_quotient = pd.DataFrame(ndq.__diff_quotient__())
    print("牛顿插值差商表：")
    print(tmp_diff_quotient)


def run_2():
    """test Newton difference quotient fit_interp()"""
    ndq.fit_interp()
    print("牛顿均差插值多项式:")
    print(ndq.polynomial)
    print("牛顿差商插值多项式系数向量与对应阶次")
    print(ndq.poly_coefficient)
    print(ndq.coefficient_order)
    y0 = ndq.cal_interp_x0(x0)
    print("所求插值点的值为：", y0)
    print("所求插值点的精确值为：", np.sin(x0))
    ndq.plt_interpolation()


if __name__ == "__main__":
    run_2()
