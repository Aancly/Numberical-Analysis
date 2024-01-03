from data_interpolation.lagrange_interpolation import LagrangeInterpolation
import numpy as np


def run():
    x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    y = np.sin(x)
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.789])

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
    print(np.sin(x0))
    lag_interp.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
