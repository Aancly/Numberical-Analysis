"""
@author: Andy
@date: 2023/01/17 19:21
@description: final test for piecewise 2 points cubic hermite interpolation
"""

import numpy as np
from data_interpolation.piece_2p3d_hermite_interpolation import Piece2P3DHermiteInterpolation


def run():
    x = np.linspace(0, 2 * np.pi, 10)
    y = 2 * np.exp(-x) * np.sin(x)
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])
    dy = 2 * np.exp(-x) * (np.cos(x) - np.sin(x))

    p23h = Piece2P3DHermiteInterpolation(x, y, dy)
    p23h.fit_interp()
    print("插值多项式系数:\n", p23h.poly_coefficient)
    y0 = p23h.cal_interp_x0(x0)
    print("插值点的值: ", y0)
    print("精  确  值: ", 2 * np.exp(-x0) * np.sin(x0))
    p23h.plt_interpolation(x0, y0)


if __name__ == "__main__":
    run()
