import matplotlib.pyplot as plt
import numpy as np


def cal_interp_x0(polynomial, x0):
    """
    计算给定的插值点的数值，即插值
    :param polynomial: 插值多项式
    :param x0: 所求插值的x坐标值
    :return y0: 所求插值的y坐标值
    """
    x0 = np.asarray(x0, dtype=np.float_)
    n0 = len(x0)  # 所求插值点的个数
    y0 = np.zeros(n0)  # 存储插值点x0所对应的插值
    t = polynomial.free_symbols.pop()  # 返回值是集合，获取插值多项式的自由变量pop
    for i in range(n0):
        y0[i] = polynomial.evalf(subs={t: x0[i]})
    return y0


def plt_interpolation(params):
    """
    可视化插值图像
    """
    polynomial, x, y, x0, y0, title = params
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "ro", label="Interpolation base points")
    xi = np.linspace(min(x), max(x), 100)  # 模拟100个值
    yi = cal_interp_x0(polynomial, xi)
    plt.plot(xi, yi, "b--", label="Interpolation polynomial")
    if x0 is not None and y0 is not None:
        plt.plot(x0, y0, "g*", label="Interpolation point values")
    plt.legend()
    plt.xlabel("x", fontdict={'fontsize': 12})
    plt.ylabel("y", fontdict={'fontsize': 12})
    plt.title(title + " interpolation polynomial and values", fontdict={'fontsize': 14})
    plt.grid(ls=":")
    plt.show()
