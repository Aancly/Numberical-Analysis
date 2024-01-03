"""
@author: Andy
@date: 2023/01/17 18:43
@description: piece 2 points and cubic(3 degree) hermite interpolation polynomial
"""

import numpy as np
import sympy
from data_interpolation.utils import piecewise_utils


class Piece2P3DHermiteInterpolation:
    """
    分段两点三次埃尔米特插值多项式
    """

    def __init__(self, x, y, dy):
        """
        分段两点三次埃尔米特插值必要参数的初始化，以及健壮性的检测
        :param x: 已知数据x的坐标点
        :param y: 已知数据y的坐标点
        :param dy: 已知数据点的导数值
        """
        # 类型转换，数据结构采用array
        self.x = np.asarray(x, dtype=np.float_)
        self.y = np.asarray(y, dtype=np.float_)
        self.dy = np.asarray(dy, dtype=np.float_)

        # 数据判断
        if len(self.x) > 1 and len(self.x) == len(self.y) and len(self.dy) == len(self.dy):
            self.n = len(self.x)  # 已知离散数据点的个数
        else:
            raise ValueError("插值数据(x, y, dy)维度不匹配")

        self.polynomial = None  # 最终的插值多项式，符号表示
        self.poly_coefficient = None  # 最终的插值多项式的系数向量，幂从高到低
        self.y0 = None  # 所求插值的y坐标值

    def fit_interp(self):
        """
        生成分段两点三次埃尔米特插值多项式
        :return:
        """
        t = sympy.Symbol("t")
        self.polynomial = dict()  # 三次多项式
        self.poly_coefficient = np.zeros((self.n - 1, 4))  # 系数矩阵
        for i in range(self.n - 1):
            hi = self.x[i + 1] - self.x[i]  # 每个小区间的步长
            # 分段两点三次埃尔米特插值函数公式
            poly23_i = (
                    self.y[i] * (1 + 2 * (t - self.x[i]) / hi) * ((t - self.x[i + 1]) / hi) ** 2 +
                    self.y[i + 1] * (1 - 2 * (t - self.x[i + 1]) / hi) * ((t - self.x[i]) / hi) ** 2 +
                    self.dy[i] * (t - self.x[i]) * ((t - self.x[i + 1]) / hi) ** 2 +
                    self.dy[i + 1] * (t - self.x[i + 1]) * ((t - self.x[i]) / hi) ** 2
            )
            self.polynomial[i] = sympy.simplify(poly23_i)
            poly23_obj = sympy.Poly(poly23_i, t)  # 根据多项式构造多项式对象
            # 某项系数可能为0， 为防止存储错误，分别对应各阶次存储
            mons = poly23_obj.monoms()  # 多项式系数对应的阶次
            for j in range(len(mons)):
                self.poly_coefficient[i, mons[j][0]] = poly23_obj.coeffs()[j]  # 获取多项式的系数

    def cal_interp_x0(self, x0):
        """
        计算所给定的插值点的数值
        :param x0:
        :return:
        """
        self.y0 = piecewise_utils.cal_interp_x0(self.polynomial, self.x, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        """可视化插值图像和所求的插值点"""
        params = (self.polynomial, self.x, self.y,
                  "Piecewise 2 points Cubic hermite", x0, y0)
        piecewise_utils.plt_interpolation(params)
