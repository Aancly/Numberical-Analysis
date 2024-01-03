"""
@author: Andy
@date: 2023/01/16 11:25
@description:
"""

import numpy as np
import sympy
from data_interpolation.utils import piecewise_utils


class PiecewiseLinearInterpolation:
    """
    分段线性插值算法实现
    """

    def __init__(self, x, y):
        """
        分段线性插值函数必要参数的初始化，以及健壮性的检测
        :param x: 已知数据x的坐标点
        :param y: 已知数据y的坐标点
        """
        # 类型转换，数据结构采用array
        self.x = np.asarray(x, dtype=np.float_)
        self.y = np.asarray(y, dtype=np.float_)

        # 数据判断
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)  # 已知离散数据点的个数
        else:
            raise ValueError("插值数据(x, y)维度不匹配")

        self.linear_poly = None  # 分段线性插值
        self.linear_coefficient = None  # 插值多项式的系数矩阵
        self.y0 = None  # 所求插值的y坐标值

    def fit_interp(self):
        """
        生成分段线性插值多项式
        :return:
        """
        t = sympy.Symbol("t")
        self.linear_poly = dict()  # 线性函数
        self.linear_coefficient = np.zeros((self.n - 1, 2))  # 系数矩阵
        for i in range(self.n - 1):
            hi = self.x[i + 1] - self.x[i]  # 每个小区间的步长
            # 分段线性插值函数
            linear_i = self.y[i + 1] * (t - self.x[i]) / hi - self.y[i] * (t - self.x[i + 1]) / hi
            self.linear_poly[i] = sympy.simplify(linear_i)
            linear_poly_obj = sympy.Poly(linear_i, t)  # 根据多项式构造多项式对象
            # 某项系数可能为0， 为防止存储错误，分别对应各阶次存储
            mons = linear_poly_obj.monoms()  # 多项式系数对应的阶次
            for j in range(len(mons)):
                self.linear_coefficient[i, mons[j][0]] = linear_poly_obj.coeffs()[j]  # 获取多项式的系数

    def cal_interp_x0(self, x0):
        """
        计算所给定的插值点的数值
        :param x0:
        :return:
        """
        self.y0 = piecewise_utils.cal_interp_x0(self.linear_poly, self.x, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        """可视化插值图像和所求的插值点"""
        params = (self.linear_poly, self.x, self.y,
                  "piecewise linear", x0, y0)
        piecewise_utils.plt_interpolation(params)
