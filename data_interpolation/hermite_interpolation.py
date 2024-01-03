"""
@author: Andy
@date: 2023/01/15 23:00
@description: hermite interpolation with first derivative
"""

import numpy as np
import sympy
from data_interpolation.utils import interp_utils


class HermiteInterpolation:
    """solve Hermite interpolation with first derivative"""

    def __init__(self, x, y, dy):

        """
        hermite 必要参数的初始化，以及健壮性的检测
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
        self.coefficient_order = None  # 对应多项式系数的阶次
        self.y0 = None  # 所求插值的y坐标值

    def fit_interp(self):
        """
        Hermite interpolation 核心算法
        :return:
        """

        t = sympy.Symbol("t")  # 符合变量
        self.polynomial = 0.0  # 多项式初始化
        for i in range(self.n):
            hi, ai = 1.0, 0.0  # 插值多项式辅助“函数”构造（☞数学中的函数）
            for j in range(self.n):
                if j != i:
                    hi *= ((t - self.x[j]) / (self.x[i] - self.x[j])) ** 2
                    ai += 1 / (self.x[i] - self.x[j])
            self.polynomial += \
                hi * ((self.x[i] - t) * (2 * ai * self.y[i] - self.dy[i]) + self.y[i])

        self.polynomial = sympy.expand(self.polynomial)  # 多项式展开
        polynomial = sympy.Poly(self.polynomial)  # 根据多项式构造多项式对象
        self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
        self.coefficient_order = polynomial.monoms()  # 多项式系数对应的阶次

    def cal_interp_x0(self, x0):
        """
        计算给定的插值点的数值，即插值
        :param x0: 所求插值的x坐标值
        :return  : 所求插值的y坐标值
        """
        self.y0 = interp_utils.cal_interp_x0(self.polynomial, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        """
        可视化插值图像
        """
        params = (self.polynomial, self.x, self.y, x0, y0, "Hermite")
        interp_utils.plt_interpolation(params)
