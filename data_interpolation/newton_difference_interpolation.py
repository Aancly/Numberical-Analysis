"""
@author: Andy
@date: 2023/1/14 18:58
@description: Newton Difference Interpolation
"""
import math

import numpy as np
import sympy
import matplotlib.pyplot as plt
from data_interpolation.utils import interp_utils


class NewtonDifferenceInterpolation:

    def __init__(self, x, y, diff_type="forward"):
        """
        牛顿差分插值必要参数的初始化，以及健壮性的检测
        :param x: 已知数据x的坐标点
        :param y: 已知数据y的坐标点
        """
        # 类型转换，数据结构采用array
        self.x = np.asarray(x, dtype=np.float_)
        self.y = np.asarray(y, dtype=np.float_)

        # 数据判断
        if len(self.x) < 1 or len(self.x) != len(self.y):
            raise ValueError("插值数据(x, y)维度不匹配")
        else:
            self.n = len(self.x)  # 已知离散数据点的个数

        # 判读数据点x是否是等距的
        tmp_x = np.linspace(np.min(x), np.max(x), self.n, endpoint=True)
        if not (x == tmp_x).all() and not (x == tmp_x[::-1]).all():
            raise ValueError("非等距节点，不适合用牛顿差分插值")

        """
        弃用：
        d_x = np.diff(x)
        flag_x = d_x[0]-abs(d_x)
        if flag_x.any():
            raise ValueError("非等距节点，不适合用牛顿差分插值")
        else:
            self.h = d_x[0]
            
        弃用原因：计算精度问题，由于舍入误差，两个相同的数相减，可能结果会出现非常小的数而不是0
        而作为Bool型，非0就是1
        
        例:
        >>> x = np.linspace(0, 2 * np.pi, 3, endpoint=True)
        >>> dx = np.diff(x)
        array([2.0943951, 2.0943951, 2.0943951])
        >>> fa = dx[0]-abs(dx)
        array([ 0.0000000e+00,  0.0000000e+00, -4.4408921e-16])
        >>> fa.any()
        True
        """

        self.diff_type = diff_type  # 差分形式：分为 前向forward 后向 backward
        self.polynomial = None  # 最终的插值多项式，符号表示
        self.poly_coefficient = None  # 最终的插值多项式的系数向量，幂从高到低
        self.coefficient_order = None  # 对应多项式系数的阶次
        self.y0 = None  # 所求插值的y坐标值
        self.diff_mat = None  # 存储离散数据点的差分
        self.h = None  # 步长
        self.__x__ = None  # 起始值'

    def __difference_matrix__(self):
        """计算牛顿差分矩阵"""
        self.diff_mat = np.zeros((self.n, self.n))
        self.diff_mat[:, 0] = self.y  # 差分矩阵第一列存y值，即0阶差分
        if self.diff_type == "forward":
            for j in range(1, self.n):
                for i in range(self.n - j):
                    self.diff_mat[i, j] = \
                        self.diff_mat[i + 1, j - 1] - self.diff_mat[i, j - 1]
        elif self.diff_type == "backward":
            for j in range(1, self.n):
                for i in range(j, self.n):
                    self.diff_mat[i, j] = \
                        self.diff_mat[i, j - 1] - self.diff_mat[i - 1, j - 1]
        else:
            raise ValueError("差分形式仅有前向forward和后向backward")

    def fit_interp(self):
        """
        构造牛顿差分插值多项式
        :return:
        """
        self.__difference_matrix__()  # 计算差分矩阵
        self.h = self.x[1] - self.x[0]  # 等距步长的值
        t = sympy.Symbol("t")  # 定义符合变量
        self.polynomial = 0.0  # 插值多项式实例化
        term_poly, fact = 1, 1  # 每一项与t有关的因式、每一项阶乘初始化
        if self.diff_type == "forward":
            self.__x__ = self.x[0]  # 起始值为第1个值
            df = self.diff_mat[0, :]  # 向前差分只用第一行
            for i in range(self.n):
                self.polynomial += df[i] * term_poly / fact
                term_poly *= (t - i)
                fact *= (i + 1)
        elif self.diff_type == "backward":
            self.__x__ = self.x[-1]  # 起始值为第n个值
            db = self.diff_mat[-1, :]  # 向后差分只去最后一行
            for i in range(self.n):
                self.polynomial += db[i] * term_poly / fact
                term_poly *= (t + i)
                fact *= i + 1
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
        t0 = (x0 - self.__x__) / self.h  # 求解t
        self.y0 = interp_utils.cal_interp_x0(self.polynomial, t0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        """
        可视化插值图像
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.x, self.y, "ro", label="Interpolation base points")
        xi = np.linspace(min(self.x), max(self.x), 100)  # 模拟100个值
        yi = self.cal_interp_x0(xi)
        plt.plot(xi, yi, "b--", label="Interpolation polynomial")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "g*", label="Interpolation point values")
        plt.legend()
        plt.xlabel("x", fontdict={'fontsize': 12})
        plt.ylabel("y", fontdict={'fontsize': 12})
        plt.title("Newton " + self.diff_type
                  + " difference interpolation polynomial and values", fontdict={'fontsize': 14})
        plt.grid(ls=":")
        plt.show()







