"""
@author: Andy
@date: 2023/01/19 16:49
@description: cubic spline interpolation
"""

import numpy as np
import sympy
import warnings
from data_interpolation.utils import piecewise_utils


class CubicSplineInterpolation:
    """
    三次样条插值（Cubic Spline Interpolation）简称Spline插值。
    是通过一系列形值点的一条光滑曲线，
    数学上通过求解三弯矩方程组得出曲线函数组的过程。
    """

    def __init__(self, x, y, boundary_type="natural", boundary_values=None):
        """
        三次样条插值必要参数的初始化，以及健壮性的检测
        :param x: 已知数据x的坐标点
        :param y: 已知数据y的坐标点
        :param boundary_type: 边界类型
        :param boundary_values: 边界条件值
        """
        # 类型转换，数据结构采用array
        self.x = np.asarray(x, dtype=np.float_)
        self.y = np.asarray(y, dtype=np.float_)
        # 数据判断
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)  # 已知离散数据点的个数
        else:
            raise ValueError("插值数据(x, y）维度不匹配")

        if boundary_values is None:
            self.boundary_values = None
        elif len(boundary_values) == 2:
            self.boundary_values = np.asarray(boundary_values, dtype=np.float_)  # 边界条件
        else:
            raise ValueError("边界条件不合规")

        self.boundary_type = boundary_type  # 边界条件类型，默认自然边界条件
        self.polynomial = None  # 最终插值多项式，符号表示
        self.poly_coefficient = None  # 最终插值多项式的系数向量，幂从高到低
        self.y0 = None  # 所求插值的y坐标值
        self.__boundary_function = {  # 不同边界条件所对应的函数
            "complete": self.__complete_spline,
            "second": self.__second_spline,
            "natural": self.__natural_spline,
            "periodic": self.__periodic_spline,
        }

    def fit_interp(self):
        """
        生成三次样条插值多项式
        :return:
        """
        t = sympy.Symbol("t")
        self.polynomial = dict()
        self.poly_coefficient = np.zeros((self.n - 1, 4))  # 系数矩阵
        if self.boundary_type in self.__boundary_function.keys():  # 判断键中是否有传入的类型
            print("boundary type: " + str(self.boundary_type))
            moment = self.__boundary_function[self.boundary_type](self.x, self.y, self.boundary_values)
        else:
            raise ValueError("错误的边界类型，'" + str(self.boundary_type) +
                             "' 类型不属于 set: {'complete', 'second', 'natural', 'periodic'}")

        self.spline_poly(t, self.x, self.y, moment)  # 构造多项式

    @staticmethod
    def __complete_spline(x, y, boundary_values):
        """
        求解第一类边界条件
        :param x: 已知数据点x的坐标
        :param y: 已知数据点y的坐标
        :param boundary_values: 边界条件(dy)
        :return: 返回三弯矩方程组的解，即多项式的矩
        """
        n = len(x)  # 重新计算长度（性能什么的无所谓，编译器会出手）
        mat_a = np.diag(2 * np.ones(n))  # 方程组系数矩阵
        c = np.zeros(n)
        for i in range(1, n - 1):  # 从1开始循环到n-2 (n-2 times)
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            u = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c[i] = (
                    3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) +
                    3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            )
            mat_a[i, i - 1], mat_a[i, i + 1] = lambda_, u
        # 边界条件应用，np.array类型，list类型不行，如：2*[1,2] is [1,2,1,2]
        c[0], c[-1] = 2 * boundary_values[[0, -1]]

        m = np.linalg.solve(mat_a, c)
        return m

    @staticmethod
    def __second_spline(x, y, boundary_values):
        """
        求解第二类边界条件
        :param x: 已知数据点x的坐标
        :param y: 已知数据点y的坐标
        :param boundary_values: 边界条件(d2y)
        :return: 返回三弯矩方程组的解，即多项式的矩
        """
        n = len(x)  # 重新计算长度（性能什么的无所谓，编译器会出手）
        mat_a = np.diag(2 * np.ones(n))  # 方程组系数矩阵
        mat_a[0, 1], mat_a[-1, -2] = 1, 1  # 边界的特殊情况
        c = np.zeros(n)
        for i in range(1, n - 1):  # 从1开始循环到n-2 (n-2 times)
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            u = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c[i] = (
                    3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) +
                    3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            )
            mat_a[i, i - 1], mat_a[i, i + 1] = lambda_, u
        # 边界条件应用
        c[0] = 3 * (y[1] - y[0]) / (x[1] - x[0]) - (x[1] - x[0]) * boundary_values[0] / 2
        c[-1] = 3 * (y[-1] - y[-2]) / (x[-1] - x[-2]) - (x[-2] - x[-1]) * boundary_values[-1] / 2

        m = np.linalg.solve(mat_a, c)
        return m

    def __natural_spline(self, x, y, boundary_values):
        """
        求解自然边界条件
        :param x: 已知数据点x的坐标
        :param y: 已知数据点y的坐标
        :param boundary_values: 自然边界条件不需要参数(None)
        :return: 返回三弯矩方程组的解，即多项式的矩
        """
        if boundary_values is not None:
            warnings.warn("自然边界类型无需边界值")
        boundary_values = np.array([0, 0], dtype=np.float_)
        return self.__second_spline(x, y, boundary_values)

    @staticmethod
    def __periodic_spline(x, y, boundary_values):
        """
        求解第三类（周期）边界条件 ***感觉有些问题，先挖个坑，估计不会填了***
        :param x: 已知数据点x的坐标
        :param y: 已知数据点y的坐标
        :param boundary_values: 周期边界无需另加边界条件，只是为了形式上的美观
        :return:
        """
        # 周期类型无需另加边界条件
        if boundary_values is not None:
            warnings.warn("自然边界类型无需边界值")
        n = len(x)  # 重新计算长度（性能什么的无所谓，编译器会出手）
        mat_a = np.diag(2 * np.ones(n - 1))  # 方程组系数矩阵

        # 边界特殊情况
        h0, h1, he = x[1] - x[0], x[2] - x[1], x[-1] - x[-2]
        # 系数矩阵特殊情况
        mat_a[0, 1] = h0 / (h0 + h1)  # 表示u_1
        mat_a[0, -1] = 1 - mat_a[0, 1]  # 表示lambda_1
        mat_a[-1, 0] = he / (he + h0)  # 表示u_n
        mat_a[-1, -2] = 1 - mat_a[-1, 0]  # 表示lambda_n
        # 向量c的边界特殊情况
        c = np.zeros(n - 1)  # 向量c初始化
        c[-1] = 3 * (he * (y[1] - y[0]) / h0 + h0 * (y[-1] - y[-2]) / he) / (he + h0)

        for i in range(1, n - 1):  # 从1开始循环到n-2 (n-2 times)
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            u = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])
            c[i - 1] = (  # 数学中c是从1开始，python是从0开始，在这里发生转换（值存储位置）
                    3 * lambda_ * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) +
                    3 * u * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            )
            if i < n - 2:  # 避免最后发生越界
                mat_a[i, i - 1], mat_a[i, i + 1] = lambda_, u

        m = np.zeros(n)
        m[1:] = np.linalg.solve(mat_a, c)
        m[0] = m[-1]  # 最后一个系数值赋予第一个（周期条件）
        return m

    def spline_poly(self, t, x, y, m):
        """
        构造三次样条多项式
        :param t: 符号变量
        :param x: 已知数据点x的坐标
        :param y: 已知数据点y的坐标
        :param m: 多项式的矩，三弯矩方程组的解
        :return:
        """
        for i in range(self.n - 1):
            hi = x[i + 1] - x[i]  # 子区间长度
            pi = (
                    y[i] / hi ** 3 * (2 * (t - x[i]) + hi) * (x[i + 1] - t) ** 2 +
                    y[i + 1] / hi ** 3 * (2 * (x[i + 1] - t) + hi) * (t - x[i]) ** 2 +
                    m[i] / hi ** 2 * (t - x[i]) * (x[i + 1] - t) ** 2 -
                    m[i + 1] / hi ** 2 * (x[i + 1] - t) * (t - x[i]) ** 2
            )
            self.polynomial[i] = sympy.simplify(pi)
            poly_obj = sympy.Poly(pi, t)  # 根据多项式构造多项式对象
            # 某项系数可能为0， 为防止存储错误，分别对应各阶次存储
            mons = poly_obj.monoms()  # 多项式系数对应的阶次
            for j in range(len(mons)):
                self.poly_coefficient[i, mons[j][0]] = poly_obj.coeffs()[j]  # 获取多项式的系数

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
                  str(self.boundary_type) + " boundary Cubic spline", x0, y0)
        piecewise_utils.plt_interpolation(params)
