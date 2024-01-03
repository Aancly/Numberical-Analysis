"""
@author: Andy
@data: 2023/1/13 22:44
@description: Newton difference quotient interpolation
@more: It's better that we should write a new function to add new points has
    time complexity of (O(n)), but It's hard to make it,
    all in all, The ideal and the reality is always a gap.
"""

import numpy as np
import sympy
from data_interpolation.utils import interp_utils


class NewtonDifferenceQuotient:
    """
    牛顿差商（均差）插值算法
    """

    def __init__(self, x, y):
        """
        牛顿差商插值必要参数的初始化，以及健壮性的检测
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

        #
        self.polynomial = None  # 最终的插值多项式，符号表示
        self.poly_coefficient = None  # 最终的插值多项式的系数向量，幂从高到低
        self.coefficient_order = None  # 对应多项式系数的阶次
        self.y0 = None  # 所求插值的y坐标值
        self.diff_quot = None  # 存储离散数据点的差商

    def __diff_quotient__(self):
        """计算牛顿差商（均差）"""
        diff_quot = np.zeros((self.n, self.n))  # 存储差商矩阵
        diff_quot[:, 0] = self.y  # 差商表中第一列存储y值
        for j in range(1, self.n):  # 按列计算
            for i in range(j, self.n):  # 行，起始位置为差商表中对角线
                diff_quot[i, j] = (diff_quot[i, j - 1] - diff_quot[i - 1, j - 1]) \
                                  / (self.x[i] - self.x[i - j])
        self.diff_quot = diff_quot
        return diff_quot

    def fit_interp(self):
        """
        核心算法：生成Newton difference quotient interpolation polynomial
        """
        self.__diff_quotient__()  # 计算差商
        tmp_dp = np.diag(self.diff_quot)  # 构造多项式只需要对角线的值即可
        # 数值运算，符号运算
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = 0.0  # 插值多项式实例化
        term_poly = 1  # 初始牛顿多项式每一项
        for i in range(self.n):
            self.polynomial += tmp_dp[i] * term_poly
            term_poly *= t - self.x[i]

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
        params = (self.polynomial, self.x, self.y, x0, y0, "Newton difference quotient")
        interp_utils.plt_interpolation(params)
