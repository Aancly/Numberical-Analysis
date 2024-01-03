from data_interpolation.lagrange_interpolation import LagrangeInterpolation


def run():
    x = [0.32, 0.34, 0.36]
    y = [0.314567, 0.333487, 0.352274]
    x0 = [0.3367]

    lag_interp_1 = LagrangeInterpolation(x=x[0:2], y=y[0:2])
    lag_interp_1.fit_interp()
    lag_interp_2 = LagrangeInterpolation(x=x, y=y)
    lag_interp_2.fit_interp()
    y0_1 = lag_interp_2.cal_interp_x0(x0)
    y0_2 = lag_interp_2.cal_interp_x0(x0)
    print("线性插值计算sin(x0)=" + str(y0_1))
    print("抛物线插值计算sin(x0)=" + str(y0_2))
    lag_interp_1.plt_interpolation(x0, y0_1)
    lag_interp_2.plt_interpolation(x0, y0_2)


if __name__ == "__main__":
    run()
