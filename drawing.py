import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from graphic import Graphic


def draw1d(linspace, epoches, point=None):
    plt.plot(linspace, epoches)
    if point is not None:
        plt.axvline(linspace[point], color="r")
    plt.show()


def draw2d(x_linspace, y_linspace, epoches, point=None):
    X, Y = np.meshgrid(x_linspace, y_linspace)
    if point is not None:
        plt.plot(x_linspace[point[1]], y_linspace[point[0]], marker="o", markerfacecolor="red")
    plt.contourf(X, Y, epoches)
    plt.show()


def draw_steps(points: ndarray, with_contour: bool, g: Graphic):
    x = points[:, 0]
    y = points[:, 1]
    x_lin = np.linspace(x.min(), x.max(), 100)
    y_lin = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    plt.plot(points[:, 0], points[:, 1], "o-")
    if with_contour:
        plt.contour(X, Y, g(np.array([X, Y])), levels=sorted(
            list(set([g(np.array([p[0], p[1]])) for p in points] + list(np.linspace(-1, 1, 100))))))
    plt.show()
