from typing import List
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class Graphic:

    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def get(self, x: ndarray) -> float:
        pass

    @abstractmethod
    def grad(self, x: ndarray) -> ndarray:
        pass

    def __call__(self, x: ndarray) -> float:
        return self.get(x)

    # Supports only 3d
    def draw(self):
        fig = plt.figure()
        fig.set_size_inches(20, 20)
        dim_string = str(self.dimension()) + "d"
        ax = fig.add_subplot(111, projection=dim_string)
        x = y = np.arange(-3.0, 3.0, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)
        plt.show()


class F(Graphic):

    def dimension(self):
        return 3

    def get(self, x: ndarray) -> float:
        return x[0] ** 2 + 4 * x[1] ** 2 - 3 * x[0] * x[1]

    def grad(self, x: ndarray) -> ndarray:
        return np.array([2 * x[0] - 3 * x[1], 8 * x[1] - 3 * x[0]])


class G(Graphic):

    def dimension(self):
        return 3

    def get(self, x: ndarray) -> float:
        return x[0] ** 2 + 4 * x[1] ** 2 - 3 * x[0] * x[1] + 5 * x[0] - x[1]

    def grad(self, x: ndarray) -> ndarray:
        return np.array([2 * x[0] - 3 * x[1] + 5, 8 * x[1] - 3 * x[0] - 1])
