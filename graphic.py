from typing import List
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class Graphic:

    @abstractmethod
    def get(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def grad(self, x: float, y: float) -> ndarray:
        pass

    def __call__(self, x: float, y: float) -> float:
        return self.get(x, y)

    def draw(self):
        fig = plt.figure()
        fig.set_size_inches(20, 20)
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-3.0, 3.0, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)
        plt.show()


class F(Graphic):
    def get(self, x: float, y: float) -> float:
        return x ** 2 + 4 * y ** 2 - 3 * x * y

    def grad(self, x: float, y: float) -> ndarray:
        return np.array([2 * x - 3 * y, 8 * y - 3 * x])


class G(Graphic):
    def get(self, x: float, y: float) -> float:
        return x ** 2 + 4 * y ** 2 - 3 * x * y + 5 * x - y

    def grad(self, x: float, y: float) -> ndarray:
        return np.array([2 * x - 3 * y + 5, 8 * y - 3 * x - 1])
