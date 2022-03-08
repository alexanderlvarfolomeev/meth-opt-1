from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class Graphic:

    def __init__(self):
        self.count = 0

    @abstractmethod
    def dimension(self):
        pass

    @abstractmethod
    def get(self, x: ndarray) -> float:
        pass

    @abstractmethod
    def grad(self, x: ndarray) -> ndarray:
        pass

    def clear(self):
        self.count = 0

    def __call__(self, x: ndarray) -> float:
        self.count += 1
        return self.get(x)

    # Supports only 3d
    def draw(self):
        fig = plt.figure()
        # fig.set_size_inches(20, 20)
        ax = fig.add_subplot(111, projection='3d')
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


class F2(Graphic):
    def dimension(self):
        return 3

    def get(self, x: ndarray) -> float:
        return np.sin(0.5 * x[0] ** 2 - 0.25 * x[1] ** 2 + 3) * np.cos(2 * x[0] + 1 - np.exp(x[1]))

    def grad(self, x: ndarray) -> ndarray:
        h = 1e-5
        return np.array([(self.get(np.array([x[0] + h, x[1]])) - self.get(np.array([x[0] - h, x[1]]))) / (2 * h),
                         (self.get(np.array([x[0], x[1] + h])) - self.get(np.array([x[0], x[1] - h]))) / (2 * h)])


class RandomGraphic(Graphic):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        random_matrix = np.random.rand(n, n)
        self.matrix = np.dot(random_matrix, np.transpose(random_matrix))
        self.dmatrix = self.matrix + np.transpose(self.matrix)

    def dimension(self):
        return self.n

    def get(self, x: ndarray) -> float:
        res = np.sum(np.tensordot(self.matrix, x, axes=1) * x, axis=0)
        return res.item() if res.shape == () else res

    def grad(self, x: ndarray) -> ndarray:
        return np.dot(self.dmatrix, x)


