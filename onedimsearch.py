from abc import abstractmethod

import numpy as np
from numpy import ndarray

from graphic import Graphic


class OneDimensionSearch:
    @abstractmethod
    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray, eps: float) -> ndarray:
        pass


class StepByStep(OneDimensionSearch):
    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray, eps: float) -> ndarray:
        return start - rate * vector


class GoldSplit(OneDimensionSearch):

    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray, eps: float) -> ndarray:
        l = start
        r = start - rate * vector
        phi = (1 + (5 ** 0.5)) / 2
        dphi = 2 - phi
        x1 = l + dphi * (r - l)
        x2 = r - dphi * (r - l)
        f1 = graphic(x1)
        f2 = graphic(x2)
        while np.linalg.norm(r - l) > eps:
            if f1 > f2:
                l = x1
                x1 = x2
                f1 = f2
                x2 = r - dphi * (r - l)
                f2 = graphic(x2)
            else:
                r = x2
                x2 = x1
                f2 = f1
                x1 = l + dphi * (r - l)
                f1 = graphic(x1)
        return (x1 + x2) / 2


class WolfeCondition(OneDimensionSearch):

    def __init__(self, c_1=None, c_2=None):
        if c_1 is None:
            c_1 = 0.1
        if c_2 is None:
            c_2 = 0.9
        self.c_1 = c_1
        self.c_2 = c_2

    def mul_vectors(
            a1: ndarray,
            a2: ndarray
    ) -> float:
        return np.matmul(a2[np.newaxis, :], a1[:, np.newaxis])[0][0]

    def first_condition(
            f: Graphic,
            p_k: ndarray,
            x_k: ndarray,
            grad_f_k: ndarray,
            a_k: float,
            c_1: float,
    ) -> bool:
        b = WolfeCondition.mul_vectors(grad_f_k, p_k)
        return f.get(x_k + a_k * p_k) <= f(x_k) + c_1 * a_k * b

    def second_condition(
            f: Graphic,
            p_k: ndarray,
            x_k: ndarray,
            grad_f_k: ndarray,
            a_k: float,
            c_2: float,
    ) -> bool:
        b = WolfeCondition.mul_vectors(grad_f_k, p_k)
        a = WolfeCondition.mul_vectors(f.grad(x_k + a_k * p_k), p_k)
        return a >= c_2 * b

    def calculate_step(self,
                       f: Graphic,
                       gradient: ndarray,
                       x_k: ndarray
                       ) -> float:
        anti_gradient = -1 * gradient
        a_k = 1e-7

        while not WolfeCondition.first_condition(
                f, anti_gradient, x_k, gradient, a_k, self.c_1
        ) or not WolfeCondition.second_condition(f, anti_gradient, x_k,
                                                 gradient, a_k, self.c_2):
            a_k *= 2
        return a_k

    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray, eps: float) -> ndarray:
        step = self.calculate_step(graphic, vector, start)
        return StepByStep().point(graphic, start, step, vector, eps)
