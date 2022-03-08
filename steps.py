from abc import abstractmethod

import numpy as np
from numpy import ndarray

from graphic import Graphic
from onedimsearch import WolfeCondition, StepByStep


class StepScheduler:
    @abstractmethod
    def step(self, i: int, graphic: Graphic, start: ndarray, vector: ndarray) -> float:
        pass


class RateStep(StepScheduler):

    def __init__(self, step: float):
        self.rateStep = step

    def step(self, i: int, graphic: Graphic, start: ndarray, vector: ndarray) -> float:
        return self.rateStep


class LinearStep(StepScheduler):
    def __init__(self, start: float, step: float):
        self.rateStep = step
        self.start = start
        self.current = start

    def step(self, i: int, graphic: Graphic, start: ndarray, vector: ndarray) -> float:
        if self.current >= 0.01:
            self.current *= self.rateStep
        return self.current


class ParabolicStep(StepScheduler):

    def __init__(self, k: float):
        self.k = k

    def step(self, i: int, graphic: Graphic, start: ndarray, vector: ndarray) -> float:
        return max(self.k / (i ** 2), 0.01)


class WolfSteps(StepScheduler):

    def __init__(self, c_1=None, c_2=None):
        if c_1 is None:
            c_1 = 0.1
        if c_2 is None:
            c_2 = 0.9
        self.c_1 = c_1
        self.c_2 = c_2
        self.stepByStep = StepByStep()

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

    def step(self, i: int, graphic: Graphic, start: ndarray, vector: ndarray) -> float:
        return self.calculate_step(graphic, vector, start)