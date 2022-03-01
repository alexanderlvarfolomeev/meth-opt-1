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