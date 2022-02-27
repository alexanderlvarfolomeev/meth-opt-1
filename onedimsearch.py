from abc import abstractmethod

import numpy as np
from numpy import ndarray

from graphic import Graphic


class OneDimensionSearch:
    @abstractmethod
    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray) -> ndarray:
        pass


class StepByStep(OneDimensionSearch):
    def point(self, graphic: Graphic, start: ndarray, rate: float, vector: ndarray) -> ndarray:
        return start - rate * vector
