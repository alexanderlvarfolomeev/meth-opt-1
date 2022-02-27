from abc import abstractmethod


class StepScheduler:
    @abstractmethod
    def step(self, i: int) -> float:
        pass


class RateStep(StepScheduler):

    def __init__(self, step: float):
        self.rateStep = step

    def step(self, i: int) -> float:
        return self.rateStep


class LinearStep(StepScheduler):
    def __init__(self, start: float, step: float):
        self.rateStep = step
        self.start = start
        self.current = start

    def step(self, i: int) -> float:
        if self.current >= 0.01:
            self.current *= self.rateStep
        return self.current


class ParabolicStep(StepScheduler):

    def __init__(self, k: float):
        self.k = k

    def step(self, i: int) -> float:
        return max(self.k / (i ** 2), 0.01)
