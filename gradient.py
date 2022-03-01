import numpy as np
from numpy import ndarray

from graphic import Graphic
from onedimsearch import OneDimensionSearch
from steps import StepScheduler, RateStep, LinearStep, ParabolicStep


def gradient_steps(
        function: Graphic,
        step_scheduler: StepScheduler,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray
) -> np.ndarray:
    points = np.ndarray(shape=(0, 2), dtype=float, order='F')
    points = np.append(points, [current_position], axis=0)
    for i in range(1, max_epoch):
        G = function.grad(current_position)
        step = step_scheduler.step(i)
        next_position = one_dimension.point(function, current_position, step, G, eps)
        points = np.append(points, [next_position], axis=0)
        if np.linalg.norm(G) < eps:
            return points
        current_position = next_position
    return points


def find_best_learning_rate_gradient_steps(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray
) -> float:
    t = np.linspace(0.01, 1, 100)
    epoches = np.array(
        [(gradient_steps(function, RateStep(lr), one_dimension, max_epoch, eps, current_position).size // 2) for lr in
         t])
    return t[epoches.argmin()]


def fund_best_schedule(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray
):
    const_steps = np.linspace(0.01, 1, 100)
    epoches_const = np.array(
        [(gradient_steps(function, RateStep(lr), one_dimension, max_epoch, eps, current_position).size // 2) for lr in
         const_steps])
    best_const = epoches_const.argmin()
    linear_starts = np.linspace(0.01, 0.8, 80)
    linear_steps = np.linspace(0.97, 0.999, 30)
    epoches_linear = np.array(
        [[gradient_steps(function, LinearStep(ls, step), one_dimension, max_epoch, eps, current_position)
              .size // 2 for ls in linear_starts] for step in
         linear_steps])
    best_linear = np.unravel_index(epoches_linear.argmin(), epoches_linear.shape)
    parabaloic_k = np.linspace(1, 300, 300)
    epoches_parabaloic = np.array(
        [(gradient_steps(function, ParabolicStep(k), one_dimension, max_epoch, eps, current_position).size // 2) for k
         in
         parabaloic_k])
    best_parabaloic = epoches_parabaloic.argmin()
    print(f"Best solutions:\n"
          f"Const: Rate={const_steps[best_const]} with {epoches_const[best_const]} steps\n"
          f"Linear: Start={linear_starts[best_linear[1]]}, Step={linear_steps[best_linear[0]]} with {epoches_linear[best_linear]} steps\n"
          f"Parabaloic: K={parabaloic_k[best_parabaloic]} with {epoches_parabaloic[best_parabaloic]} steps")
