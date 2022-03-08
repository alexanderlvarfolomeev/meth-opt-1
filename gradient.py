import numpy as np
from numpy import ndarray

import steps
from drawing import draw1d, draw2d
from graphic import Graphic, RandomGraphic
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
    points = np.ndarray(shape=(0, current_position.size), dtype=float, order='F')
    points = np.append(points, [current_position], axis=0)
    for i in range(1, max_epoch):
        grad = function.grad(current_position)
        step = step_scheduler.step(i, function, current_position, grad)
        next_position = one_dimension.point(function, current_position, step, grad, eps)
        points = np.append(points, [next_position], axis=0)
        if np.linalg.norm(grad) < eps:
            return points
        current_position = next_position
    return points


def binary_rises(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        eps: float,
        current_position: ndarray,
        limit: int,
        def_fun
):
    top = 10
    while top < limit:
        if def_fun(function, one_dimension, top, eps, current_position, False) is not None:
            return def_fun(function, one_dimension, top * 10, eps, current_position, False)
        top *= 2
    return None


def find_best_learning_rate_gradient_steps(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray
) -> float:
    t = np.linspace(0.01, 1, 100)
    epoches = np.array(
        [(gradient_steps(function, RateStep(lr), one_dimension, max_epoch, eps, current_position).shape[0]) for lr in
         t])
    return t[epoches.argmin()]


def find_best_const_schedule(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray,
        drawing: bool
):
    const_steps = np.linspace(0.01, 1, 70)
    epoches_const = np.array(
        [(gradient_steps(function, RateStep(lr), one_dimension, max_epoch, eps, current_position).shape[0]) for lr in
         const_steps])
    best_const = epoches_const.argmin()

    if drawing:
        print(f"Const: Rate={const_steps[best_const]} with {epoches_const[best_const]} steps")
        draw1d(const_steps, epoches_const, best_const)

    if epoches_const[best_const] != max_epoch:
        return steps.RateStep(const_steps[best_const])
    else:
        return None


def find_best_linear_schedule(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray,
        drawing: bool
):
    linear_starts = np.linspace(0.01, 0.8, 20)
    linear_steps = np.linspace(0.97, 0.999, 10)
    epoches_linear = np.array(
        [[gradient_steps(function, LinearStep(ls, step), one_dimension, max_epoch, eps, current_position)
              .shape[0] for ls in linear_starts] for step in
         linear_steps])
    best_linear = np.unravel_index(epoches_linear.argmin(), epoches_linear.shape)

    if drawing:
        print(
            f"Linear: Start={linear_starts[best_linear[1]]}, Step={linear_steps[best_linear[0]]} with {epoches_linear[best_linear]} steps")
        draw2d(linear_starts, linear_steps, epoches_linear, best_linear)

    if epoches_linear[best_linear] != max_epoch:
        return steps.LinearStep(linear_starts[best_linear[1]], linear_steps[best_linear[0]])
    else:
        return None


def find_best_parabolic_schedule(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray,
        drawing: bool
):
    parabaloic_k = np.linspace(1, 3000, 70)
    epoches_parabaloic = np.array(
        [(gradient_steps(function, ParabolicStep(k), one_dimension, max_epoch, eps, current_position).shape[0]) for k
         in
         parabaloic_k])
    best_parabaloic = epoches_parabaloic.argmin()

    if drawing:
        print(f"Parabaloic: K={parabaloic_k[best_parabaloic]} with {epoches_parabaloic[best_parabaloic]} steps")
        draw1d(parabaloic_k, epoches_parabaloic, best_parabaloic)

    if epoches_parabaloic[best_parabaloic] != max_epoch:
        return steps.ParabolicStep(parabaloic_k[best_parabaloic])
    else:
        return None

def find_best_wolf_schedule(
        function: Graphic,
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        eps: float,
        current_position: ndarray,
        drawing: bool
):
    wolf_c1 = np.linspace(0.001, 0.999, 10)
    wolf_c2 = np.linspace(0.001, 0.999, 10)
    epoches_wolf = np.array(
        [[max_epoch
          if (c1 >= c2)
          else gradient_steps(function, steps.WolfSteps(c1, c2), one_dimension, max_epoch, eps, current_position).shape[0]
          for c1 in wolf_c1] for c2 in wolf_c2])
    best_wolf = np.unravel_index(epoches_wolf.argmin(), epoches_wolf.shape)

    if drawing:
        print(
            f"Wolfe: C1={wolf_c1[best_wolf[1]]}, C2={wolf_c2[best_wolf[0]]} with {epoches_wolf[best_wolf]} steps")
        draw2d(wolf_c1, wolf_c2, epoches_wolf, best_wolf)

    if epoches_wolf[best_wolf] != max_epoch:
        return steps.WolfSteps(wolf_c1[best_wolf[1]], wolf_c2[best_wolf[0]])
    else:
        return None

def test_different_argument_count(
        one_dimension: OneDimensionSearch,
        max_epoch: int,
        avg_count: int,
        eps: float
):
    arg_counts = np.linspace(1, 10, 10)
    epoches = np.array(
        [np.average(
            [(gradient_steps(
                RandomGraphic(round(arg_count)),
                RateStep(0.19),
                one_dimension,
                max_epoch,
                eps,
                np.ones(round(arg_count)) * 20
            ).size // arg_count)
             for _ in range(avg_count)]
        )
            for arg_count in arg_counts]
    )

    draw1d(arg_counts, epoches)
