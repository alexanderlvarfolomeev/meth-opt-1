import numpy as np
from matplotlib import pyplot as plt

import gradient
import graphic
import onedimsearch
import steps


def _01():
    gold_split_const_best = gradient.find_best_const_schedule(graphic.F(), onedimsearch.GoldSplit(), 1000, 1e-5,
                                                              np.array([15, 15]),
                                                              False)
    gold_split_linear_best = gradient.find_best_linear_schedule(graphic.F(), onedimsearch.GoldSplit(), 1000, 1e-5,
                                                                np.array([15, 15]), False)
    gold_split_parabolic_best = gradient.find_best_parabolic_schedule(graphic.F(), onedimsearch.GoldSplit(), 1000, 1e-5,
                                                                      np.array([15, 15]), False)

    print(steps.RateStep(gold_split_const_best), steps.LinearStep(*gold_split_linear_best),
          steps.ParabolicStep(gold_split_parabolic_best))


def _02():
    eps = 1e-4

    for search in [onedimsearch.StepByStep(), onedimsearch.GoldSplit()]:
        for g in [graphic.F(), graphic.G(), graphic.RandomGraphic(2)]:
            print(search.__class__, g.__class__)
            for type_best in [gradient.find_best_const_schedule, gradient.find_best_linear_schedule,
                              gradient.find_best_parabolic_schedule]:
                gradient.binary_rises(
                    g,
                    search,
                    eps,
                    np.array([15, 15]),
                    100000,
                    type_best
                )

def _03():
    graphics = []

    for _ in range(5):
        graphics.append(graphic.RandomGraphic(2))

    for g in graphics:
        g.draw()
        for search_method in [onedimsearch.StepByStep(), onedimsearch.GoldSplit()]:
            print(search_method.__class__)
            for type_best in [gradient.find_best_const_schedule, gradient.find_best_linear_schedule,
                              gradient.find_best_parabolic_schedule]:
                print(type_best.__name__)
                best = gradient.binary_rises(
                    g,
                    search_method,
                    1e-5,
                    np.array([15, 15]),
                    10000,
                    type_best
                )
                points = gradient.gradient_steps(g, best, search_method, 10000, 1e-5, np.array([15, 15]))
                x = points[:, 0]
                y = points[:, 1]
                x_lin = np.linspace(x.min(), x.max(), 100)
                y_lin = np.linspace(y.min(), y.max(), 100)
                X, Y = np.meshgrid(x_lin, y_lin)
                plt.plot(points[:, 0], points[:, 1], "o-")
                # plt.contour(X, Y, levels=sorted(list(set([g(np.array([p[0], p[1]])) for p in points] + list(
                # np.linspace(-1, 1, 100))))))
                plt.show()


if __name__ == '__main__':
    _03()
