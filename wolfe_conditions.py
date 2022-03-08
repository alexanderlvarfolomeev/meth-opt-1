import imageio
import numpy as np
from matplotlib import pyplot as plt

from gradient import gradient_steps
from graphic import F
from onedimsearch import WolfeCondition
from steps import RateStep


def wolfe_analyzing(
        create_gif: bool):
    len_x = 10
    len_y = 10
    len_y_add = 5
    c_1_array = np.linspace(1e-7, 0.8, len_x)  # 10000000
    c_2_array = np.linspace(0.1, 0.9, len_y)
    len_y += len_y_add
    c_2_array = np.append(c_2_array, np.linspace(0.91, 0.99, len_y_add))
    eps = 1e-15
    iter_num = 10000
    start_pos = np.array([15, 15])
    X, Y = np.meshgrid(c_1_array, c_2_array)
    print('creating analyzing data for F() function, c_1 in range [1e-7, 0.8], c_2 in range [0.1, 0.99]')
    Zs = np.array([
        0 if c_1 > c_2 else len(
            gradient_steps(F(), RateStep(0.1), WolfeCondition(c_1, c_2), iter_num, eps, start_pos)
        )
        for c_1, c_2 in zip(np.ravel(X), np.ravel(Y))])
    print('got analyzed data')
    Z = Zs.reshape(len_y, len_x)
    [min_y, min_x] = np.unravel_index(Z[np.nonzero(Z)].argmin(), Z.shape)
    min_c_2 = c_2_array[min_y]
    min_c_1 = c_1_array[min_x]
    min_val = Z[np.nonzero(Z)].min()
    print(f'minimal iteration number with c_1={min_c_1} c_2={min_c_2} iteration={min_val}')
    if create_gif:
        fig = plt.figure(figsize=(6, 6))  # @100 dpi it's 500×300 pixels
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('c_1')
        ax.set_ylabel('c_2')
        ax.set_zlabel('iteration number')
        images = []
        print('creating plots for gif')
        for angle in range(0, 360, 5):
            ax.view_init(30, angle)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images.append(image.reshape(600, 600, 3))
        imageio.mimsave('wolfe_condition_analysis.gif', images)
        print('results saved')


if __name__ == '__main__':
    wolfe_analyzing(True)
