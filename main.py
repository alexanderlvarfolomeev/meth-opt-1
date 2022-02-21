import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)


def f(x, y):
    return x ** 2 + 4 * y ** 2 - 3 * x * y


def grad(x, y):
    return [2 * x - 3 * y, 8 * y - 3 * x]


def g(x, y):
    return x ** 2 + 4 * y ** 2 - 3 * x * y + 5 * x - y


def grad_g(x, y):
    return [2 * x - 3 * y + 5, 8 * y - 3 * x - 1]


def norm(a, b):
    return (a ** 2 + b ** 2) ** 0.5


def dist(a, b):
    return norm(a[0] - b[0], a[1] - b[1])


def with_const_lr(lr, max_epoch, eps=1e-7):
    xy = [-15, -15]

    for i in range(1, max_epoch):
        G = grad(xy[0], xy[1])
        new_xy = xy - lr * np.array(G)
        if norm(G[0], G[1]) < eps:
            return i
        xy = new_xy

    return max_epoch


# Part #1
def test_const_lr(max_epoch=1000):
    t = np.linspace(0.01, 1, 100)
    epoches = np.array([with_const_lr(lr, max_epoch) for lr in t])

    idx = epoches.argmin()

    print(f"With learning rate: {t[idx]}\nCan achieve minimum on {epoches[idx]} epoches")

    l_bound = idx
    while l_bound >= 0 and epoches[l_bound] != max_epoch:
        l_bound -= 1
    l_bound = max(l_bound, 0)
    u_bound = idx
    while u_bound < len(epoches) and epoches[u_bound] != max_epoch:
        u_bound += 1
    print(f"Descent reaches minimum on interval [{t[l_bound]};{t[u_bound] if u_bound < len(epoches) else 1}).")

    plt.plot(t, epoches)
    plt.show()

    return t[idx]


def with_mul_lr(lr, step, max_epoch, eps=1e-7):
    xy = [-15, -15]

    for i in range(1, max_epoch):
        G = grad(xy[0], xy[1])
        new_xy = xy - lr * np.array(G)
        if norm(G[0], G[1]) < eps:
            return i
        if lr >= 0.01:
            lr *= step
        xy = new_xy

    return max_epoch


# Part #2
def test_mut_lr(max_epoch=1000):
    t = np.linspace(0.01, 0.8, 80)
    r = np.linspace(0.97, 0.999, 30)
    epoches = np.array([[with_mul_lr(lr, step, max_epoch) for lr in t] for step in r])

    idx = np.unravel_index(epoches.argmin(), epoches.shape)

    print(idx)

    print(f"With learning rate: {t[idx[1]]}\n"
          f"With step: {r[idx[0]]}\n"
          f"Can achieve minimum on {epoches[idx]} epoches")

    X, Y = np.meshgrid(t, r)
    plt.plot(t[idx[1]], r[idx[0]], marker="o", markerfacecolor="red")
    plt.contourf(X, Y, epoches)
    plt.show()



def main():
    t = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(t, t)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(X, Y, f(X, Y))

    lr = 0.15
    epoch = 1000
    xy = [-15, -15]

    points = np.zeros((epoch, 2))
    points[0] = xy
    for i in range(1, epoch):
        xy = xy - lr * np.array(grad(xy[0], xy[1]))
        points[i] = xy

    print(points)
    plt.plot(points[:, 0], points[:, 1], "o-")
    plt.contour(X, Y, f(X, Y), levels=sorted(list(set([f(p[0], p[1]) for p in points] + list(np.linspace(-1, 1, 100))))))
    plt.show()


if __name__ == '__main__':
    # main()
    test_const_lr()
    test_mut_lr()
